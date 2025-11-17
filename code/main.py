import os
import time
import argparse
import numpy as np
import random
import logging
import sys
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: WandB not available, continuing without logging")

import model
import evaluate
import data_utils
from loss import HybridPLLLoss, LossScheduleConfig
from loss_tracker import LossTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon_book',
                        help='dataset name (e.g., amazon_book_denoised_v2)')
    parser.add_argument('--model', type=str, default='NeuMF-end',
                        help='model used for training')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--factor_num', type=int, default=32, help='latent factor number')
    parser.add_argument('--num_layers', type=int, default=3, help='MLP depth')
    parser.add_argument('--num_ng', type=int, default=1, help='negative samples per positive')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--out', action='store_true', help='whether to save model checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--track_loss', action='store_true', help='enable TP/FP loss tracking')
    parser.add_argument('--track_freq', type=int, default=50, help='loss tracking frequency')

    # PLL schedule options
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='epochs with BCE-only warmup')
    parser.add_argument('--transition-epochs', type=int, default=3,
                        help='epochs for PLL ramp-up after warmup')
    parser.add_argument('--pll-alpha', type=float, default=0.5,
                        help='target PLL weight after transition')
    parser.add_argument('--tau-start', type=float, default=1.5,
                        help='initial temperature for PLL softmax')
    parser.add_argument('--tau-end', type=float, default=0.5,
                        help='final temperature for PLL softmax')
    parser.add_argument('--candidate-momentum', type=float, default=0.9,
                        help='momentum factor for candidate updates')
    parser.add_argument('--min-pos-weight', type=float, default=0.2,
                        help='minimum BCE weight for positive samples')
    parser.add_argument('--noise-margin', type=float, default=0.0,
                        help='required gap (fp_noise - tp_noise); <=0 disables extra penalty')
    parser.add_argument('--noise-margin-weight', type=float, default=0.0,
                        help='scale of the noise-margin penalty term')
    parser.add_argument('--use_reverse_schedule', action='store_true',
                        help='reverse the alpha schedule (PLL first, BCE later)')
    parser.add_argument('--disable-momentum-update', action='store_true',
                        help='disable momentum blending for candidate updates')
    parser.add_argument('--fixed-temperature', action='store_true',
                        help='keep PLL temperature constant throughout training')
    parser.add_argument('--no-weighted-bce', action='store_true',
                        help='disable PLL-weighted BCE (use standard BCE)')
    parser.add_argument('--no-warmup', action='store_true',
                        help='disable BCE warmup (PLL active from epoch 0)')
    parser.add_argument('--pll-reweight-eval', action='store_true',
                        help='evaluate additional metrics with PLL reweighting/filtering on test set')

    parser.add_argument('--separate_pll_embeddings', action='store_true',
                        help='use separate embeddings for PLL head')
    parser.add_argument('--pll_hidden_dim', type=int, default=128,
                        help='hidden dimension for PLL head')
    parser.add_argument('--init_noise_prob', type=float, default=0.3,
                        help='initial noise probability for FP samples')
    parser.add_argument('--top_k', type=int, nargs='+', default=[5, 10, 20, 50],
                        help='top-K cutoffs for evaluation metrics')

    args = parser.parse_args()
    return args


def set_random_seed():
    torch.manual_seed(2019)
    np.random.seed(2019)
    random.seed(2019)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2019)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def resolve_data_path(dataset: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    release_data_root = os.path.join(base_dir, '..', 'datasets')
    primary_names = {'amazon_book', 'amazon_movie', 'ml_100k'}
    if dataset in primary_names:
        candidate = os.path.join(release_data_root, dataset)
        if os.path.exists(candidate):
            root = candidate
        else:
            root = os.path.join(base_dir, '..', 'data', dataset)
    elif dataset.endswith('_denoised') or dataset.endswith('_denoised_v2') or dataset.startswith('amazon_book_denoised') or dataset.startswith('ml_1m_denoised'):
        candidate = os.path.join(release_data_root, dataset)
        if os.path.exists(candidate):
            root = candidate
        else:
            root = os.path.join(base_dir, '..', 'data_28Oct', dataset)
    elif 'large_scale_pll' in dataset or 'medium_pll' in dataset:
        root = os.path.join(base_dir, '..', 'data_1sep', dataset, 'multilevel')
    elif 'v5' in dataset:
        root = os.path.join(base_dir, '..', 'data_1sep', dataset, 'normal')
    else:
        root = os.path.join(base_dir, '..', 'data', dataset)
    return os.path.abspath(root)


def init_logging(args) -> logging.Logger:
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_id = os.path.basename(os.path.normpath(args.dataset))
    log_filename = f'{log_dir}/{dataset_id}_{args.model}_alpha{args.pll_alpha}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"log file: {log_filename}")
    return logger


def init_wandb(args, data_path):
    if not args.use_wandb or not WANDB_AVAILABLE:
        return False
    try:
        dataset_id = os.path.basename(os.path.normpath(args.dataset))
        wandb.init(
            project="baseline-comparison",
            name=f"PLL31_{dataset_id}_{args.model}_alpha{args.pll_alpha}",
            config={
                "dataset": dataset_id,
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "pll_alpha": args.pll_alpha,
                "warmup_epochs": args.warmup_epochs,
                "transition_epochs": args.transition_epochs,
                "tau_start": args.tau_start,
                "tau_end": args.tau_end,
                "candidate_momentum": args.candidate_momentum,
                "data_path": data_path,
            }
        )
        return True
    except Exception as exc:  # pragma: no cover - wandb optional
        print(f"Warning: WandB initialization failed: {exc}")
        return False


def stage_freeze_pll(model_instance: model.NCF_PLL, trainable: bool):
    model_instance.set_pll_trainable(trainable)


def audit_candidates(dataset: data_utils.NCFDataPLL) -> Dict[str, float]:
    candidates = dataset.candidate_labels.cpu()
    labels = torch.tensor(dataset.noisy_or_not, dtype=torch.long)
    tp_mask = labels == 1
    fp_mask = labels == 0

    stats = {}
    if tp_mask.sum() > 0:
        tp_pos = candidates[tp_mask, 0]
        tp_noise = candidates[tp_mask, 2]
        stats["tp_pos_mean"] = float(tp_pos.mean().item())
        stats["tp_noise_mean"] = float(tp_noise.mean().item())
        stats["tp_pos_gt_0.7"] = float((tp_pos > 0.7).float().mean().item())
        stats["tp_noise_lt_0.3"] = float((tp_noise < 0.3).float().mean().item())
    if fp_mask.sum() > 0:
        fp_pos = candidates[fp_mask, 0]
        fp_noise = candidates[fp_mask, 2]
        stats["fp_pos_mean"] = float(fp_pos.mean().item())
        stats["fp_noise_mean"] = float(fp_noise.mean().item())
        stats["fp_noise_gt_0.7"] = float((fp_noise > 0.7).float().mean().item())
        stats["fp_pos_lt_0.3"] = float((fp_pos < 0.3).float().mean().item())
    stats["tp_count"] = int(tp_mask.sum().item())
    stats["fp_count"] = int(fp_mask.sum().item())
    return stats


def evaluate_pll_head(net: model.NCF_PLL, dataset: data_utils.NCFDataPLL,
                      device: torch.device, sample_size: int = 8000, batch_size: int = 2048) -> Dict[str, float]:
    total = len(dataset.features_ps)
    if total == 0:
        return {
            'sample_size': 0,
            'accuracy': 0.0,
            'tp_recall': 0.0,
            'fp_detect_rate': 0.0,
            'tp_noise_mean': 0.0,
            'fp_noise_mean': 0.0,
            'tp_noise': np.array([]),
            'fp_noise': np.array([]),
        }

    sample_size = min(sample_size, total)
    indices = torch.randperm(total)[:sample_size]
    labels = torch.tensor(dataset.noisy_or_not, dtype=torch.long)[indices]

    users = torch.tensor([dataset.features_ps[i][0] for i in indices], dtype=torch.long)
    items = torch.tensor([dataset.features_ps[i][1] for i in indices], dtype=torch.long)

    probs_list = []
    with torch.no_grad():
        net.eval()
        for start in range(0, sample_size, batch_size):
            end = start + batch_size
            u = users[start:end].to(device)
            it = items[start:end].to(device)
            _, pll_logits = net(u, it, return_both=True)
            pll_probs = F.softmax(pll_logits, dim=1)
            probs_list.append(pll_probs.cpu())

    pll_probs_all = torch.cat(probs_list, dim=0)
    preds = (pll_probs_all[:, 0] > pll_probs_all[:, 2]).long()
    label_pos = (labels == 1).long()

    accuracy = (preds == label_pos).float().mean().item()
    tp_mask = labels == 1
    fp_mask = labels == 0

    tp_recall = float(((preds == 1) & tp_mask).float().sum() / max(tp_mask.sum(), torch.tensor(1)))
    fp_detect = float(((preds == 0) & fp_mask).float().sum() / max(fp_mask.sum(), torch.tensor(1)))

    tp_noise = pll_probs_all[tp_mask, 2].numpy() if tp_mask.any() else np.array([])
    fp_noise = pll_probs_all[fp_mask, 2].numpy() if fp_mask.any() else np.array([])

    return {
        'sample_size': int(sample_size),
        'accuracy': accuracy,
        'tp_recall': tp_recall,
        'fp_detect_rate': fp_detect,
        'tp_noise_mean': float(np.mean(tp_noise)) if tp_noise.size else 0.0,
        'fp_noise_mean': float(np.mean(fp_noise)) if fp_noise.size else 0.0,
        'tp_noise': tp_noise,
        'fp_noise': fp_noise,
    }


def save_noise_hist(tp_noise: np.ndarray, fp_noise: np.ndarray, path: str) -> None:
    if tp_noise.size == 0 or fp_noise.size == 0:
        return
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, 1, 40)
    plt.hist(tp_noise, bins=bins, alpha=0.6, label='TP P(noise)', color='green', density=True)
    plt.hist(fp_noise, bins=bins, alpha=0.6, label='FP P(noise)', color='red', density=True)
    plt.xlabel('P(noise)')
    plt.ylabel('Density')
    plt.title('PLL noise probability distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_random_seed()
    logger = init_logging(args)
    data_path = resolve_data_path(args.dataset)
    model_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', args.dataset))
    os.makedirs(model_root, exist_ok=True)
    checkpoint_path = os.path.join(model_root, f"{args.model}_alpha{args.pll_alpha}.pth")

    logger.info(f"Using device: {device}")
    logger.info(f"arguments: {args}")
    logger.info(f"config model: {args.model}")
    logger.info(f"config data path: {data_path}")
    logger.info(f"config model path: {model_root}")

    wandb_active = init_wandb(args, data_path)

    # Load data
    train_data, valid_data, test_data_pos, user_pos, user_num, item_num, train_mat, train_noisy = data_utils.load_all(args.dataset, data_path)
    test_data_fp = getattr(data_utils.load_all, 'last_test_fp', {})
    logger.info("data loaded! user_num:%d, item_num:%d train_data_len:%d test_user_num:%d",
                user_num, item_num, len(train_data), len(test_data_pos))
    if test_data_fp:
        fp_interactions = sum(len(items) for items in test_data_fp.values())
        logger.info("test FP users:%d fp_interactions:%d", len(test_data_fp), fp_interactions)

    train_dataset = data_utils.NCFDataPLL(
        train_data, item_num, train_mat, args.num_ng, 0, train_noisy,
        init_noise_prob=args.init_noise_prob)
    valid_dataset = data_utils.NCFDataPLL(valid_data, item_num, train_mat, args.num_ng, 1)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(2019 + worker_id)
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Build model
    net = model.NCF_PLL(
        user_num, item_num, args.factor_num, args.num_layers,
        args.dropout, args.model, pll_hidden_dim=args.pll_hidden_dim,
        separate_pll_embeddings=args.separate_pll_embeddings
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    schedule_cfg = LossScheduleConfig(
        total_epochs=args.epochs,
        warmup_epochs=0 if args.no_warmup else args.warmup_epochs,
        transition_epochs=args.transition_epochs,
        final_alpha=args.pll_alpha,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        min_positive_weight=args.min_pos_weight,
        candidate_momentum=0.0 if args.disable_momentum_update else args.candidate_momentum,
        use_reverse_schedule=args.use_reverse_schedule,
        fixed_temperature=args.fixed_temperature,
        use_weighted_bce=not args.no_weighted_bce,
        noise_margin=args.noise_margin,
        noise_margin_weight=args.noise_margin_weight,
    )
    loss_manager = HybridPLLLoss(schedule_cfg)

    tracking_dir = os.path.join('loss_tracking', args.dataset)
    os.makedirs(tracking_dir, exist_ok=True)

    loss_tracker = None
    if args.track_loss:
        loss_tracker = LossTracker()
        logger.info("Loss tracking enabled")

    global_step = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        stage_freeze_pll(net, trainable=(epoch >= schedule_cfg.warmup_epochs))

        net.train()
        train_loader.dataset.ng_sample()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_pll = 0.0
        batch_count = 0

        for batch_idx, (user, item, label, noisy_or_not, candidate, idx) in enumerate(train_loader):
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)
            candidate = candidate.to(device)

            optimizer.zero_grad()
            bce_pred, pll_logits = net(user, item, return_both=True)
            stats = loss_manager(bce_pred, pll_logits, label, candidate, noisy_or_not.to(device), epoch)
            stats['loss'].backward()
            optimizer.step()

            momentum = 0.0 if args.disable_momentum_update else args.candidate_momentum
            train_loader.dataset.update_candidate_labels(idx, stats['updated_candidates'], momentum=momentum)

            epoch_loss += stats['loss'].item()
            epoch_bce += stats.get('bce_loss', 0.0)
            epoch_pll += stats.get('pll_loss', 0.0)
            batch_count += 1
            global_step += 1

            if loss_tracker and batch_count % args.track_freq == 0:
                loss_tracker.record(global_step,
                                     float(stats['pll_pos_loss']),
                                     float(stats['pll_noise_loss']),
                                     float(stats['bce_loss']))

            if wandb_active and batch_count % args.track_freq == 0:
                wandb.log({
                    'train_loss_step': stats['loss'].item(),
                    'pll_alpha': stats['alpha'],
                    'pll_temperature': stats['temperature'],
                    'pll_pos_loss': stats['pll_pos_loss'],
                    'pll_noise_loss': stats['pll_noise_loss'],
                }, step=global_step)

        epoch_time = time.time() - epoch_start
        denom = max(1, batch_count)
        avg_loss = epoch_loss / denom
        avg_bce = epoch_bce / denom
        avg_pll = epoch_pll / denom
        logger.info(
            "Epoch %d completed - Avg Loss: %.6f (BCE %.6f | PLL %.6f), Time: %.2fs, alpha=%.3f, tau=%.3f",
            epoch + 1,
            avg_loss,
            avg_bce,
            avg_pll,
            epoch_time,
            loss_manager.current_alpha,
            loss_manager.current_temperature,
        )
        if wandb_active:
            wandb.log({
                'train_loss_epoch': avg_loss,
                'train_bce_epoch': avg_bce,
                'train_pll_epoch': avg_pll,
                'train_alpha_epoch': loss_manager.current_alpha,
                'train_tau_epoch': loss_manager.current_temperature,
            }, step=global_step)

        # Validation (per epoch)
        net.eval()
        with torch.no_grad():
            precision, recall, ndcg, _ = evaluate.test_all_users(
                net, 4096, item_num, test_data_pos, user_pos, args.top_k, test_data_fp=test_data_fp)
        metric_groups = {
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
        }
        for name, values in metric_groups.items():
            detail = ", ".join(f"@{k}: {values[idx]:.6f}" for idx, k in enumerate(args.top_k))
            logger.info("Validation %s %s", name, detail)

        audit_stats = audit_candidates(train_dataset)
        pll_eval = evaluate_pll_head(net, train_dataset, device)
        logger.info("Candidate audit: %s", audit_stats)
        logger.info("PLL head accuracy %.4f, TP recall %.4f, FP detect %.4f",
                    pll_eval['accuracy'], pll_eval['tp_recall'], pll_eval['fp_detect_rate'])

        hist_path = os.path.join(tracking_dir, f'pll_noise_epoch{epoch+1}.png')
        save_noise_hist(pll_eval['tp_noise'], pll_eval['fp_noise'], hist_path)

        if wandb_active:
            wandb_val_metrics = {
                f"val_{name}@{k}": float(values[idx])
                for name, values in metric_groups.items()
                for idx, k in enumerate(args.top_k)
            }
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                **wandb_val_metrics,
                'audit/tp_pos_mean': audit_stats.get('tp_pos_mean', 0.0),
                'audit/fp_noise_mean': audit_stats.get('fp_noise_mean', 0.0),
                'pll_eval/accuracy': pll_eval['accuracy'],
                'pll_eval/tp_recall': pll_eval['tp_recall'],
                'pll_eval/fp_detect': pll_eval['fp_detect_rate'],
            }, step=global_step)

    # Save final model
    if args.out:
        torch.save(net.state_dict(), checkpoint_path)
        logger.info("Model saved to %s", checkpoint_path)

    # Final evaluation on test split
    net.eval()
    with torch.no_grad():
        precision, recall, ndcg, _ = evaluate.test_all_users(
            net, 4096, item_num, test_data_pos, user_pos, args.top_k, test_data_fp=test_data_fp)
    logger.info("################### FINAL TEST ######################")
    metric_groups = {
        'precision': precision,
        'recall': recall,
        'ndcg': ndcg,
    }
    for name, values in metric_groups.items():
        detail = ", ".join(f"@{k}: {values[idx]:.6f}" for idx, k in enumerate(args.top_k))
        logger.info("%s %s", name.capitalize(), detail)

    results = {
        key: [float(v) for v in values]
        for key, values in metric_groups.items()
    }
    fp_metrics = getattr(evaluate.test_all_users, 'last_fp_metrics', None)
    if fp_metrics is not None:
        results['fp_at_k'] = [float(v) for v in fp_metrics]
        fp_detail = ", ".join(f"@{k}: {fp_metrics[idx]:.6f}" for idx, k in enumerate(args.top_k))
        logger.info("FP@K %s", fp_detail)

    if wandb_active:
        wandb_payload = {
            f"final_{name}@{k}": float(values[idx])
            for name, values in metric_groups.items()
            for idx, k in enumerate(args.top_k)
        }
        if fp_metrics is not None:
            wandb_payload.update({
                f"final_fp@{k}": float(fp_metrics[idx])
                for idx, k in enumerate(args.top_k)
            })
        wandb.log(wandb_payload)

    if args.pll_reweight_eval:
        from evaluate_pll import test_all_users_with_pll
        with torch.no_grad():
            pll_prec, pll_rec, pll_ndcg, _ = test_all_users_with_pll(
                net, 4096, item_num, test_data_pos, user_pos, args.top_k,
                use_pll_reweight=True, test_data_fp=test_data_fp)
        pll_metric_groups = {
            'precision': pll_prec,
            'recall': pll_rec,
            'ndcg': pll_ndcg,
        }
        for name, values in pll_metric_groups.items():
            detail = ", ".join(f"@{k}: {values[idx]:.6f}" for idx, k in enumerate(args.top_k))
            logger.info("PLL reweighted %s %s", name, detail)
        if wandb_active:
            wandb.log({
                f"pll_reweight/{name}@{k}": float(values[idx])
                for name, values in pll_metric_groups.items()
                for idx, k in enumerate(args.top_k)
            })

    if loss_tracker:
        try:
            tracking_dir = os.path.join('loss_tracking', args.dataset)
            os.makedirs(tracking_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tracking_file = os.path.join(tracking_dir, f'{args.model}_alpha{args.pll_alpha}_{timestamp}.json')
            loss_tracker.save(tracking_file)
            logger.info("Loss tracker data saved to %s", tracking_file)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to save loss tracker data: {exc}")

    logger.info("Final Results: %s", results)

    if wandb_active:
        wandb.finish()


if __name__ == '__main__':
    main()
