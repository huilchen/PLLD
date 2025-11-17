import torch
import torch.nn.functional as F
from dataclasses import dataclass


def _linear_schedule(epoch: int, start_epoch: int, end_epoch: int, start_val: float, end_val: float) -> float:
    if end_epoch <= start_epoch:
        return end_val
    if epoch <= start_epoch:
        return start_val
    if epoch >= end_epoch:
        return end_val
    progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return start_val + progress * (end_val - start_val)


@dataclass
class LossScheduleConfig:
    total_epochs: int
    warmup_epochs: int = 3
    transition_epochs: int = 3
    final_alpha: float = 0.5
    tau_start: float = 1.5
    tau_end: float = 0.5
    min_positive_weight: float = 0.2
    candidate_momentum: float = 0.9
    use_reverse_schedule: bool = False
    fixed_temperature: bool = False
    use_weighted_bce: bool = True
    noise_margin: float = 0.0
    noise_margin_weight: float = 0.0


class HybridPLLLoss:
    """Hybrid BCE + PLL loss with staged training, temperature annealing and momentum updates."""

    def __init__(self, cfg: LossScheduleConfig):
        self.cfg = cfg
        self._alpha = 0.0
        self._temperature = cfg.tau_start

    @property
    def current_alpha(self) -> float:
        return self._alpha

    @property
    def current_temperature(self) -> float:
        return self._temperature

    def _update_schedules(self, epoch: int) -> None:
        warmup = self.cfg.warmup_epochs
        transition = self.cfg.transition_epochs
        final_alpha = self.cfg.final_alpha

        if epoch < warmup:
            alpha = 0.0
        elif epoch < warmup + transition:
            alpha = _linear_schedule(epoch, warmup, warmup + transition, 0.0, final_alpha)
        else:
            alpha = final_alpha

        if self.cfg.use_reverse_schedule:
            # Reverse schedule: start with PLL, end with mixture
            epoch_clamped = min(max(epoch, 0), self.cfg.total_epochs)
            alpha = _linear_schedule(
                epoch_clamped,
                0,
                self.cfg.total_epochs,
                final_alpha,
                0.0,
            )

        tau = _linear_schedule(epoch, 0, self.cfg.total_epochs, self.cfg.tau_start, self.cfg.tau_end)
        if self.cfg.fixed_temperature:
            tau = self.cfg.tau_start

        self._alpha = float(alpha)
        self._temperature = float(tau)

    def __call__(self, bce_logits: torch.Tensor, pll_logits: torch.Tensor,
                 targets: torch.Tensor, candidates: torch.Tensor,
                 noisy_mask: torch.Tensor, epoch: int) -> dict:
        """Compute hybrid loss and return diagnostics."""
        self._update_schedules(epoch)
        alpha = self._alpha
        temperature = self._temperature

        eps = 1e-8
        labels = targets.float()
        noisy_mask = noisy_mask.float()

        # Temperature-controlled softmax for PLL branch
        pll_probs = F.softmax(pll_logits / temperature, dim=1)

        # Weighted BCE: positives scaled by PLL P(positive)
        if self.cfg.use_weighted_bce:
            weights = torch.ones_like(labels)
            pos_mask = labels > 0.5
            if pos_mask.any():
                positive_conf = pll_probs[:, 0].detach()
                min_w = self.cfg.min_positive_weight
                weights[pos_mask] = torch.clamp(positive_conf[pos_mask], min=min_w, max=1.0)
            bce_loss = F.binary_cross_entropy_with_logits(bce_logits, labels, weight=weights, reduction='mean')
        else:
            weights = torch.ones_like(labels)
            bce_loss = F.binary_cross_entropy_with_logits(bce_logits, labels, reduction='mean')

        # PLL loss with correctional handling of negative components
        pll_loss_terms = -(candidates * torch.log(pll_probs + eps))  # [batch, 3]
        component_means = pll_loss_terms.mean(dim=0)
        positive_part = torch.clamp(component_means, min=0.0)
        negative_part = torch.clamp(-component_means, min=0.0)
        pll_loss = positive_part.sum() - negative_part.sum()
        correction = float(negative_part.sum().item())

        total_loss = (1.0 - alpha) * bce_loss + alpha * pll_loss

        margin_term_value = 0.0
        if self.cfg.noise_margin_weight > 0.0:
            pos_mask = labels > 0.5
            tp_mask = pos_mask & (noisy_mask > 0.5)
            fp_mask = pos_mask & (noisy_mask <= 0.5)
            if tp_mask.any() and fp_mask.any():
                tp_noise_mean = pll_probs[tp_mask, 2].mean()
                fp_noise_mean = pll_probs[fp_mask, 2].mean()
                margin_violation = self.cfg.noise_margin + tp_noise_mean - fp_noise_mean
                margin_term = torch.clamp(margin_violation, min=0.0)
                total_loss = total_loss + self.cfg.noise_margin_weight * margin_term
                margin_term_value = float(margin_term.item())

        # Momentum update for candidates
        momentum = self.cfg.candidate_momentum
        updated_candidates = (
            momentum * candidates + (1.0 - momentum) * pll_probs.detach()
        )
        updated_candidates = updated_candidates / (updated_candidates.sum(dim=1, keepdim=True) + eps)

        diagnostics = {
            'loss': total_loss,
            'alpha': alpha,
            'temperature': temperature,
            'bce_loss': float(bce_loss.item()),
            'pll_loss': float(pll_loss.item()),
            'pll_pos_loss': float(component_means[0].item()),
            'pll_neg_loss': float(component_means[1].item()),
            'pll_noise_loss': float(component_means[2].item()),
            'margin_loss': margin_term_value,
            'correction': correction,
            'weights': weights.detach(),
            'pll_probs': pll_probs.detach(),
            'updated_candidates': updated_candidates.detach(),
        }
        return diagnostics
