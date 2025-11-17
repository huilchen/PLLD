import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from copy import deepcopy
import random
import torch
import torch.utils.data as data


def load_all(dataset, data_path):

	base_path = os.path.abspath(data_path)
	dataset_name = os.path.basename(os.path.normpath(base_path))
	name_candidates = [dataset_name]
	if dataset_name != dataset and '/' not in dataset:
		name_candidates.append(dataset)

	file_variants = []
	for name in name_candidates:
		file_variants.extend([
			(
				f"{name}.train.rating",
				f"{name}.valid.rating",
				f"{name}.test.negative",
				f"{name}.test.rating",
			),
			(
				f"{name}_normal.train.rating",
				f"{name}_normal.valid.rating",
				f"{name}_normal.test.negative",
				f"{name}_normal.test.rating",
			),
		])

	train_rating = valid_rating = test_negative = test_rating = None
	for train_name, valid_name, test_neg_name, test_rating_name in file_variants:
		candidate = os.path.join(base_path, train_name)
		if os.path.exists(candidate):
			train_rating = candidate
			valid_rating = os.path.join(base_path, valid_name)
			test_negative = os.path.join(base_path, test_neg_name)
			test_rating = os.path.join(base_path, test_rating_name)
			break

	if train_rating is None:
		raise FileNotFoundError(f"Could not locate training file for dataset '{dataset}' under {base_path}")

	################# load training/validation/test metadata #################
	train_data = pd.read_csv(
		train_rating,
		sep='\t', header=None, names=['user', 'item', 'noisy'],
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

	valid_data = pd.read_csv(
		valid_rating,
		sep='\t', header=None, names=['user', 'item', 'noisy'],
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

	test_df = None
	test_positive_pairs = []
	test_fp_pairs = []
	if test_rating and os.path.exists(test_rating):
		test_df = pd.read_csv(
			test_rating,
			sep='\t', header=None, names=['user', 'item', 'label'],
			usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int8})
		for user, item, label in test_df.itertuples(index=False):
			if label == 1:
				test_positive_pairs.append((user, item))
			else:
				test_fp_pairs.append((user, item))
	elif os.path.exists(test_negative):
		with open(test_negative, 'r') as fd:
			line = fd.readline()
			while line is not None and line != '':
				if '\\t' in line:
					arr = line.strip().split('\\t')
				else:
					arr = line.strip().split('\t')

				if len(arr) < 2:
					line = fd.readline()
					continue

				if arr[0].startswith('('):
					u = int(arr[0][1:].split(',')[0])
				else:
					u = int(arr[0])

				item_token = arr[1]
				if item_token.endswith(')'):
					item_token = item_token[:-1]
				if ',' in item_token:
					item_token = item_token.split(',')[0]
				i = int(item_token)

				test_positive_pairs.append((u, i))
				line = fd.readline()
	else:
		raise FileNotFoundError(f"No test file found for dataset '{dataset}' under {base_path}")

	if dataset == "adressa":
		user_num = 212231
		item_num = 6596
	else:
		max_user = train_data['user'].max()
		max_item = train_data['item'].max()
		if not valid_data.empty:
			max_user = max(max_user, valid_data['user'].max())
			max_item = max(max_item, valid_data['item'].max())
		if test_positive_pairs:
			max_user = max(max_user, max(u for u, _ in test_positive_pairs))
			max_item = max(max_item, max(i for _, i in test_positive_pairs))
		if test_fp_pairs:
			max_user = max(max_user, max(u for u, _ in test_fp_pairs))
			max_item = max(max_item, max(i for _, i in test_fp_pairs))
		user_num = int(max_user) + 1
		item_num = int(max_item) + 1
	print("user, item num")
	print(user_num, item_num)
	train_data = train_data.values.tolist()
	valid_data = valid_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	train_data_list = []
	train_data_noisy = []
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0
		train_data_list.append([x[0], x[1]])
		train_data_noisy.append(x[2])

	################# load validation data #################
	valid_data_list = []
	for x in valid_data:
		valid_data_list.append([x[0], x[1]])

	user_pos = {}
	for x in train_data_list:
		if x[0] in user_pos:
			user_pos[x[0]].append(x[1])
		else:
			user_pos[x[0]] = [x[1]]
	for x in valid_data_list:
		if x[0] in user_pos:
			user_pos[x[0]].append(x[1])
		else:
			user_pos[x[0]] = [x[1]]


	################# load testing data #################
	test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

	test_data_pos = {}
	test_data_fp = {}

	if test_positive_pairs:
		for u, i in test_positive_pairs:
			test_data_pos.setdefault(u, []).append(i)
			test_mat[u, i] = 1.0
	if test_fp_pairs:
		for u, i in test_fp_pairs:
			test_data_fp.setdefault(u, []).append(i)
			test_mat[u, i] = 1.0

	# expose FP interactions for downstream usage
	load_all.last_test_fp = test_data_fp

	return train_data_list, valid_data_list, test_data_pos, user_pos, user_num, item_num, train_mat, train_data_noisy


load_all.last_test_fp = {}


class NCFDataPLL(data.Dataset):
	"""
	NCF Dataset with Partial Label Learning (PLL) support.

	Key differences from NCFData:
	1. Maintains candidate label sets for each sample (3-way: positive, negative, noise)
	2. Candidate sets are updated during training based on model predictions
	3. Returns candidate labels and sample indices for updating
	"""

	def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=0,
	             noisy_or_not=None, init_noise_prob=0.5):
		super(NCFDataPLL, self).__init__()
		"""
		Args:
			features: List of [user, item] pairs (positive samples from dataset)
			num_item: Total number of items
			train_mat: Sparse matrix of training interactions
			num_ng: Number of negative samples per positive sample
			is_training: 0=train, 1=valid, 2=test
			noisy_or_not: List indicating if each positive sample is clean (1) or noisy (0)
			init_noise_prob: Initial probability for noise candidate in FP samples
		"""
		self.features_ps = features
		if is_training == 0:
			self.noisy_or_not = noisy_or_not
		else:
			self.noisy_or_not = [0 for _ in range(len(features))]

		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]
		self.init_noise_prob = init_noise_prob

		# **NEW: Initialize candidate label sets for PLL**
		# Shape: [N, 3] where 3 = [P(positive), P(negative), P(noise)]
		self.candidate_labels = None
		if is_training == 0:  # Only for training set
			self.candidate_labels = self._init_candidate_sets()

	def _init_candidate_sets(self):
		"""
		Initialize candidate label sets based on noisy_or_not labels.

		Returns:
			torch.Tensor [N_positive, 3]: Candidate distributions for positive samples

		Candidate set design:
		- TP (noisy_or_not=1): [1.0, 0.0, 0.0] → Confirmed positive
		- FP (noisy_or_not=0): [init_noise_prob, 0.0, 1-init_noise_prob] → Controlled by init_noise_prob
		  - init_noise_prob=0.1: [0.1, 0, 0.9] → Strong noise bias
		  - init_noise_prob=0.5: [0.5, 0, 0.5] → Balanced (default)
		  - init_noise_prob=0.9: [0.9, 0, 0.1] → Strong positive bias
		"""
		num_pos = len(self.features_ps)
		candidates = torch.zeros(num_pos, 3, dtype=torch.float32)

		for idx, noisy_label in enumerate(self.noisy_or_not):
			if noisy_label == 1:  # TP: Clean positive
				candidates[idx] = torch.tensor([1.0, 0.0, 0.0])
			else:  # FP: Potentially noisy
				# Use init_noise_prob to control bias
				candidates[idx] = torch.tensor([self.init_noise_prob, 0.0, 1 - self.init_noise_prob])

		# Normalize (ensure sum to 1)
		candidates = candidates / candidates.sum(dim=1, keepdim=True)

		return candidates

	def update_candidate_labels(self, indices, new_candidates, momentum: float = 0.9):
		"""Update candidate label sets using momentum blending."""
		if self.candidate_labels is None:
			return

		if isinstance(indices, torch.Tensor):
			indices = indices.detach().cpu().tolist()
		if torch.is_tensor(new_candidates):
			new_candidates = new_candidates.detach().cpu()

		for i, idx in enumerate(indices):
			if idx < len(self.features_ps):
				current = self.candidate_labels[idx]
				candidate = new_candidates[i]
				updated = momentum * current + (1 - momentum) * candidate
				updated = updated / updated.sum()
				self.candidate_labels[idx] = updated
				self.candidate_labels_fill[idx] = updated

	def adaptive_init_candidates(self, bce_scores, thresholds=(0.7, 0.4), probs=(0.7, 0.5, 0.3)):
		"""
		Adaptively initialize candidate labels based on BCE scores.

		Scheme C: BCE Score-based Adaptive Initialization
		- High BCE score → More likely TP → Higher P(positive)
		- Low BCE score → More likely FP → Higher P(noise)

		Args:
			bce_scores: torch.Tensor [N_positive] of BCE scores in [0, 1]
			thresholds: tuple (high_thresh, low_thresh) for score categorization
			probs: tuple (high_p_pos, med_p_pos, low_p_pos) for P(positive) assignment

		Example with default thresholds=(0.7, 0.4) and probs=(0.7, 0.5, 0.3):
			- score > 0.7 → [0.7, 0.0, 0.3] (high confidence TP)
			- 0.4 < score <= 0.7 → [0.5, 0.0, 0.5] (uncertain)
			- score <= 0.4 → [0.3, 0.0, 0.7] (likely FP)
		"""
		if self.candidate_labels is None:
			return

		high_thresh, low_thresh = thresholds
		high_p_pos, med_p_pos, low_p_pos = probs

		num_pos = len(self.features_ps)
		assert len(bce_scores) == num_pos, f"BCE scores length {len(bce_scores)} != num_pos {num_pos}"

		for idx, score in enumerate(bce_scores):
			score_val = score.item() if torch.is_tensor(score) else score

			if score_val > high_thresh:
				# High BCE score → Likely TP
				self.candidate_labels[idx] = torch.tensor([high_p_pos, 0.0, 1.0 - high_p_pos])
			elif score_val > low_thresh:
				# Medium BCE score → Uncertain
				self.candidate_labels[idx] = torch.tensor([med_p_pos, 0.0, 1.0 - med_p_pos])
			else:
				# Low BCE score → Likely FP
				self.candidate_labels[idx] = torch.tensor([low_p_pos, 0.0, 1.0 - low_p_pos])

		# Ensure normalization
		self.candidate_labels = self.candidate_labels / self.candidate_labels.sum(dim=1, keepdim=True)

	def random_diff_init(self, noise_bias_ratio=0.56, noise_bias_probs=(0.3, 0.7), neutral_probs=(0.5, 0.5)):
		"""
		Random differentiated initialization for candidate labels.

		Scheme C-Prime: Learned from AC3 experiment
		Key insight: Breaking uniformity by randomly assigning different initial distributions
		is more effective than trying to use uninformative BCE scores.

		Args:
			noise_bias_ratio: Ratio of samples to initialize with noise bias (default: 0.56, from AC3)
			noise_bias_probs: (P(positive), P(noise)) for noise-biased samples (default: (0.3, 0.7))
			neutral_probs: (P(positive), P(noise)) for neutral samples (default: (0.5, 0.5))

		Mechanism:
			- 56% samples → [0.3, 0.0, 0.7] (noise bias)
			- 44% samples → [0.5, 0.0, 0.5] (neutral)

		Why it works (from AC3):
			1. Breaks uniformity: Not all samples start from [0.5, 0.5]
			2. Probabilistic advantage: 56% likely contains some FPs starting from favorable position
			3. Correctable: Even if TPs are misassigned, they can be corrected during training
			4. Reduces positive feedback loop: FPs don't all collapse to [1, 0, 0]
		"""
		if self.candidate_labels is None:
			return

		num_pos = len(self.features_ps)
		p_pos_noise, p_noise_noise = noise_bias_probs
		p_pos_neutral, p_noise_neutral = neutral_probs

		# Randomly select samples for noise bias
		import numpy as np
		np.random.seed()  # Ensure different randomness each run
		noise_bias_mask = np.random.rand(num_pos) < noise_bias_ratio

		for idx in range(num_pos):
			if noise_bias_mask[idx]:
				# Noise-biased initialization
				self.candidate_labels[idx] = torch.tensor([p_pos_noise, 0.0, p_noise_noise])
			else:
				# Neutral initialization
				self.candidate_labels[idx] = torch.tensor([p_pos_neutral, 0.0, p_noise_neutral])

		# Ensure normalization
		self.candidate_labels = self.candidate_labels / self.candidate_labels.sum(dim=1, keepdim=True)

	def ng_sample(self):
		"""
		Sample negative items for each positive sample.
		After sampling, creates candidate labels for negative samples.
		"""
		assert self.is_training != 2, 'no need to sampling when testing'

		self.features_ng = []
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]
		self.noisy_or_not_fill = self.noisy_or_not + [1 for _ in range(len(self.features_ng))]
		self.features_fill = self.features_ps + self.features_ng
		assert len(self.noisy_or_not_fill) == len(self.features_fill)
		self.labels_fill = labels_ps + labels_ng

		# **NEW: Create candidate labels for negative samples**
		# Negative samples: [0, 1, 0] → Confirmed negative
		if self.candidate_labels is not None:
			neg_candidates = torch.zeros(len(self.features_ng), 3, dtype=torch.float32)
			neg_candidates[:, 1] = 1.0  # All probability on "negative"

			# Concatenate positive and negative candidates
			self.candidate_labels_fill = torch.cat([self.candidate_labels, neg_candidates], dim=0)
		else:
			self.candidate_labels_fill = None

	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training != 2 \
					else self.features_ps
		labels = self.labels_fill if self.is_training != 2 \
					else self.labels
		noisy_or_not = self.noisy_or_not_fill if self.is_training != 2 \
					else self.noisy_or_not

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		noisy_label = noisy_or_not[idx]

		# **NEW: Return candidate labels and index**
		if self.candidate_labels_fill is not None:
			candidate = self.candidate_labels_fill[idx]
			return user, item, label, noisy_label, candidate, idx
		else:
			# For validation/test, return dummy candidate and index
			candidate = torch.tensor([0.0, 0.0, 0.0])
			return user, item, label, noisy_label, candidate, idx
