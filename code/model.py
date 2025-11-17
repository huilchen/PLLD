import torch
import torch.nn as nn
import torch.nn.functional as F


class NCF_PLL(nn.Module):
	"""
	Neural Collaborative Filtering with Partial Label Learning (PLL).

	Architecture:
		User/Item Embeddings → GMF + MLP → Shared Representation
		                                         ↓
		                           ┌─────────────┴──────────────┐
		                           ↓                            ↓
		                    BCE Head                       PLL Head
		                   (1 output)                    (3 outputs)
		                       ↓                              ↓
		                  sigmoid score          [P(pos), P(neg), P(noise)]

	Dual-head design allows:
	1. BCE head: Standard recommendation objective (for inference)
	2. PLL head: Denoising objective with 3-way classification
	"""

	def __init__(self, user_num, item_num, factor_num, num_layers,
	             dropout, model, GMF_model=None, MLP_model=None,
	             pll_hidden_dim=128, separate_pll_embeddings=False):
		super(NCF_PLL, self).__init__()
		"""
		Args:
			user_num: Number of users
			item_num: Number of items
			factor_num: Number of predictive factors
			num_layers: Number of layers in MLP model
			dropout: Dropout rate between fully connected layers
			model: 'MLP', 'GMF', 'NeuMF-end', or 'NeuMF-pre'
			GMF_model: Pre-trained GMF weights (optional)
			MLP_model: Pre-trained MLP weights (optional)
			pll_hidden_dim: Hidden dimension for PLL head
			separate_pll_embeddings: If True, use separate embeddings for PLL (prevents BCE pollution)
		"""
		self.dropout = dropout
		self.model = model
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model
		self.separate_pll_embeddings = separate_pll_embeddings

		# ============ BCE Embedding Layers ============
		self.embed_user_GMF = nn.Embedding(user_num, factor_num)
		self.embed_item_GMF = nn.Embedding(item_num, factor_num)
		self.embed_user_MLP = nn.Embedding(
			user_num, factor_num * (2 ** (num_layers - 1)))
		self.embed_item_MLP = nn.Embedding(
			item_num, factor_num * (2 ** (num_layers - 1)))

		# ============ PLL Separate Embeddings (Optional) ============
		if separate_pll_embeddings:
			self.pll_embed_user_GMF = nn.Embedding(user_num, factor_num)
			self.pll_embed_item_GMF = nn.Embedding(item_num, factor_num)
			self.pll_embed_user_MLP = nn.Embedding(
				user_num, factor_num * (2 ** (num_layers - 1)))
			self.pll_embed_item_MLP = nn.Embedding(
				item_num, factor_num * (2 ** (num_layers - 1)))

		# ============ BCE MLP Layers ============
		MLP_modules = []
		for i in range(num_layers):
			input_size = factor_num * (2 ** (num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size // 2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		# ============ PLL Separate MLP Layers (Optional) ============
		if separate_pll_embeddings:
			pll_MLP_modules = []
			for i in range(num_layers):
				input_size = factor_num * (2 ** (num_layers - i))
				pll_MLP_modules.append(nn.Dropout(p=self.dropout))
				pll_MLP_modules.append(nn.Linear(input_size, input_size // 2))
				pll_MLP_modules.append(nn.ReLU())
			self.pll_MLP_layers = nn.Sequential(*pll_MLP_modules)

		# ============ Shared Representation Size ============
		if self.model in ['MLP', 'GMF']:
			concat_size = factor_num
		else:
			concat_size = factor_num * 2

		# ============ HEAD A: BCE Prediction Head ============
		# For standard recommendation (used during inference)
		self.predict_layer = nn.Linear(concat_size, 1)

		# ============ HEAD B: PLL Classification Head ============
		# For 3-way classification: [positive, negative, noise]
		self.pll_head = nn.Sequential(
			nn.Linear(concat_size, pll_hidden_dim),
			nn.ReLU(),
			nn.Dropout(p=self.dropout),
			nn.Linear(pll_hidden_dim, 3)  # Output: [P(pos), P(neg), P(noise)]
		)

		self._init_weight_()

	def _init_weight_(self):
		"""Weight initialization"""
		if not self.model == 'NeuMF-pre':
			# Initialize BCE embeddings
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			# Initialize PLL separate embeddings (if enabled)
			if self.separate_pll_embeddings:
				nn.init.normal_(self.pll_embed_user_GMF.weight, std=0.01)
				nn.init.normal_(self.pll_embed_user_MLP.weight, std=0.01)
				nn.init.normal_(self.pll_embed_item_GMF.weight, std=0.01)
				nn.init.normal_(self.pll_embed_item_MLP.weight, std=0.01)

			# Initialize BCE MLP layers
			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)

			# Initialize PLL separate MLP layers (if enabled)
			if self.separate_pll_embeddings:
				for m in self.pll_MLP_layers:
					if isinstance(m, nn.Linear):
						nn.init.xavier_uniform_(m.weight)

			# Initialize BCE head
			nn.init.kaiming_uniform_(self.predict_layer.weight,
			                         a=1, nonlinearity='sigmoid')

			# Initialize PLL head
			for m in self.pll_head:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)

			# Zero-initialize biases
			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()

		else:
			# Pre-training mode (load from GMF and MLP models)
			self.embed_user_GMF.weight.data.copy_(
				self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(
				self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(
				self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(
				self.MLP_model.embed_item_MLP.weight)

			# Copy MLP layers
			for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# Copy predict layers
			predict_weight = torch.cat([
				self.GMF_model.predict_layer.weight,
				self.MLP_model.predict_layer.weight], dim=1)
			precit_bias = self.GMF_model.predict_layer.bias + \
			              self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)

			# PLL head is randomly initialized even in pre-training mode

	def _get_representation(self, user, item, for_pll=False):
		"""
		Get representation from GMF and MLP.

		Args:
			user: User indices
			item: Item indices
			for_pll: If True and separate_pll_embeddings=True, use PLL embeddings

		Returns:
			concat: torch.Tensor [batch, concat_size]
		"""
		# Choose which embeddings and layers to use
		if for_pll and self.separate_pll_embeddings:
			# Use PLL-specific embeddings and layers
			embed_user_GMF = self.pll_embed_user_GMF
			embed_item_GMF = self.pll_embed_item_GMF
			embed_user_MLP = self.pll_embed_user_MLP
			embed_item_MLP = self.pll_embed_item_MLP
			MLP_layers = self.pll_MLP_layers
		else:
			# Use standard BCE embeddings and layers
			embed_user_GMF = self.embed_user_GMF
			embed_item_GMF = self.embed_item_GMF
			embed_user_MLP = self.embed_user_MLP
			embed_item_MLP = self.embed_item_MLP
			MLP_layers = self.MLP_layers

		if not self.model == 'MLP':
			user_gmf = embed_user_GMF(user)
			item_gmf = embed_item_GMF(item)
			output_GMF = user_gmf * item_gmf

		if not self.model == 'GMF':
			user_mlp = embed_user_MLP(user)
			item_mlp = embed_item_MLP(item)
			interaction = torch.cat((user_mlp, item_mlp), -1)
			output_MLP = MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		return concat

	def set_pll_trainable(self, trainable: bool):
		for param in self.pll_head.parameters():
			param.requires_grad = trainable
		if self.separate_pll_embeddings:
			for emb in [self.pll_embed_user_GMF, self.pll_embed_item_GMF,
			            self.pll_embed_user_MLP, self.pll_embed_item_MLP]:
				emb.weight.requires_grad = trainable
			for param in self.pll_MLP_layers.parameters():
				param.requires_grad = trainable

	def forward(self, user, item, return_pll=False, return_both=False, stop_grad_for_pll=False):
		"""
		Forward pass with dual heads.

		Args:
			user: torch.Tensor [batch] - User indices
			item: torch.Tensor [batch] - Item indices
			return_pll: bool - If True, return PLL logits instead of BCE prediction
			return_both: bool - If True, return both BCE and PLL outputs
			stop_grad_for_pll: bool - If True, detach features for PLL head to prevent BCE pollution

		Returns:
			If return_pll=True: pll_logits [batch, 3]
			If return_both=True: (bce_prediction [batch], pll_logits [batch, 3])
			Otherwise: bce_prediction [batch]
		"""
		if return_both:
			# Get BCE representation
			bce_concat = self._get_representation(user, item, for_pll=False)
			bce_pred = self.predict_layer(bce_concat).view(-1)

			# Get PLL representation (separate if enabled, otherwise shared)
			pll_concat = self._get_representation(user, item, for_pll=True)

			# Apply stop gradient for PLL if requested (and not using separate embeddings)
			if stop_grad_for_pll and not self.separate_pll_embeddings:
				pll_concat = pll_concat.detach()

			pll_logits = self.pll_head(pll_concat)
			return bce_pred, pll_logits

		elif return_pll:
			# Return PLL head only (for training)
			pll_concat = self._get_representation(user, item, for_pll=True)

			# Apply stop gradient if requested (and not using separate embeddings)
			if stop_grad_for_pll and not self.separate_pll_embeddings:
				pll_concat = pll_concat.detach()

			pll_logits = self.pll_head(pll_concat)
			return pll_logits

		else:
			# Return BCE head only (for inference)
			concat = self._get_representation(user, item, for_pll=False)
			prediction = self.predict_layer(concat)
			return prediction.view(-1)

	def get_pll_probs(self, user, item):
		"""
		Get PLL probabilities (after softmax).

		Returns:
			probs: torch.Tensor [batch, 3] - [P(pos), P(neg), P(noise)]
		"""
		pll_logits = self.forward(user, item, return_pll=True)
		return F.softmax(pll_logits, dim=1)


if __name__ == '__main__':
	"""
	Unit tests for NCF_PLL model.
	"""
	print("Testing NCF_PLL model...")

	# Test 1: Model initialization
	print("\n=== Test 1: Model Initialization ===")
	model = NCF_PLL(user_num=100, item_num=50, factor_num=8, num_layers=3,
	                dropout=0.0, model='NeuMF-end', pll_hidden_dim=32)
	print(f"Model created successfully")
	print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

	# Test 2: Forward pass (BCE head)
	print("\n=== Test 2: BCE Head Forward Pass ===")
	user = torch.tensor([0, 1, 2])
	item = torch.tensor([5, 10, 15])
	bce_pred = model(user, item, return_pll=False)
	print(f"Input shape: user {user.shape}, item {item.shape}")
	print(f"BCE output shape: {bce_pred.shape}")
	print(f"BCE predictions: {bce_pred}")

	# Test 3: Forward pass (PLL head)
	print("\n=== Test 3: PLL Head Forward Pass ===")
	pll_logits = model(user, item, return_pll=True)
	print(f"PLL logits shape: {pll_logits.shape}")
	print(f"PLL logits:\n{pll_logits}")

	pll_probs = F.softmax(pll_logits, dim=1)
	print(f"PLL probabilities:\n{pll_probs}")
	print(f"Probability sums: {pll_probs.sum(dim=1)}")

	# Test 4: Dual head output
	print("\n=== Test 4: Both Heads ===")
	bce_pred, pll_logits = model(user, item, return_both=True)
	print(f"BCE predictions: {bce_pred.shape}")
	print(f"PLL logits: {pll_logits.shape}")

	# Test 5: Batch processing
	print("\n=== Test 5: Batch Processing ===")
	batch_user = torch.randint(0, 100, (32,))
	batch_item = torch.randint(0, 50, (32,))
	batch_bce = model(batch_user, batch_item)
	batch_pll = model(batch_user, batch_item, return_pll=True)
	print(f"Batch size: 32")
	print(f"BCE output: {batch_bce.shape}")
	print(f"PLL output: {batch_pll.shape}")

	print("\n✅ All tests passed!")
