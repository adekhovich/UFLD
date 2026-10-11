import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerClassifierKVCached(nn.Module):
    """Causal Transformer that processes one patch at a time, with a key/value cache.

    It has the same submodules and parameter names as ``CausalTransformerSupCon`` (``proj``,
    ``time_pos_emb``, ``patch_id_emb``, ``transformer``, ``final_ln``, ``fc``, ``contrastive_embedding``),
    so a pretrained checkpoint loads directly. Instead of running the full ``nn.TransformerEncoder`` over
    the whole sequence, ``make_step`` encodes only the newly acquired patch and attends to the cached keys
    and values of the previous patches. The encoder layers are assumed to be pre-norm (``norm_first=True``).
    """

    def __init__(
        self,
        input_dim=16,
        hidden_dim=128,
        num_classes=8,
        seq_len=64,
        num_layers=4,
        num_heads=4,
        patch_size=4,
        dropout_prob=0.0,
        batch_first=True,
        layer_norm=True,
        use_cls_token=False,
        embedding_dim=None,
        use_patch_id_emb=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embedding_dim or hidden_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.batch_first = batch_first
        self.use_cls_token = use_cls_token
        self.use_patch_id_emb = use_patch_id_emb

        # Patch features -> model dimension
        self.proj = nn.Linear(input_dim, self.embed_dim)

        # Temporal embedding: position of the patch in the acquisition sequence
        self.time_pos_emb = nn.Embedding(seq_len + (1 if use_cls_token else 0), self.embed_dim)

        # Spatial embedding: location (patch id) of the patch on the wafer map
        if use_patch_id_emb:
            self.patch_id_emb = nn.Embedding(seq_len, self.embed_dim)

        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.cls, std=0.02)

        # Same layer configuration as CausalTransformerSupCon
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.embed_dim,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.final_ln = nn.LayerNorm(self.embed_dim) if layer_norm else nn.Identity()

        self.fc = nn.Linear(self.embed_dim, num_classes)
        self.contrastive_embedding = nn.Linear(self.embed_dim, self.embed_dim // 2)

    # ----------------------------
    # Weight loading
    # ----------------------------
    def load_weights(self, checkpoint_path, strict=True, verbose=True):
        """Load a checkpoint saved from ``CausalTransformerSupCon`` (same parameter names).

        ``strict=True`` (default) raises if any parameter is missing or unexpected, so a
        checkpoint from a different architecture or model size cannot be loaded silently.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError("Checkpoint format not understood (expected dict-like).")

        clean = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            clean[k] = v

        result = self.load_state_dict(clean, strict=strict)

        missing = list(getattr(result, "missing_keys", result[0]))
        unexpected = list(getattr(result, "unexpected_keys", result[1]))

        if verbose:
            if missing:
                print(f"[load_weights] Missing keys ({len(missing)}):")
                for k in missing:
                    print("  ", k)
            if unexpected:
                print(f"[load_weights] Unexpected keys ({len(unexpected)}):")
                for k in unexpected:
                    print(f"  {k}")

        return missing, unexpected

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    # ----------------------------
    # KV-cache utilities
    # ----------------------------
    @staticmethod
    def _split_qkv(in_proj_weight, in_proj_bias, x):
        """Input projection of MultiheadAttention: x [B, 1, E] -> q, k, v, each [B, 1, E]."""
        proj = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = proj.chunk(3, dim=-1)
        return q, k, v

    @staticmethod
    def _shape_heads(x, nhead):
        """[B, T, E] -> [B, heads, T, head_dim]."""
        B, T, E = x.shape
        head_dim = E // nhead
        return x.view(B, T, nhead, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x):
        """[B, heads, T, head_dim] -> [B, T, E]."""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def init_state(self, batch_size, device, dtype=None):
        """Empty key/value caches, one per layer, in head format [B, heads, t, head_dim]."""
        dtype = dtype or torch.float32

        layers = self.transformer.layers
        n_layers = len(layers)
        n_heads = layers[0].self_attn.num_heads
        head_dim = self.embed_dim // n_heads

        state = {
            "t": 0,  # number of observed patch tokens (CLS not counted)
            "k": [torch.zeros(batch_size, n_heads, 0, head_dim, device=device, dtype=dtype) for _ in range(n_layers)],
            "v": [torch.zeros(batch_size, n_heads, 0, head_dim, device=device, dtype=dtype) for _ in range(n_layers)],
        }

        # With a CLS token, seed the caches with it (time position 0, patch id 0)
        if self.use_cls_token:
            with torch.no_grad():
                cls_tok = self.cls.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
                cls_tok = cls_tok + self.time_pos_emb(torch.zeros(batch_size, 1, device=device, dtype=torch.long))
                if self.use_patch_id_emb:
                    cls_tok = cls_tok + self.patch_id_emb(torch.zeros(batch_size, 1, device=device, dtype=torch.long))
                _ = self._encode_one_token(cls_tok, state)
        return state

    def _encode_one_token(self, x_new, state):
        """Pass one token [B, 1, E] through all layers, appending its keys/values to the caches.

        Each pre-norm layer computes ``x = x + SA(norm1(x))`` followed by ``x = x + FF(norm2(x))``.
        The cache only contains earlier tokens, so attention is causal by construction.
        Returns the token representation after the final layer norm, [B, 1, E].
        """
        layers = self.transformer.layers

        for li, layer in enumerate(layers):
            x_norm = layer.norm1(x_new)

            # Self-attention of the new token over all cached tokens (including itself)
            attn = layer.self_attn
            q, k, v = self._split_qkv(attn.in_proj_weight, attn.in_proj_bias, x_norm)

            qh = self._shape_heads(q, attn.num_heads)   # [B, H, 1, D]
            kh = self._shape_heads(k, attn.num_heads)
            vh = self._shape_heads(v, attn.num_heads)

            k_all = torch.cat([state["k"][li], kh], dim=2)   # [B, H, t+1, D]
            v_all = torch.cat([state["v"][li], vh], dim=2)
            state["k"][li] = k_all
            state["v"][li] = v_all

            scale = qh.size(-1) ** -0.5
            scores = torch.matmul(qh, k_all.transpose(-2, -1)) * scale   # [B, H, 1, t+1]
            attn_weights = torch.softmax(scores, dim=-1)

            if attn.dropout and self.training:
                attn_weights = F.dropout(attn_weights, p=attn.dropout, training=True)

            ctx = self._merge_heads(torch.matmul(attn_weights, v_all))   # [B, 1, E]
            ctx = attn.out_proj(ctx)

            x_new = x_new + layer.dropout1(ctx)

            # Feed-forward block
            x_ff_in = layer.norm2(x_new)
            x_ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_ff_in))))
            x_new = x_new + layer.dropout2(x_ff)

        return self.final_ln(x_new)

    def make_step(self, input_patch, patch_id, state, detach_state=False):
        """Acquire one patch and return the prediction after it.

        Parameters
        ----------
        input_patch : torch.Tensor [B, input_dim]
        patch_id : torch.Tensor [B] or int
            Location of the patch on the wafer map.
        state : dict
            Caches from ``init_state`` or the previous step (updated in place).
        detach_state : bool
            Detach the cached tensors (only relevant if gradients flow through the classifier).

        Returns
        -------
        logits [B, C], z [B, E] (state vector used by the policy), proj (normalised contrastive embedding),
        state, and the input token embedding [B, E].
        """
        if input_patch.dim() != 2:
            raise ValueError("make_step expects input_patch as [B, input_dim].")

        B = input_patch.size(0)
        device = input_patch.device

        if not torch.is_tensor(patch_id):
            patch_id = torch.tensor(patch_id, device=device, dtype=torch.long).view(1).expand(B)
        else:
            patch_id = patch_id.to(device).view(-1)
            if patch_id.numel() == 1:
                patch_id = patch_id.expand(B)

        # Token embedding: projection + temporal embedding + spatial embedding
        x_new = self.proj(input_patch).unsqueeze(1)   # [B, 1, E]

        time_idx = state["t"] + 1 if self.use_cls_token else state["t"]
        time_idx = torch.full((B, 1), int(time_idx), device=device, dtype=torch.long)
        time_idx = time_idx.clamp_max(self.time_pos_emb.num_embeddings - 1)
        x_new = x_new + self.time_pos_emb(time_idx)

        if self.use_patch_id_emb:
            x_new = x_new + self.patch_id_emb(patch_id.view(B, 1).clamp(0, self.seq_len - 1))

        h_new = self._encode_one_token(x_new, state)   # [B, 1, E]

        z = h_new.squeeze(1)
        logits = self.fc(z)
        proj = F.normalize(self.contrastive_embedding(z), dim=1)

        state["t"] += 1

        if detach_state:
            for li in range(len(state["k"])):
                state["k"][li] = state["k"][li].detach()
                state["v"][li] = state["v"][li].detach()

        return logits, z, proj, state, x_new.squeeze(1)
