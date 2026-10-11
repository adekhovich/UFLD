import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import img_to_patch


class CausalTransformerSupCon(nn.Module):
    """Causal Transformer classifier trained on (randomly ordered) patch sequences.

    Each token is a projected patch plus a temporal embedding (acquisition step) and a spatial embedding
    (patch location). A causal mask lets the output at position t depend only on the first t + 1 patches,
    so one forward pass gives the prediction after every number of acquired patches. ``forward`` returns
    the logits, the features and an L2-normalised projection used by the supervised contrastive loss.
    For step-by-step acquisition the same weights are loaded into ``CausalTransformerClassifierKVCached``.
    """

    def __init__(
        self,
        input_dim=16,
        hidden_dim=128,
        num_classes=8,
        seq_len=64,              # number of patches (169 for 52x52 maps and 4x4 patches)
        num_layers=4,
        num_heads=4,
        patch_size=4,
        dropout_prob=0.0,
        dataset_name="mixedwm38",
        batch_first=True,
        layer_norm=True,
        use_cls_token=False,
        embedding_dim=None,
        use_patch_id_emb=True,   # spatial (patch location) embedding
        permute_patches=False,   # feed the patches in a random order (used for pretraining)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embedding_dim or hidden_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.dataset_name = dataset_name
        self.batch_first = batch_first
        self.use_cls_token = use_cls_token

        self.use_patch_id_emb = use_patch_id_emb
        self.permute_patches = permute_patches
        self.patch_order = None

        # Project patch vectors to model dimension
        self.proj = nn.Linear(input_dim, self.embed_dim)

        # Temporal embedding: position 0..T-1 of a patch in the acquisition sequence
        self.time_pos_emb = nn.Embedding(seq_len + (1 if use_cls_token else 0), self.embed_dim)

        # Spatial embedding: location (patch id 0..seq_len-1) of a patch on the wafer map
        if use_patch_id_emb:
            self.patch_id_emb = nn.Embedding(seq_len, self.embed_dim)

        # Optional CLS token (not used in the paper)
        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.cls, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.embed_dim,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.final_ln = nn.LayerNorm(self.embed_dim) if layer_norm else nn.Identity()

        self.fc = nn.Linear(self.embed_dim, num_classes)
        self.contrastive_embedding = nn.Linear(self.embed_dim, self.embed_dim // 2)

    def _build_causal_mask(self, T, device):
        # Float mask with -inf above the diagonal
        m = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def _build_key_padding_mask(self, token_lengths, T, device):
        B = token_lengths.size(0)
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        return idx >= token_lengths.unsqueeze(1)

    def forward(
        self,
        inputs,
        hidden=None,
        multioutput=False,
        patch_order=None,         # patch id of each token position, [B, S] or [S]
        prediction_index=-1,
        lengths=None
    ):
        # Patchify
        if inputs.dim() == 4:
            x = img_to_patch(inputs, patch_size=self.patch_size, flatten_channels=True)  # [B,S,input_dim]
        else:
            x = inputs  # [B,S,input_dim]

        B, S, _ = x.shape
        device = x.device

        # At most seq_len patches
        if S > self.seq_len:
            x = x[:, :self.seq_len, :]
            S = self.seq_len

        # patch_order[b, t] is the patch id acquired at step t. If not given: a random permutation per wafer
        # (permute_patches=True) or the identity order 0..S-1.
        if patch_order is None:
            if self.permute_patches:
                patch_order = torch.stack([torch.randperm(S, device=device) for _ in range(B)], dim=0)
            else:
                patch_order = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        else:
            if patch_order.dim() == 1:
                patch_order = patch_order.unsqueeze(0).expand(B, -1).to(device)
            else:
                patch_order = patch_order.to(device)

        # Reorder the patches into acquisition order
        if self.permute_patches:
            # After the gather, token position t holds patch patch_order[:, t]
            x = x.gather(dim=1, index=patch_order.unsqueeze(-1).expand_as(x))

        self.patch_order = patch_order

        # Project to the model dimension
        x = self.proj(x)  # [B,S,E]

        # Optional CLS token
        if self.use_cls_token:
            cls_tok = self.cls.expand(B, -1, -1)  # [B,1,E]
            x = torch.cat([cls_tok, x], dim=1)    # [B,1+S,E]
            T = 1 + S
        else:
            T = S

        # Temporal embedding (acquisition step)
        pos_t = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B,T]
        x = x + self.time_pos_emb(pos_t)

        # Spatial embedding (patch location)
        if self.use_patch_id_emb:
            if self.use_cls_token:
                # Patch id 0 for the CLS token
                patch_ids = torch.cat(
                    [torch.zeros(B, 1, dtype=torch.long, device=device), patch_order], dim=1
                )  # [B,T]
            else:
                patch_ids = patch_order  # [B,S]

            x = x + self.patch_id_emb(patch_ids.clamp(0, self.seq_len - 1))

        # Masks
        causal_mask = self._build_causal_mask(T, device=device)
        key_padding_mask = None
        if lengths is not None:
            token_lengths = lengths.clamp(min=0, max=S)
            if self.use_cls_token:
                token_lengths = token_lengths + 1
            key_padding_mask = self._build_key_padding_mask(token_lengths, T, device)

        # Causal Transformer
        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)  # [B,T,E]
        h = self.final_ln(h)

        # Logits, features and normalised contrastive embedding at a position
        def logits_and_proj_at(pos_idx):
            z = h[:, pos_idx, :]
            logits = self.fc(z)
            proj = F.normalize(self.contrastive_embedding(z), dim=1)
            return logits, z, proj

        # Outputs at all positions (multioutput), a list of positions, or one position
        if multioutput:
            fc_output, contrastive_output, output_hidden = [], [], []
            for t in range(T):
                lo, zz, pr = logits_and_proj_at(t)
                fc_output.append(lo); output_hidden.append(zz); contrastive_output.append(pr)

        elif isinstance(prediction_index, list):
            fc_output, contrastive_output, output_hidden = [], [], []
            for idx in prediction_index:
                # Negative indices count from the end
                if idx < 0:
                    idx = T + idx
                lo, zz, pr = logits_and_proj_at(idx)
                fc_output.append(lo); output_hidden.append(zz); contrastive_output.append(pr)

        else:
            idx = prediction_index
            if idx < 0:
                idx = T + idx
            lo, zz, pr = logits_and_proj_at(idx)
            fc_output = lo
            output_hidden = zz
            contrastive_output = pr

        return fc_output, output_hidden, contrastive_output
