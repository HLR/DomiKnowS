"""Amortized dual-multiplier network for R5 Phase B.

In the constraint-granularity dual (R5 Phase A) a single scalar multiplier
``lambda_c`` scales a whole constraint's summed violation, so it cannot
re-attribute pressure among individual groundings (a barely-violated pair and a
badly-violated pair get the same weight). The :class:`DualCritic` predicts a
*per-grounding* multiplier ``lambda_g`` from cheap, detached features, giving
finer credit assignment while staying a drop-in for the existing primal-dual
loop:

* it is a submodule of the constraint model, so its parameters are exactly the
  ``cmodel.parameters()`` the program's constraint optimizer already ascends —
  ``reverse_sign_grad`` + ``copt.step`` maximises ``sum_g lambda_g * v_g`` with
  no change to ``train_epoch``;
* the multiplier is bounded to ``[0, lmbd_p_c]`` by a scaled sigmoid, so no
  projection step is needed;
* features are **detached**, so there is no ``theta -> lambda`` gradient path;
  the primal gradient stays ``sum_g lambda_g * d v_g / d theta``.

Feature vector per grounding (fixed width, so one MLP serves every constraint):

    [ violation_g,  lit_mean, lit_min, lit_max,  <constraint embedding> ]

The three literal-summary slots come from the participating classifier
probabilities exported by the compiled LC path (``groundingFeatures``); when
those are unavailable (interpreter path) they are zero-filled, which reduces the
critic to violation + constraint identity.
"""

import torch


#: Number of fixed literal-summary features (mean/min/max of the participating
#: classifier probabilities per grounding).
N_LITERAL_FEATURES = 3


class DualCritic(torch.nn.Module):
    def __init__(self, nconstr: int, embed_dim: int = 8, hidden: int = 32):
        """
        :param nconstr: number of logical constraints (embedding table size).
        :param embed_dim: per-constraint learned embedding width.
        :param hidden: hidden width of the multiplier MLP.
        """
        super().__init__()
        self.nconstr = max(int(nconstr), 1)
        self.embed_dim = int(embed_dim)
        self.n_literal_features = N_LITERAL_FEATURES

        self.constraint_embedding = torch.nn.Embedding(self.nconstr, self.embed_dim)
        in_dim = 1 + self.n_literal_features + self.embed_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def literal_summary(self, features, device, dtype):
        """Fixed-width [G, 3] summary of per-grounding literal probabilities.

        ``features`` is a detached ``[G, L]`` tensor (participating classifier
        probabilities) or None → zero summary.
        """
        if features is None or not torch.is_tensor(features) or features.numel() == 0:
            return None
        f = features.detach().to(device=device, dtype=dtype)
        if f.dim() == 1:
            f = f.unsqueeze(-1)
        return torch.stack([f.mean(dim=-1), f.amin(dim=-1), f.amax(dim=-1)], dim=-1)

    def forward(self, constraint_index: int, violation: torch.Tensor,
                literal_features=None):
        """Per-grounding multipliers in ``(0, 1)`` (scale by ``lmbd_p`` outside).

        :param constraint_index: index of this constraint (for the embedding).
        :param violation: ``[G]`` detached per-grounding violation values.
        :param literal_features: optional ``[G, L]`` detached literal probs.
        :returns: ``[G]`` tensor in ``(0, 1)``.
        """
        device = self.constraint_embedding.weight.device
        dtype = self.constraint_embedding.weight.dtype
        v = violation.detach().to(device=device, dtype=dtype).reshape(-1)
        v = torch.nan_to_num(v, nan=0.0)
        G = v.shape[0]

        lit = self.literal_summary(literal_features, device, dtype)
        if lit is None or lit.shape[0] != G:
            lit = torch.zeros(G, self.n_literal_features, device=device, dtype=dtype)
        lit = torch.nan_to_num(lit, nan=0.0)

        idx = torch.full((G,), int(constraint_index) % self.nconstr,
                         dtype=torch.long, device=device)
        emb = self.constraint_embedding(idx)  # [G, embed_dim]

        x = torch.cat([v.unsqueeze(-1), lit, emb], dim=-1)  # [G, in_dim]
        raw = self.mlp(x).squeeze(-1)  # [G]
        return torch.sigmoid(raw)
