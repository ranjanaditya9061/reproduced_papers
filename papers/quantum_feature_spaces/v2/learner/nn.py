"""A torch MLP student: one training loop, parameterised by width/depth/optimiser.

De-duplicates the two legacy loops (``learner/photonic_quantum.py::train_and_eval_quantum`` and
``learner/mlp.py::train_and_eval``), which differed only in the model they wrapped.

**Scope.** This is the *classical* nn student, which is all the paired comparison in
:mod:`v2.learner.compare` needs -- it trains on ``(x, soft)`` and never touches a circuit.  A
trainable **quantum** student (a differentiable MZI sandwich or IQP circuit, i.e. ``v2/parametric/``)
is deliberately not implemented: it is a fourth model layer whose only consumer is this file, and
the comparison's decision logic is about whether *some* adequate learner separates the two label
sets, not about which family it comes from.  Add it if the question becomes "can a quantum student
do better", which is a different question from the one §7 poses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Learner


class MlpLearner(Learner):
    """MLP regressor with early stopping on a held-out slice of TRAIN.

    Early stopping never sees the test split -- otherwise the reported held-out log-likelihood is
    optimistic and the paired difference is contaminated.
    """

    name = "mlp"

    def __init__(self, *, hidden: int = 128, depth: int = 2, epochs: int = 400, lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 256, patience: int = 40,
                 val_fraction: float = 0.15, seed: int = 0):
        super().__init__(hidden=hidden, depth=depth, epochs=epochs, lr=lr,
                         weight_decay=weight_decay, batch_size=batch_size, patience=patience,
                         val_fraction=val_fraction, seed=seed)
        self.net = None

    def _build(self, d_in: int) -> nn.Module:
        h, depth = self.hparams["hidden"], self.hparams["depth"]
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "MlpLearner":
        h = self.hparams
        torch.manual_seed(int(h["seed"]))
        n = X.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(h["seed"])))
        n_val = max(1, int(n * float(h["val_fraction"])))
        vi, ti = perm[:n_val], perm[n_val:]
        Xt, yt, Xv, yv = X[ti], y[ti], X[vi], y[vi]

        # Standardise inputs and targets: the observable scales range over four orders of
        # magnitude across the registry, so a shared learning rate is otherwise meaningless.
        self.x_mu, self.x_sd = Xt.mean(0), Xt.std(0).clamp(min=1e-8)
        self.y_mu, self.y_sd = yt.mean(), yt.std().clamp(min=1e-8)

        self.net = self._build(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
        loss_fn = nn.MSELoss()

        best, best_state, since = float("inf"), None, 0
        for _ in range(int(h["epochs"])):
            self.net.train()
            order = torch.randperm(Xt.shape[0])
            for i in range(0, Xt.shape[0], int(h["batch_size"])):
                b = order[i:i + int(h["batch_size"])]
                opt.zero_grad()
                out = self.net((Xt[b] - self.x_mu) / self.x_sd).squeeze(-1)
                loss_fn(out, (yt[b] - self.y_mu) / self.y_sd).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(loss_fn(self.net((Xv - self.x_mu) / self.x_sd).squeeze(-1),
                                   (yv - self.y_mu) / self.y_sd))
            if vl < best - 1e-7:
                best, best_state, since = vl, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                since += 1
                if since >= int(h["patience"]):
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self._record_residual(X, y)
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        z = self.net((X - self.x_mu) / self.x_sd).squeeze(-1)
        return z * self.y_sd + self.y_mu
