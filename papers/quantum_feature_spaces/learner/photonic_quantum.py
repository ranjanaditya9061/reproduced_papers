"""Variational quantum circuit (sandwich student) for classification experiments.

The student mirrors the teacher sandwich structure from data.photonic_quantum:

    Teacher:  W1 (fixed Haar) → phase encode(x) → W2 (fixed Haar)
    Student:  T1 (trainable)  → phase encode(x) → T2 (trainable)

T1, T2 are full MZI interferometers; the data encoding is identical to the
teacher (one phase shifter per mode 0..m-2, matching input x ∈ [0,2π]^{m-1}).

For depth > 1, data re-uploading is used:
    T1 → encode(x) → T2 → encode(x) → T3 → … → T_{depth+1}

A small linear head (n_fock_states → 2 logits) completes the classifier so
the same model can be evaluated on any of the three generators.
"""

from __future__ import annotations

import math

import torch.nn.functional as F
import perceval as pcvl
from perceval.components.generic_interferometer import InterferometerShape
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from perceval.utils.algorithms.circuit_optimizer import CircuitOptimizer

import merlin as ML
import numpy
from data import GENERATORS
from data.photonic_quantum import parity_of_key, majority_score_of_key


# ---------------------------------------------------------------------------
# Teacher validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_teacher_params(
    metadata: dict,
    m_circuits: list[int],
) -> None:
    """For each student circuit size, check if the teacher W1/W2 are reachable at depth=1.

    Uses ``CircuitOptimizer.optimize_rectangle()`` to numerically optimize an
    m_c-mode RECTANGLE interferometer onto W1/W2 (embedded in m_c×m_c when
    m_c > m).  Prints a fidelity table: fidelity ~1.0 means the student at
    that size can represent the teacher exactly at depth=1.
    """
    import numpy as np

    m  = metadata["m"]
    W1 = metadata["teacher_W1"]
    W2 = metadata["teacher_W2"]

    opt = CircuitOptimizer(threshold=1e-4, ntrials=4)

    print(f"  Teacher reachability at depth=1 (fidelity of RECTANGLE(m_c) → W1/W2):")
    print(f"  {'m_c':>5}  {'fid(W1)':>10}  {'fid(W2)':>10}  {'reachable':>10}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}")
    for m_c in sorted(set(m_circuits)):
        if m_c >= m:
            # Embed W1/W2 in an m_c×m_c unitary (identity on extra modes)
            W1_p = np.eye(m_c, dtype=complex)
            W1_p[:m, :m] = W1
            W2_p = np.eye(m_c, dtype=complex)
            W2_p[:m, :m] = W2
        else:
            # m_c < m: use the top-left m_c×m_c submatrix (informative but approximate)
            W1_p = W1[:m_c, :m_c]
            W2_p = W2[:m_c, :m_c]

        try:
            C1 = opt.optimize_rectangle(pcvl.Matrix(W1_p), allow_error=True)
            C2 = opt.optimize_rectangle(pcvl.Matrix(W2_p), allow_error=True)
            U1 = np.array(C1.compute_unitary())
            U2 = np.array(C2.compute_unitary())
            fid1 = (abs(np.trace(U1.conj().T @ W1_p)) / m_c) ** 2
            fid2 = (abs(np.trace(U2.conj().T @ W2_p)) / m_c) ** 2
            ok = "✓" if min(fid1, fid2) >= 0.999 else "✗ low"
            print(f"  {m_c:>5}  {fid1:>10.6f}  {fid2:>10.6f}  {ok:>10}")
        except Exception as e:
            print(f"  {m_c:>5}  {'n/a':>10}  {'n/a':>10}  {'⚠ err':>10}  ({e})")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _evenly_spaced_input_state(m_circuit: int, k: int) -> list[int]:
    """Spread k photons evenly across m_circuit modes.

    Examples
    --------
    k=3, m=6  → [1,0,1,0,1,0]  (positions 0, 2, 4)
    k=3, m=9  → [1,0,0,1,0,0,1,0,0]  (positions 0, 3, 6)
    k=2, m=6  → [1,0,0,1,0,0]  (positions 0, 3)
    """
    state = [0] * m_circuit
    for i in range(k):
        pos = (i * m_circuit) // k
        state[pos] = 1
    return state


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QuantumClassifier(nn.Module):
    """Variational quantum circuit with parity readout for binary classification.

    The forward pass mirrors the teacher measurement exactly:
        logit = probs @ parity_vec
    where parity_vec[i] = +1 if the i-th Fock state has even photon count on
    parity_modes, -1 otherwise.  parity_vec is a fixed buffer (not trained).

    Returning 2-class logits [-logit, +logit] keeps the CrossEntropyLoss
    interface identical to the MLP student.

    Parameters
    ----------
    m : int
        Number of optical modes.
    k : int
        Number of injected photons.
    depth : int
        Number of (trainable unitary → data encoding) repetitions.
        depth=1 reproduces the single sandwich structure of the teacher.
        depth>1 enables data re-uploading.
    parity_modes : sequence of int, optional
        Modes used for the parity measurement.  Defaults to range((m+1)//2),
        matching the teacher's default in generate_data.py.
    init_scale : float, optional
        Scale of the initial parameter distribution, as a fraction of π.
        1.0 (default of QuantumLayer) = wide random init, N(0, π).
        0.05 (recommended) = near-zero init that maximises initial gradient
        signal (empirically optimal across m=4..6) while avoiding the
        barren plateau of the full-random regime (≥0.2).
    """

    def __init__(
        self,
        m: int,
        k: int,
        depth: int = 1,
        m_circuit: int | None = None,
        n_features: int | None = None,
        learner_encoding="parity",
        parity_modes=None,
        init_scale: float = 0.01,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.k = k
        self.depth = depth
        self.m_circuit = m_circuit if m_circuit is not None else m

        if parity_modes is None:
            parity_modes = tuple(range((m + 1) // 2))
        self.parity_modes = tuple(parity_modes)

        n_features = n_features if n_features is not None else m - 1
        self.n_features = n_features
        m_c = self.m_circuit
        self.learner_encoding = learner_encoding

        # Build the circuit directly with pcvl.GenericInterferometer.
        #
        # RECTANGLE (Clements) topology: m*(m-1)/2 MZIs × 2 phase params each
        # = m*(m-1) MZI params, plus m diagonal output phases = m² total.
        # For SU(m) only m²-1 independent params are needed (1 global-phase
        # redundancy), so this gives exactly 1 redundant direction per block —
        # acceptable and standard.
        #
        # NOTE: the final block's diagonal phases phi{depth} fall immediately
        # before photon-number measurement, which is phase-invariant.  Those
        # m_c phases are physically undetectable and always carry zero gradient;
        # they are therefore omitted from the circuit (see loop below).
        #
        # Each block Wd: GenericInterferometer RECTANGLE with Clements MZIs
        #   (BS // PS(li) // BS // PS(lo)) gives m*(m-1) params.
        # Intermediate blocks also get m diagonal phases phi (= m² total).
        # Final block has no phi (invisible to measurement).
        circuit = pcvl.Circuit(m_c, name="student_sandwich")
        trainable_prefixes: list[str] = []

        for d in range(depth + 1):
            def _mzi_factory(d: int = d) -> object:
                def factory(i: int) -> pcvl.Circuit:
                    return (
                        pcvl.BS()
                        // pcvl.PS(pcvl.P(f"W{d}_li{i}"))
                        // pcvl.BS()
                        // pcvl.PS(pcvl.P(f"W{d}_lo{i}"))
                    )
                return factory

            gi = pcvl.GenericInterferometer(
                m_c,
                _mzi_factory(),
                shape=InterferometerShape.RECTANGLE,
            )
            circuit.add(0, gi, merge=True)
            trainable_prefixes.append(f"W{d}")

            # Diagonal phases complete U(m_c) expressibility for this block.
            # The final block's phi is omitted: those phases lie immediately
            # before photon-number measurement and are physically invisible
            # (|<n|e^{iφ}|ψ>|² = |<n|ψ>|²), so they always have zero gradient.
            if d < depth:
                for j in range(m_c):
                    circuit.add(j, pcvl.PS(pcvl.P(f"phi{d}_{j}")))
                trainable_prefixes.append(f"phi{d}")

                # Angle encoding (inserted between blocks, not after the last one).
                # Parameter names are unique per depth layer (x{d*n_features+j+1})
                # so that depth>1 re-uploading does not conflict in Perceval.
                # The input tensor is repeated (see parity_expectation) so that
                # layer d receives input[d*(m-1) : (d+1)*(m-1)] = the same x.
                for j in range(n_features):
                    circuit.add(j, pcvl.PS(pcvl.P(f"x{d * n_features + j + 1}")))

        # Spread photons evenly to cover the full interferometer light cone,
        # matching the teacher's default_input_state in data.photonic_quantum.
        input_state = _evenly_spaced_input_state(m_c, k)

        self.quantum_layer = ML.QuantumLayer(
            input_size=n_features * depth,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=trainable_prefixes,
            input_parameters=["x"],
            measurement_strategy=ML.MeasurementStrategy.probs(
                ML.ComputationSpace.FOCK
            ),
        )

        # Near-identity initialization: override the QuantumLayer's default
        # wide random init (randn * π) with a much smaller scale.  Starting
        # near phase=0 keeps the circuit in a region where gradients are
        # O(1) rather than exponentially small (barren-plateau avoidance).
        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.quantum_layer.parameters():
                    p.data.normal_(0.0, init_scale * math.pi)

        # Fixed parity vector: +1 / -1 per Fock state (not trainable)
        if self.learner_encoding == "parity":
            parity_vals = torch.tensor(
                [parity_of_key(key, self.parity_modes) #majority_score_of_key(key, m_c, k) #
                for key in self.quantum_layer.output_keys],
                dtype=torch.float32,
            )
        elif self.learner_encoding == "majority":
            parity_vals = torch.tensor(
                [majority_score_of_key(key, m_c, k) #
                for key in self.quantum_layer.output_keys],
                dtype=torch.float32,
            )
        self.register_buffer("parity_vec", parity_vals)  # (n_fock,)

        # Scalar bias b ∈ [-1, 1] (Havlíček et al. decision rule: sign(p + b)).
        # Trainable when use_bias=True (used with hloss); fixed at 0 otherwise.
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("bias", torch.zeros(1))

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parity_expectation(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous parity expectation ∈ [-1, +1], shape (B,)."""
        if self.depth > 1:
            x_in = x.repeat(1, self.depth)  # (B, depth*n_features)
        else:
            x_in = x
        probs = self.quantum_layer(x_in)          # (B, n_fock)
        return probs @ self.parity_vec            # (B,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, m-1)  angles in [0, 2π]

        Returns
        -------
        logits : (B, 2)  compatible with CrossEntropyLoss
            logits[:, 1] - logits[:, 0] = probs @ parity_vec + bias
        """
        logit = self.parity_expectation(x) + self.bias
        return torch.stack([-logit, logit], dim=1)  # (B, 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _get_params(model: nn.Module) -> torch.Tensor:
    """Flatten all trainable parameters into a 1-D tensor."""
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])


def _set_params(model: nn.Module, vec: torch.Tensor) -> None:
    """Copy a flat parameter vector back into the model in-place."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            n = p.numel()
            p.data.copy_(vec[offset:offset + n].reshape(p.shape))
            offset += n


def train_and_eval_quantum(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    m: int,
    k: int,
    depth: int = 1,
    m_circuit: int | None = None,
    n_features: int | None = None,
    data_encoding: str = "parity",
    learner_encoding: str = "parity",
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 64,
    seed: int = 0,
    parity_modes=None,
    optimizer_name: str = "Adam",
    loss: str = "ce",
    soft_train: torch.Tensor | None = None,
    soft_test: torch.Tensor | None = None,
    early_stopping_patience: int = 60,
) -> float:
    """Train one QuantumClassifier and return test accuracy.

    optimizer_name
        'Adam', 'AdamW', 'SGD' — gradient-based (uses epochs / lr / batch_size).
        'CMA'                  — CMA-ES black-box optimizer (ignores lr/batch_size;
                                 uses the full training set each evaluation;
                                 epochs is interpreted as the max number of
                                 objective function evaluations).
        'COBYLA', 'NelderMead', 'Powell' — scipy gradient-free methods (full dataset,
                                 epochs = max function evaluations).

    loss
        'ce'  — cross-entropy on hard ±/0/1 labels (default).
        'mse' — regress the student's continuous parity expectation against the
                teacher's continuous parity expectation. Requires soft_train and
                soft_test (both shape (N,)). Bypasses the weak-CE-gradient
                problem by giving a strong dense gradient signal.
    """
    if loss not in ("ce", "mse", "hloss"):
        raise ValueError(f"loss must be 'ce', 'mse', or 'hloss', got {loss!r}")
    if loss == "mse" and (soft_train is None or soft_test is None):
        raise ValueError("loss='mse' requires soft_train and soft_test (teacher parity expectation).")

    torch.manual_seed(seed)

    y_tr = y_train.long()
    y_te = y_test.long()

    use_bias = (loss == "hloss")
    model = QuantumClassifier(m=m, k=k, depth=depth, m_circuit=m_circuit,
                              n_features=n_features,
                              learner_encoding=learner_encoding,
                              parity_modes=parity_modes, init_scale=0.05,
                              use_bias=use_bias)
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    # Tracking across all branches; CMA/scipy don't have a val split so stay at -1
    best_val_acc = -1.0
    best_train_acc_at_best_val = 0.0

    if loss == "mse":
        s_tr = soft_train.view(-1).to(torch.float32)
    else:
        s_tr = None

    def _train_loss(model: nn.Module, x: torch.Tensor, yb: torch.Tensor, sb: torch.Tensor | None):
        if loss == "mse":
            return mse_criterion(model.parity_expectation(x), sb)
        if loss == "hloss":
            # Havlíček et al. sigmoid loss: -log σ(y_±1 · (parity_exp + b))
            # Equivalent to BCE_with_logits(parity_exp + b, y) for y ∈ {0,1}.
            return F.binary_cross_entropy_with_logits(
                model.parity_expectation(x) + model.bias, yb.float()
            )
        return ce_criterion(model(x), yb)

    # ------------------------------------------------------------------ #
    # SPSA branch                                                          #
    # ------------------------------------------------------------------ #
    if optimizer_name == "SPSA":
        import numpy as np
        import copy

        # Spall (1992) recommended hyper-parameters.
        # a, c decay as a_k = a/(k+A+1)^alpha  and  c_k = c/(k+1)^gamma
        # with alpha=0.602, gamma=0.101 (proven optimal asymptotic rates).
        # We scale `a` so that the initial step is roughly lr * pi (same
        # intuition as Adam's lr).  `c` controls the finite-difference probe
        # width; c ~ 0.1 works well for circuit angles in [0, 2pi].
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        alpha = 0.602
        gamma = 0.101
        A     = 0.01 * epochs          # stability constant ≈ 10% of budget
        a     = lr * math.pi           # initial effective step size
        c     = 0.10                   # finite-difference probe width (radians)

        # Val split for monitoring (10% of train, not used for optimisation)
        n_val   = max(1, len(X_train) // 10)
        X_val   = X_train[-n_val:]
        y_val   = y_tr[-n_val:]
        X_tr2   = X_train[:-n_val]
        y_tr2   = y_tr[:-n_val]
        s_tr2   = s_tr[:-n_val] if s_tr is not None else None

        # Each SPSA "epoch" is one full pass over mini-batches (same as Adam).
        # Each mini-batch uses 2 forward passes (θ+cΔ and θ-cΔ).
        if s_tr2 is not None:
            loader = DataLoader(
                TensorDataset(X_tr2, y_tr2, s_tr2),
                batch_size=batch_size, shuffle=True,
            )
        else:
            loader = DataLoader(
                TensorDataset(X_tr2, y_tr2),
                batch_size=batch_size, shuffle=True,
            )

        rng        = np.random.default_rng(seed)
        best_state = None
        best_val_acc = -1.0
        best_train_acc_at_best_val = 0.0
        step = 0  # global step counter (for a_k, c_k decay)

        for ep in range(epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for batch in loader:
                if s_tr2 is not None:
                    xb, yb, sb = batch
                else:
                    xb, yb = batch
                    sb = None

                step += 1
                a_k = a / (step + A + 1) ** alpha
                c_k = c / (step + 1) ** gamma

                # Random ±1 Bernoulli perturbation vector
                delta = torch.tensor(
                    rng.choice([-1.0, 1.0], size=n_params),
                    dtype=torch.float32,
                )

                theta = _get_params(model).clone()

                with torch.no_grad():
                    _set_params(model, theta + c_k * delta)
                    loss_plus  = _train_loss(model, xb, yb, sb).item()

                    _set_params(model, theta - c_k * delta)
                    loss_minus = _train_loss(model, xb, yb, sb).item()

                    # SPSA gradient estimate
                    g_hat = torch.tensor(
                        (loss_plus - loss_minus) / (2.0 * c_k),
                        dtype=torch.float32,
                    ) / delta  # element-wise: g_i ≈ Δf / (2 c_k Δ_i)

                    # Update
                    _set_params(model, theta - a_k * g_hat)

                epoch_loss += (loss_plus + loss_minus) / 2.0
                n_batches  += 1

            if ep % 10 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    train_acc    = (model(X_tr2).argmax(1) == y_tr2).float().mean().item()
                    val_acc_ep   = (model(X_val).argmax(1) == y_val).float().mean().item()
                    avg_loss     = epoch_loss / max(n_batches, 1)
                if val_acc_ep > best_val_acc:
                    best_val_acc = val_acc_ep
                    best_train_acc_at_best_val = train_acc
                    best_state   = copy.deepcopy(model.state_dict())
                print(
                    f"[depth={depth}] epoch {ep:3d}/{epochs}"
                    f"  spsa_loss={avg_loss:.4f}"
                    f"  train_acc={train_acc:.4f}"
                    f"  val_acc={val_acc_ep:.4f}"
                    f"  (best={best_val_acc:.4f})"
                )
                model.train()

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[depth={depth}] ✓ Best val_acc={best_val_acc:.4f}")

    # ------------------------------------------------------------------ #
    # Other gradient-free methods (CMA, scipy)                            #
    # ------------------------------------------------------------------ #
    elif optimizer_name in ("CMA", "COBYLA", "NelderMead", "Powell"):
        import numpy as np

        with torch.no_grad():
            def objective(w: np.ndarray) -> float:
                _set_params(model, torch.tensor(w, dtype=torch.float32))
                return _train_loss(model, X_train, y_tr, s_tr).item()

        x0 = _get_params(model).numpy()
        max_evals = max(epochs, 10)  # epochs repurposed as budget

        if optimizer_name == "CMA":
            import cma
            # sigma0: initial step size in radians. MZI angles live in [0, 2π]
            # so σ=0.5 is a reasonable fraction of the search space.
            sigma0 = 0.5
            opts = cma.CMAOptions()
            opts["maxfevals"] = max_evals
            opts["verbose"] = -9        # silent
            opts["tolx"] = 1e-5
            opts["tolfun"] = 1e-5
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            es.optimize(objective)
            best_w = es.result.xbest
        else:
            from scipy.optimize import minimize
            method_map = {
                "COBYLA":     "COBYLA",
                "NelderMead": "Nelder-Mead",
                "Powell":     "Powell",
            }
            # scipy options differ by method: COBYLA uses rhobeg/maxiter,
            # Nelder-Mead and Powell use maxfev.
            if optimizer_name == "COBYLA":
                opts = {"maxiter": max_evals, "disp": False}
            else:
                opts = {"maxfev": max_evals, "disp": False}
            res = minimize(
                objective, x0,
                method=method_map[optimizer_name],
                options=opts,
            )
            best_w = res.x

        _set_params(model, torch.tensor(best_w, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    # Gradient-based branch                                                #
    # ------------------------------------------------------------------ #
    else:
        if loss == "mse":
            loader = DataLoader(
                TensorDataset(X_train, y_tr, s_tr),
                batch_size=batch_size, shuffle=True,
            )
        else:
            loader = DataLoader(
                TensorDataset(X_train, y_tr),
                batch_size=batch_size, shuffle=True,
            )
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # ReduceLROnPlateau keeps the LR high until the train loss stops
        # improving, then halves it.  This is much better than
        # CosineAnnealingLR(T_max=epochs) which decays the LR to 0 exactly when
        # the model finishes its 300-epoch budget — killing learning for circuits
        # that converge slowly (e.g. m=6 still improves at epoch 299).
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-5
        )

        best_state = None
        best_train_loss = float("inf")
        no_improve_epochs = 0

        # 10% of train as val for monitoring (not shuffled)
        n_val = max(1, len(X_train) // 10)
        X_val, y_val = X_train[-n_val:], y_tr[-n_val:]
        X_tr2, y_tr2 = X_train[:-n_val], y_tr[:-n_val]
        if loss == "mse":
            s_tr2 = s_tr[:-n_val]
            loader = DataLoader(
                TensorDataset(X_tr2, y_tr2, s_tr2),
                batch_size=batch_size, shuffle=True,
            )
        else:
            loader = DataLoader(
                TensorDataset(X_tr2, y_tr2),
                batch_size=batch_size, shuffle=True,
            )
        grad_list =[]

        model.train()
        for ep in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                if loss == "mse":
                    xb, yb, sb = batch
                    lv = _train_loss(model, xb, yb, sb)
                else:
                    xb, yb = batch
                    lv = _train_loss(model, xb, yb, None)
                lv.backward()
            
                
                grad_val = [p.grad.detach().cpu().numpy()
                    for p in model.parameters() if p.grad is not None]
                grad_val=numpy.concatenate(grad_val, axis=0)
                
                
                grad_list.append(grad_val)

                optimizer.step()
                epoch_loss += lv.item()
            
            scheduler.step(epoch_loss / len(loader))

            if ep % 10 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    train_acc = (model(X_tr2).argmax(1) == y_tr2).float().mean().item()
                    val_acc_ep = (model(X_val).argmax(1) == y_val).float().mean().item()
                    avg_loss = epoch_loss / len(loader)
                val_improved  = val_acc_ep  > best_val_acc
                loss_improved = avg_loss    < best_train_loss

                if val_improved:
                    best_val_acc = val_acc_ep
                    best_train_acc_at_best_val = train_acc
                    import copy
                    best_state = copy.deepcopy(model.state_dict())
                if loss_improved:
                    best_train_loss = avg_loss

                if not val_improved and not loss_improved:
                    no_improve_epochs += 10
                else:
                    no_improve_epochs = 0
                print(
                    f"[depth={depth}] epoch {ep:3d}/{epochs}"
                    f"  train_loss={avg_loss:.4f}"
                    f"  train_acc={train_acc:.4f}"
                    f"  val_acc={val_acc_ep:.4f}"
                    f"  (best={best_val_acc:.4f})"
                )
                model.train()
                if no_improve_epochs >= early_stopping_patience:
                    print(f"[depth={depth}] Early stop at epoch {ep} (no improvement for {no_improve_epochs} epochs)")
                    break
        
        numpy.save(f"grad_data\{m}_{k}grad_{data_encoding}_data_{learner_encoding}_learner_{int(best_val_acc*10000)}_best_acc.npy",numpy.array(grad_list))


        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[depth={depth}] ✓ Best val_acc={best_val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_te).float().mean().item()

    return accuracy, best_val_acc, best_train_acc_at_best_val


# ---------------------------------------------------------------------------
# Search over depths
# ---------------------------------------------------------------------------

def find_min_depth(
    m: int,
    k: int,
    generator: str = "quantum",
    target_accuracy: float = 0.90,
    dataset_size: int = 2000,
    depths: list[int] | None = None,
    m_circuits: list[int] | None = None,
    data_encoding: str = "parity",
    learner_encoding: str = "parity",
    optimizer_name: str = "Adam",
    test_fraction: float = 0.20,
    epochs: int = 300,
    lr: float = 1e-2,
    batch_size: int = 64,
    data_seed: int = 42,
    model_seed: int = 0,
    balanced: bool = False,
    loss: str = "ce",
    min_margin: float = 0.0,
    bail_threshold: float = 0.30,
    max_resample_iter: int = 10,
    early_stopping_patience: int = 60,
    observable: str = "parity",
    n_features: int | None = None,
    nsample: int = 0,
) -> None:
    """Find the minimum (m_circuit, depth) configuration reaching the target accuracy.

    depth=1 is the single sandwich T1 → encode(x) → T2 that mirrors the teacher.
    depth>1 adds more data re-uploading layers.
    m_circuits controls the interferometer size independently from the dataset's m;
    when m_circuit > m the extra modes add expressivity with evenly-spaced photons.
    """
    if depths is None:
        depths = [1, 2, 3, 4]
    if m_circuits is None:
        m_circuits = [m]  # default: circuit size matches dataset

    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator {generator!r}. Choose from {list(GENERATORS)}.")

    margin_str = f"  min_margin={min_margin}" if min_margin > 0.0 else ""
    obs_str    = f"  observable={observable}" if generator == "photonic_quantum" else ""
    nf_str     = f"  n_features={n_features}" if n_features is not None else ""
    nshots_str = f"  nsample={nsample}" if nsample > 0 else ""
    print(f"Generating dataset  generator={generator}  m={m}  k={k}  size={dataset_size}"
          f"  balanced={balanced}{margin_str}{obs_str}{nf_str}{nshots_str} ...")
    gen_kwargs: dict = dict(size=dataset_size, m=m, k=k, seed=data_seed,
                            balanced=balanced, min_margin=min_margin,
                            bail_threshold=bail_threshold,
                            max_iter=max_resample_iter,
                            nsample=nsample)
    if generator == "photonic_quantum":
        gen_kwargs["observable"] = observable
        if n_features is not None:
            gen_kwargs["n_features"] = n_features
    ds = GENERATORS[generator](**gen_kwargs)

    if loss == "mse" and ds.soft_targets is None:
        raise ValueError(
            f"loss='mse' requires soft_targets in the dataset; generator={generator!r} "
            "does not provide them."
        )

    n_test = int(len(ds.X) * test_fraction)
    n_train = len(ds.X) - n_test
    if ds.soft_targets is not None:
        # Carry the (N, 1) soft target as an extra column so random_split keeps
        # rows and targets in lock-step.
        soft_col = ds.soft_targets.view(-1)
        full = TensorDataset(ds.X, ds.y, soft_col)
    else:
        full = TensorDataset(ds.X, ds.y)
    train_ds, test_ds = random_split(
        full,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(data_seed),
    )
    X_train, y_train = train_ds[:][0], train_ds[:][1]
    X_test, y_test = test_ds[:][0], test_ds[:][1]
    if ds.soft_targets is not None:
        soft_train, soft_test = train_ds[:][2], test_ds[:][2]
    else:
        soft_train = soft_test = None

    parity_modes = ds.metadata.get("parity_modes", None)

    print(f"Train: {n_train}  Test: {n_test}")
    print(f"Target accuracy: {target_accuracy:.0%}")

    # Pre-training sanity check: for each student size, can depth=1 RECTANGLE represent the teacher?
    if "teacher_W1" in ds.metadata:
        validate_teacher_params(ds.metadata, m_circuits)

    multi_size = len(m_circuits) > 1
    if multi_size:
        print(f"\n{'m_circ':>7}  {'depth':>6}  {'params':>8}  {'accuracy':>10}")
        print("-" * 38)
    else:
        print(f"\n{'depth':>6}  {'params':>8}  {'accuracy':>10}")
        print("-" * 30)

    eff_n_features = n_features if n_features is not None else m - 1

    def _n_params(depth: int, m_circuit: int) -> int:
        return QuantumClassifier(m=m, k=k, depth=depth,
                                 m_circuit=m_circuit, n_features=eff_n_features,
                                 parity_modes=parity_modes).n_trainable

    found = None
    for m_c in m_circuits:
        for d in depths:
            acc, best_val, best_train = train_and_eval_quantum(
                X_train, y_train, X_test, y_test,
                m=m, k=k, depth=d, m_circuit=m_c,
                n_features=eff_n_features,
                data_encoding=data_encoding,
                learner_encoding=learner_encoding,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=model_seed,
                parity_modes=parity_modes,
                optimizer_name=optimizer_name,
                loss=loss,
                soft_train=soft_train,
                soft_test=soft_test,
                early_stopping_patience=early_stopping_patience,
            )
            marker = " <-- first hit" if found is None and acc >= target_accuracy else ""
            extra = "" if acc >= target_accuracy else f"  (best val={best_val:.2%} @ train={best_train:.2%})"
            if multi_size:
                print(f"{m_c:>7}  {d:>6}  {_n_params(d, m_c):>8}  {acc:>10.2%}{extra}{marker}")
            else:
                print(f"{d:>6}  {_n_params(d, m_c):>8}  {acc:>10.2%}{extra}{marker}")
            if found is None and acc >= target_accuracy:
                found = (m_c, d)
                break
        if found is not None:
            break

    print()
    if found is not None:
        m_c, d = found
        print(f"Minimal config to reach {target_accuracy:.0%}: m_circuit={m_c}  depth={d}")
    else:
        print(f"No config in m_circuits={m_circuits} × depths={depths} reached {target_accuracy:.0%}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a variational quantum sandwich circuit and find the minimal depth."
    )
    parser.add_argument(
        "--generator",
        choices=list(GENERATORS),
        default="quantum",
        help="Dataset generation strategy (default: quantum)",
    )
    parser.add_argument("--m", type=int, default=6, help="Number of optical modes")
    parser.add_argument("--k", type=int, default=3, help="Photons / complexity parameter")
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--dataset-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=0)
    cli = parser.parse_args()

    find_min_depth(
        m=cli.m,
        k=cli.k,
        generator=cli.generator,
        target_accuracy=cli.target_accuracy,
        dataset_size=cli.dataset_size,
        depths=cli.depths,
        epochs=cli.epochs,
        lr=cli.lr,
        data_seed=cli.data_seed,
        model_seed=cli.model_seed,
    )
