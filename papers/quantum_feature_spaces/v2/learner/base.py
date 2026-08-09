"""``Learner`` ABC + registry, and the two adjudication statistics.

    learner = build_learner("ridge", order=3)
    learner.fit(X_train, y_train)
    res = evaluate(learner, X_test, y_test, sigma_sq=learner.residual_variance)

**Which statistic adjudicates depends on whether the labels carry noise, and the default is
noiseless -- so the decision table in :mod:`v2.learner.compare` uses ``R^2``.**

The design intent was to adjudicate on held-out log-likelihood, on the grounds that with exact
``probs`` available the mean held-out log-likelihood against the ideal model is an unbiased KL
estimate, stronger and less scale-sensitive than ``R^2``.  That holds when the labels carry genuine
noise (``generation.shots > 0``), where the ideal model has finite entropy and the gap really is a KL
divergence.  **It does not hold at the default ``generation.shots = 0``**: the labels are then deterministic
given ``x``, the ideal model's log-likelihood is ``+inf``, and no finite KL exists.  What
:func:`gaussian_log_likelihood` reports is the learner's *own* Gaussian predictive score, which is a
valid proper score but is dominated by ``sigma^2_train`` -- i.e. by the label's scale.

Measured, which is why this is stated rather than assumed: across the paired arms the label
variances differ by up to ``4.5e12`` (measured when the ``det`` arm was strict free fermions, where
``bunching`` is structurally *constant* because the bunched sector carries exactly zero mass -- the
current readout has full support, so that extreme is now reached only via
:meth:`v2.model.fermion.FermionModel.collision_free_probs`, but the spread remains large), and the paired
``log_likelihood`` difference there is ``-12.8`` while the ``R^2`` difference is ``+0.24``.  The two
disagree in sign because one renormalises by label variance and the other does not.

So: ``R^2`` is the adjudicator for the decision table, since "fraction of the available structure
captured" is exactly what the four-row logic asks and it is comparable across arms.  Both are
reported.  Revisit this and promote log-likelihood to primary when running at ``generation.shots > 0``, where
it becomes the KL estimate it was intended to be.
"""

from __future__ import annotations

import math

import torch

#: name -> Learner subclass, populated on subclassing.
LEARNERS: dict[str, type["Learner"]] = {}


class Learner:
    """``X -> y_hat`` for a real-valued observable score.  Subclasses implement fit/predict."""

    name: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            LEARNERS[cls.name] = cls

    def __init__(self, **hparams):
        self.hparams = hparams
        #: Train-set residual variance, the Gaussian predictive's ``sigma^2``.  Fitted on TRAIN
        #: only -- using the test residuals would make the log-likelihood self-normalising and the
        #: comparison vacuous.
        self.residual_variance: float = 1.0

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "Learner":
        raise NotImplementedError

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def spec(self) -> dict:
        return {"learner": self.name, **self.hparams}

    def _record_residual(self, X: torch.Tensor, y: torch.Tensor) -> None:
        resid = (y - self.predict(X)).double()
        self.residual_variance = max(float(resid.var(unbiased=True)), 1e-12)


def build_learner(name: str, **hparams) -> Learner:
    if name not in LEARNERS:
        raise ValueError(f"unknown learner {name!r}; registered: {sorted(LEARNERS)}")
    return LEARNERS[name](**hparams)


def r2_score(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    """``1 - SSE/SST``.  Secondary: normalised by the label's own variance, hence scale-sensitive."""
    y, y_hat = y.double(), y_hat.double()
    sse = float(((y - y_hat) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-30)


def gaussian_log_likelihood(y: torch.Tensor, y_hat: torch.Tensor, sigma_sq: float) -> float:
    """Mean held-out ``log N(y; y_hat, sigma^2)`` with ``sigma^2`` fitted on TRAIN.

    In the label's own units, so a paired difference between two arms is interpretable.  Higher is
    better; it is unbounded above as ``sigma^2 -> 0``, which is why ``sigma^2`` must come from the
    training residuals rather than the test ones.
    """
    y, y_hat = y.double(), y_hat.double()
    mse = float(((y - y_hat) ** 2).mean())
    return -0.5 * (math.log(2.0 * math.pi * sigma_sq) + mse / sigma_sq)


def evaluate(learner: Learner, X: torch.Tensor, y: torch.Tensor) -> dict:
    """Held-out log-likelihood (primary) and ``R^2`` (secondary)."""
    y_hat = learner.predict(X)
    return {
        "log_likelihood": gaussian_log_likelihood(y, y_hat, learner.residual_variance),
        "r2": r2_score(y, y_hat),
        "rmse": float(((y.double() - y_hat.double()) ** 2).mean().sqrt()),
        "sigma_sq_train": learner.residual_variance,
    }
