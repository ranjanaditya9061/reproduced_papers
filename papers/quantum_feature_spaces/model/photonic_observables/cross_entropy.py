"""``xent`` / ``xent_<base>``: ``O(x) = Σ_n <base>(n) · p_x(n) log q(n)``.

``q`` is the FIXED reference distribution the circuit produces with every encoded feature set to
zero -- ``q(n) = |<n| W2 W1 |in>|^2``, the sandwich with all phase shifters at 0.  It is the
unencoded interference pattern, identical for every input, so the score asks: *where does the
encoding steer probability mass, relative to the bare circuit?*  Outcomes that the unencoded
circuit already favours contribute a small ``|log q|``; mass pushed into outcomes the bare
circuit suppresses contributes a large negative one.

**This is LINEAR in ``p``.**  Because ``q`` does not depend on ``x``, ``Σ_n p_x(n) log q(n)`` is
just the ordinary expectation ``<log q>`` of a fixed, bounded diagonal observable -- so the
family builds a plain :class:`~.base.LinearObservable` over the score vector ``log q`` and needs
no new score shape.  Two consequences worth being explicit about:

* Unlike :mod:`.entropy` (``p log p``, non-polynomial) this sits in the classically *easy*
  regime: it is additive-error estimable from single-shot samples in ~1/eps^2 shots, by averaging
  ``log q`` over them.  It is a useful probe, not a hardness witness.
* It is the second half of the KL divergence.  With :mod:`.entropy` returning ``Σ p log p`` and
  this returning ``Σ p log q``, ``KL(p || q) = ent(x) - xent(x)`` -- and the hardness of that
  combination lives entirely in the ``ent`` term.

``base`` weights each outcome's log-reference contribution, exactly as in :mod:`.sq` and
:mod:`.entropy`; bare ``xent`` weights everything by ``1``.

Sign, and how to label -- ``log q(n) <= 0`` (probabilities are ``<= 1``), so as in :mod:`.entropy`
the weights decide entirely.  Measured at ``m=6, k=3``: bare ``xent`` and ``xent_n_first`` are
non-positive (0% positive -- ``n_first``'s weight is the non-negative ``{0, 1}``, so it cannot
flip a term), so threshold those; ``xent_parity`` (~60%) and ``xent_majority`` (~38%) are
genuinely mixed-sign and label by sign; ``xent_bunching`` (~95%) is mixed but heavily biased, so
threshold it too.

``q`` is not persisted in a saved distribution, so re-scoring offline needs it passed back in
(see :func:`model.photonic.score_from_distribution`).  It is fully determined by the circuit,
i.e. by ``(m, k, n_features, seed)``, all of which already fix the dataset identity -- so this
family contributes no extra hash fields.

NOTE on the clamp: ``log q`` is clamped at :data:`XENT_EPS`.  Unlike :mod:`.entropy`'s clamp --
where ``p = 0`` kills its own term exactly, so clamping is the true ``p log p -> 0`` limit --
this one is a genuine regularisation: if ``q(n) ~ 0`` while ``p_x(n) > 0`` the exact score is
``-inf``, and the clamp replaces it with the finite bound ``p_x(n) log(XENT_EPS)``.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .base import (LinearObservable, Observable, ObservableContext, ObservableFamily,
                   base_score_vec, register)
from .majority import require_even_m
from .sq import SQ_BASES

#: Per-Fock-state bases usable under an ``xent_<base>`` observable -- the same elementary
#: scorers ``sq_<base>`` / ``ent_<base>`` accept.  Bare ``xent`` means weight ``1`` everywhere.
XENT_BASES = SQ_BASES

#: Floor for ``q`` before taking its log, bounding ``log q >= log(XENT_EPS) ~ -27.6``.  See the
#: module docstring: this is a regularisation, not an exact limit.
XENT_EPS = 1e-12

_XENT_RE = re.compile(r"^xent(?:_(.+))?$")


def parse_xent_observable(observable: str):
    """Split into ``(is_xent, base)``; ``base`` is ``None`` for bare ``xent`` (weight ``1``).

    Plain observables return ``(False, observable)``, matching
    :func:`~.sq.parse_sq_observable` / :func:`~.entropy.parse_ent_observable`.
    """
    mo = _XENT_RE.match(observable)
    if mo is None:
        return False, observable
    return True, mo.group(1)


def is_xent_observable(observable: str) -> bool:
    """True for ``xent`` or a well-formed ``xent_<base>`` (``base`` in :data:`XENT_BASES`)."""
    is_xent, base = parse_xent_observable(observable)
    return is_xent and (base is None or base in XENT_BASES)


def reference_log_probs(reference_probs, *, eps: float = XENT_EPS) -> np.ndarray:
    """``log q``, floored at ``eps`` so the vector is finite and bounded below."""
    return np.log(np.maximum(np.asarray(reference_probs, dtype=np.float64), float(eps)))


def xent_score_vec(reference_probs, base: str | None, keys, *, m: int, k: int,
                   eps: float = XENT_EPS) -> np.ndarray:
    """Score vector ``<base>(n) · log q(n)`` for an ``xent[_<base>]`` observable.

    ``base=None`` (bare ``xent``) leaves the weight at ``1``, i.e. the vector is just ``log q``.
    """
    log_q = reference_log_probs(reference_probs, eps=eps)
    if base is None:
        return log_q
    if base == "majority":
        require_even_m(m, "xent_majority")
    weight = base_score_vec(base, ObservableContext(m=m, k=k, keys=keys),
                            allowed=XENT_BASES, label="xent")
    return weight * log_q


class XEntFamily(ObservableFamily):
    describe = f"xent / xent_<base> (base in {XENT_BASES})"

    def matches(self, name: str) -> bool:
        return is_xent_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        base = parse_xent_observable(name)[1]
        q = ctx.resolve_reference_probs(f"observable {name!r}")
        obs = LinearObservable(xent_score_vec(q, base, ctx.keys, m=ctx.m, k=ctx.k))
        # q itself is not needed to score (it is folded into score_vec), but keep it reachable
        # for inspection and so a caller can hand it to an offline re-score.
        obs.register_buffer("reference_probs", torch.as_tensor(q, dtype=torch.float32))
        return obs


register(XEntFamily())
