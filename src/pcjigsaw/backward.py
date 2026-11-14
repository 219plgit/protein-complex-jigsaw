"""Backward (disassembly) policies over the seed-anchored state graph.

In the main construction every mechanism is its own terminal object and has a
unique trajectory, so the GFlowNet backward policy is trivially 1. Here the
object is instead the *assembled complex*: the single terminal is the full set
of components, and the seed-anchored histories are the trajectories that reach
it. States are sets of placed components, so a state can have several parents
(``{seed, A, B}`` is reached by adding A then B or B then A). A backward policy

    P_B(S -> S \\ {p})       "which component dissociates next"

then defines a distribution over assembly pathways, and the matching forward
policy follows from detailed balance:

    F(full) = 1,   F(S \\ {p}) += F(S) P_B(S -> S \\ {p}),
    P_F(S -> S u {p}) = F(S u {p}) P_B(S u {p} -> S) / F(S).

Sampling forward with P_F gives every history h the probability
prod_t P_B along h read in reverse (Proposition in Supplementary Section S11).

Biologically, P_B is the natural place for *disassembly* evidence (solution
disruption in native MS, interface strength), whereas forward potentials carry
encounter evidence. Using disassembly evidence to weight assembly pathways
assumes microscopic reversibility; that assumption is a modelling decision and
is admitted explicitly, like any other relation.

Admissibility is unchanged: P_B is defined only on reverse edges of legal
additions, and every admissible history keeps positive probability whenever
P_B > 0 on those edges (the analogue of Proposition 3).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Tuple

from .prob import TypedModel, seeded_legal, seeded_F

Component = str
State = FrozenSet[Component]          # placed components, excluding the seed
BackwardPolicy = Callable[[State], Dict[Component, float]]


# ---------------------------------------------------------------------------
# state graph
# ---------------------------------------------------------------------------
def reachable_states(model: TypedModel, seed: Component) -> List[State]:
    """All seed-anchored states reachable by legal additions (contact rule)."""
    rest = tuple(v for v in model.V if v != seed)
    seen = {frozenset()}
    frontier = [frozenset()]
    while frontier:
        nxt = []
        for S in frontier:
            for p in rest:
                if p not in S and seeded_legal(model, seed, p, S):
                    T = S | {p}
                    if T not in seen:
                        seen.add(T)
                        nxt.append(T)
        frontier = nxt
    return sorted(seen, key=lambda s: (len(s), sorted(s)))


def parents(model: TypedModel, seed: Component, S: State, states: Iterable[State]) -> List[Component]:
    """Components whose removal leads to a reachable state from which their
    addition is legal: the admissible reverse edges of S."""
    states = set(states)
    return [p for p in sorted(S)
            if (S - {p}) in states and seeded_legal(model, seed, p, S - {p})]


# ---------------------------------------------------------------------------
# backward policies
# ---------------------------------------------------------------------------
def uniform_backward(model: TypedModel, seed: Component) -> BackwardPolicy:
    """Every admissible dissociation is equally likely (no disassembly evidence)."""
    states = reachable_states(model, seed)

    def pb(S: State) -> Dict[Component, float]:
        ps = parents(model, seed, S, states)
        return {p: 1.0 / len(ps) for p in ps}
    return pb


def interface_backward(model: TypedModel, seed: Component,
                       interface: Mapping[FrozenSet[Component], float],
                       beta: float, scale: float = 100.0) -> BackwardPolicy:
    """Disassembly by interface strength: a component bound to the rest of the
    complex through fewer interface residues dissociates more readily,

        b(p | S) = exp(-beta * sum_{q in S u {seed}, (p,q) in C} n_pq / scale),

    normalised over the admissible dissociations of S. beta = 0 is uniform.
    `interface` maps each contact to its number of interface residues; the
    coefficient beta is declared and swept, not fitted."""
    states = reachable_states(model, seed)

    def pb(S: State) -> Dict[Component, float]:
        placed = S | {seed}
        ps = parents(model, seed, S, states)
        w = {}
        for p in ps:
            n = sum(interface.get(frozenset((p, q)), 0.0) for q in placed - {p}
                    if model.has_contact(p, q))
            w[p] = math.exp(-beta * n / scale)
        z = sum(w.values())
        return {p: v / z for p, v in w.items()}
    return pb


def counting_backward(model: TypedModel, seed: Component) -> BackwardPolicy:
    """The backward policy that makes every history equally likely:
    P_B(S -> S \\ {p}) = N_in(S \\ {p}) / N_in(S), with N_in the number of
    partial histories reaching a state. The unit-reward exact flow of the main
    text is this particular backward policy."""
    states = reachable_states(model, seed)

    @lru_cache(None)
    def n_in(S: State) -> int:
        if not S:
            return 1
        return sum(n_in(S - {p}) for p in parents(model, seed, S, states))

    def pb(S: State) -> Dict[Component, float]:
        ps = parents(model, seed, S, states)
        tot = n_in(S)
        return {p: n_in(S - {p}) / tot for p in ps}
    return pb


# ---------------------------------------------------------------------------
# exact flows from a backward policy
# ---------------------------------------------------------------------------
def backward_flow(model: TypedModel, seed: Component, pb: BackwardPolicy) -> Dict[State, float]:
    """State flows induced by P_B from the single terminal (total flow 1):
    F(S) is the probability that a pathway passes through S."""
    states = reachable_states(model, seed)
    full = max(states, key=len)
    assert len(full) == len(model.V) - 1, "the full complex is not seed-anchored reachable"
    F = {s: 0.0 for s in states}
    F[full] = 1.0
    for S in sorted(states, key=len, reverse=True):
        if not S or F[S] == 0.0:
            continue
        for p, q in pb(S).items():
            F[S - {p}] += F[S] * q
    return F


def forward_from_backward(model: TypedModel, seed: Component, pb: BackwardPolicy):
    """The forward policy consistent with P_B by detailed balance:
    P_F(S -> S u {p}) = F(S u {p}) P_B(S u {p} -> S) / F(S)."""
    F = backward_flow(model, seed, pb)

    def pf(S: State) -> Dict[Component, float]:
        out = {}
        for T, fT in F.items():
            if len(T) == len(S) + 1 and S < T:
                (p,) = tuple(T - S)
                q = pb(T).get(p, 0.0)
                if q > 0:
                    out[p] = fT * q / F[S]
        return out
    return pf, F


def histories(model: TypedModel, seed: Component) -> List[Tuple[Component, ...]]:
    rest = tuple(v for v in model.V if v != seed)
    out = []

    def grow(S, order):
        if len(order) == len(rest):
            out.append(tuple(order)); return
        for p in rest:
            if p not in S and seeded_legal(model, seed, p, frozenset(S)):
                grow(S | {p}, order + [p])
    grow(frozenset(), [])
    return out


def history_probability_backward(h, pb: BackwardPolicy) -> float:
    """prod of P_B along the history read in reverse (disassembly order)."""
    S = frozenset(h); prob = 1.0
    for p in reversed(h):
        prob *= pb(S).get(p, 0.0)
        S = S - {p}
    return prob


def history_probability_forward(h, pf) -> float:
    S = frozenset(); prob = 1.0
    for p in h:
        prob *= pf(S).get(p, 0.0)
        S = S | {p}
    return prob


def pathway_distribution(model: TypedModel, seed: Component, pb: BackwardPolicy) -> Dict[Tuple, float]:
    return {h: history_probability_backward(h, pb) for h in histories(model, seed)}


def effective_number(dist: Mapping) -> float:
    ps = [p for p in dist.values() if p > 0]
    return math.exp(-sum(p * math.log(p) for p in ps))


def prefix_support(dist: Mapping, block: Iterable[Component], seed: Component) -> float:
    """Probability that the subcomplex {seed} u block is an intermediate."""
    block = frozenset(block) - {seed}
    k = len(block)
    return sum(p for h, p in dist.items() if frozenset(h[:k]) == block)


# ---------------------------------------------------------------------------
# relation-count score, backward entropy and beta-sensitivity diagnostics
# ---------------------------------------------------------------------------
def relation_count_backward(model: TypedModel, seed: Component, beta: float) -> BackwardPolicy:
    """Minimal relation-count disassembly score: b(p | S) = n_C + n_P, the
    admitted contacts and prerequisite arcs between p and the rest of the
    complex; P_B proportional to exp(-beta * b). Included for comparison: on an
    acyclic contact graph n_C = 1 for every admissible dissociation (the
    seed-anchored analogue of Proposition 5), and n_P counts prerequisites that
    may be implied by the contacts."""
    states = reachable_states(model, seed)

    def pb(S: State) -> Dict[Component, float]:
        rest_of = S | {seed}
        ps = parents(model, seed, S, states)
        w = {}
        for p in ps:
            others = rest_of - {p}
            n_c = sum(1 for q in others if model.has_contact(p, q))
            n_p = sum(1 for a, b in model.P if (a == p and b in others) or (b == p and a in others))
            w[p] = math.exp(-beta * (n_c + n_p))
        z = sum(w.values())
        return {p: v / z for p, v in w.items()}
    return pb


def backward_entropy(pb: BackwardPolicy, S: State) -> float:
    """H_B(S) = -sum_p P_B(S -> S \\ {p}) log P_B(...), in nats: how many
    reverse explanations of state S remain comparably plausible."""
    return -sum(q * math.log(q) for q in pb(S).values() if q > 0)


def flow_weighted_backward_entropy(model: TypedModel, seed: Component, pb: BackwardPolicy) -> float:
    """Average backward entropy over states with several parents, weighted by
    the probability that a pathway passes through each state."""
    F = backward_flow(model, seed, pb)
    st = [S for S in F if len(pb(S)) > 1]
    tot = sum(F[S] for S in st)
    return sum(F[S] * backward_entropy(pb, S) for S in st) / tot if tot else 0.0


def beta_sensitivity(model: TypedModel, seed: Component, make_pb: Callable[[float], BackwardPolicy],
                     betas: Iterable[float], block: Iterable[Component]) -> Dict[str, object]:
    """Delta_beta(Q) = max_beta S_beta(Q) - min_beta S_beta(Q) for the query
    'the subcomplex {seed} u block is an intermediate', over a declared range of
    beta. A large value marks the conclusion as evidence-sensitive, not
    evidence-determined."""
    vals = {b: prefix_support(pathway_distribution(model, seed, make_pb(b)), block, seed) for b in betas}
    return {"supports": vals, "delta": max(vals.values()) - min(vals.values())}


# ---------------------------------------------------------------------------
# usefulness of evidence: linear response of a conclusion to trusting a score
# ---------------------------------------------------------------------------
Score = Callable[[Component, State], float]   # b(p | S): evidence anchoring p in S


def interface_score(model: TypedModel, seed: Component,
                    interface: Mapping[FrozenSet[Component], float], scale: float = 100.0) -> Score:
    """Interface residues binding p to the rest of the complex (per `scale`)."""
    def b(p, S):
        placed = S | {seed}
        return sum(interface.get(frozenset((p, q)), 0.0) for q in placed - {p}
                   if model.has_contact(p, q)) / scale
    return b


def relation_count_score(model: TypedModel, seed: Component) -> Score:
    """Admitted contacts plus prerequisite arcs between p and the rest."""
    def b(p, S):
        others = (S | {seed}) - {p}
        return (sum(1 for q in others if model.has_contact(p, q)) +
                sum(1 for a, c in model.P if (a == p and c in others) or (c == p and a in others)))
    return b


def score_backward(model: TypedModel, seed: Component, score: Score, beta: float) -> BackwardPolicy:
    """P_B(S -> S \\ {p}) proportional to exp(-beta * b(p | S)) over admissible dissociations."""
    states = reachable_states(model, seed)

    def pb(S: State) -> Dict[Component, float]:
        ps = parents(model, seed, S, states)
        w = {p: math.exp(-beta * score(p, S)) for p in ps}
        z = sum(w.values())
        return {p: v / z for p, v in w.items()}
    return pb


def _centered_load(model, seed, score, h) -> float:
    """D(h) = sum over the dissociations of h (read in reverse) of
    b(p | S) minus the mean of b over the admissible dissociations of S."""
    states = reachable_states(model, seed)
    S = frozenset(h); d = 0.0
    for p in reversed(h):
        ps = parents(model, seed, S, states)
        d += score(p, S) - sum(score(q, S) for q in ps) / len(ps)
        S = S - {p}
    return d


def evidence_scale(model: TypedModel, seed: Component, score: Score) -> float:
    """Root-mean-square of the centred score over dissociations, weighted by
    the beta = 0 pathway distribution. Dividing b by it puts beta in units of
    'one typical difference in evidence' and makes scores comparable."""
    states = reachable_states(model, seed)
    pb0 = uniform_backward(model, seed)
    F = backward_flow(model, seed, pb0)
    num = den = 0.0
    for S, f in F.items():
        ps = parents(model, seed, S, states)
        if len(ps) < 2:
            continue
        mean = sum(score(q, S) for q in ps) / len(ps)
        for q in ps:
            num += f * pb0(S)[q] * (score(q, S) - mean) ** 2
            den += f * pb0(S)[q]
    return math.sqrt(num / den) if den and num > 0 else 0.0


def evidence_usefulness(model: TypedModel, seed: Component, score: Score,
                        block: Iterable[Component], beta_inf: float = 60.0) -> Dict[str, float]:
    """How useful the evidence summarised by `score` is for the query
    'the subcomplex {seed} u block is an intermediate':

      S0        support when the evidence is ignored (beta = 0);
      U         dS/dbeta at beta = 0, with b in units of evidence_scale:
                U = -Cov_0(1_Q, D), the first-order effect of starting to trust
                the evidence (0: the evidence is silent on Q; sign: direction);
      S_inf     support when the evidence is treated as a rule (large beta).
    """
    sc = evidence_scale(model, seed, score)
    if sc == 0:
        s0 = prefix_support(pathway_distribution(model, seed, uniform_backward(model, seed)), block, seed)
        return {"S0": s0, "U": 0.0, "S_inf": s0, "scale": 0.0}
    std = lambda p, S: score(p, S) / sc
    d0 = pathway_distribution(model, seed, uniform_backward(model, seed))
    block = frozenset(block) - {seed}
    k = len(block)
    ind = {h: 1.0 if frozenset(h[:k]) == block else 0.0 for h in d0}
    D = {h: _centered_load(model, seed, std, h) for h in d0}
    eI = sum(d0[h] * ind[h] for h in d0)
    eD = sum(d0[h] * D[h] for h in d0)
    cov = sum(d0[h] * ind[h] * D[h] for h in d0) - eI * eD
    s_inf = prefix_support(pathway_distribution(model, seed, score_backward(model, seed, std, beta_inf)), block, seed)
    return {"S0": eI, "U": -cov, "S_inf": s_inf, "scale": sc}
