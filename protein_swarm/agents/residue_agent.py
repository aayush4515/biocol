"""
Residue Agent — the core per-position mutation proposer.

Each agent examines its local neighbourhood, consults memory, and proposes
a single residue substitution (or no-op).

Two execution paths:
  1. Heuristic (default) — fast, deterministic, no API calls
  2. LLM-backed — sends a structured prompt to an LLM and parses JSON response
Toggle via AgentInput.use_llm.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from protein_swarm.schemas import (
    AgentInput,
    GoalEvaluation,
    GlobalMemoryStats,
    MutationProposal,
    ObjectiveSpec,
    PositionMemorySummary,
    PositionMutationEvent,
    StructureContext,
)
from protein_swarm.utils.constants import (
    AMINO_ACIDS,
    HELIX_FAVORING,
    SHEET_FAVORING,
    HYDROPHOBIC,
    POLAR,
)

logger = logging.getLogger(__name__)


def run_residue_agent_local(agent_input: AgentInput) -> MutationProposal:
    """Execute the residue agent logic for a single position."""
    if agent_input.use_llm:
        return _run_llm_agent(agent_input)
    return _run_heuristic_agent(agent_input)


def _run_llm_agent(agent_input: AgentInput) -> MutationProposal:
    """LLM-backed agent: build prompt, call model, return validated proposal."""
    from protein_swarm.agents.llm_client import call_llm_for_mutation

    prompt = build_agent_prompt(agent_input)
    pos = agent_input.position
    current = agent_input.sequence[pos]

    if agent_input.dump_prompt:
        _dump_prompt(agent_input.iteration, pos, prompt)

    return call_llm_for_mutation(
        prompt,
        api_key=agent_input.llm_api_key or "",
        provider=agent_input.llm_provider,
        model=agent_input.llm_model,
        temperature=agent_input.llm_temperature,
        max_tokens=agent_input.llm_max_tokens,
        max_retries=agent_input.llm_max_retries,
        position=pos,
        current_residue=current,
    )


def _run_heuristic_agent(agent_input: AgentInput) -> MutationProposal:
    """Original heuristic-based agent with deterministic scoring."""
    seq = agent_input.sequence
    pos = agent_input.position
    window = agent_input.neighbourhood_window
    mem = agent_input.memory_summary
    obj = agent_input.objective
    mutation_rate = agent_input.mutation_rate

    rng = _build_rng(agent_input.random_seed, pos)
    current = seq[pos]

    mutation_prob = mutation_rate * (mem.mutation_bias if mem else 1.0)
    if rng.random() > mutation_prob:
        return _no_op(pos, current, "Below mutation probability threshold")

    neighbourhood = _extract_neighbourhood(seq, pos, window)
    candidates = _rank_candidates(current, neighbourhood, obj, mem)
    if not candidates:
        return _no_op(pos, current, "No viable candidates")

    proposed, confidence, reason = candidates[0]
    if mem and proposed in mem.rejected_residues:
        confidence *= 0.7
        reason += " (previously rejected — confidence dampened)"

    return MutationProposal(
        position=pos,
        current_residue=current,
        proposed_residue=proposed,
        confidence=round(confidence, 4),
        reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Paper-style prompt builder (PART 1–4)
# ═══════════════════════════════════════════════════════════════════════════════

_DSSP_LABELS: dict[str, str] = {
    "H": "Alpha helix (H)",
    "G": "3-10 helix (G)",
    "I": "Pi helix (I)",
    "E": "Extended strand (E)",
    "B": "Beta bridge (B)",
    "T": "Turn/bend/coil (T/S/-)",
    "S": "Turn/bend/coil (T/S/-)",
    "C": "Turn/bend/coil (T/S/-)",
    "-": "Turn/bend/coil (T/S/-)",
    "UNKNOWN": "Unknown (no DSSP data)",
}

_SS_CATEGORY: dict[str, str] = {
    "H": "Alpha helix", "G": "3-10 helix", "I": "Pi helix",
    "E": "Extended beta strand", "B": "Beta bridge",
    "T": "Turn/bend/coil/loop", "S": "Turn/bend/coil/loop",
    "C": "Turn/bend/coil/loop", "-": "Turn/bend/coil/loop",
    "UNKNOWN": "Unknown",
}


def build_agent_prompt(agent_input: AgentInput) -> str:
    """Build a PART 1–4 structured user prompt matching the ProteinSwarm paper."""
    seq = agent_input.sequence
    pos = agent_input.position
    current = seq[pos]
    obj = agent_input.objective
    sc = agent_input.structure_context
    gm = agent_input.global_memory_stats
    pos_history = agent_input.position_history
    nbr_history = agent_input.neighborhood_history
    goal_eval = agent_input.goal_evaluation
    mem = agent_input.memory_summary

    parts: list[str] = []

    goal_text = obj.raw_text if obj else "Optimise protein stability and function"

    # ═════════════════════════════════════════════════════════════════════
    # PART 1: Your Role and Task
    # ═════════════════════════════════════════════════════════════════════
    p1: list[str] = [
        "PART 1: Your Role and Task",
        f"Design Goal: {goal_text}",
        f"Position: {pos}",
        f"Current residue: {current}",
        f"Current full sequence: {seq}",
        "",
        "Decision Rules:",
        f"1. Most importantly, consider the design goal: {goal_text}",
        "2. Learn from the memory history, context of the current position, "
        "sequence and spatial neighbors, and structure and energy feedback "
        "to inform decision on no mutation or mutation",
        "3. Consider fundamental folding principles",
        "4. Maintain sequence diversity to avoid repetitive residues but "
        "still consider residues mentioned in the design goal",
        "",
        "Fundamental Folding Principles:",
        "- Minimize disruption to secondary structures",
        "- Favor compact, stable folding with low Rosetta energy",
        "- Prefer conservative mutations unless necessary",
        "- Consider hydrophobic core stability and surface accessibility",
        "- Alpha-helix favoring residues: A, E, L, M, Q, K, R, H",
        "- Beta-sheet favoring residues: V, I, Y, F, W, T",
        "- Turn/coil promoting residues: G, P, D, N, S",
        "- Proline (P) is a strong helix breaker; Glycine (G) is very flexible",
        "- Hydrophobic residues (A, V, I, L, M, F, W, P) pack in the core",
        "- Charged residues (D, E, K, R) are typically surface-exposed",
        "- Cysteine (C) can form disulfide bonds — use deliberately",
        "",
        "Your Task:",
        f"Choose the best amino acid for position {pos} to achieve the design "
        f"goal: {goal_text} considering the above decision rules and "
        "fundamental folding principles",
    ]
    parts.append("\n".join(p1))

    # ═════════════════════════════════════════════════════════════════════
    # PART 2: Local Neighborhood Context
    # ═════════════════════════════════════════════════════════════════════
    p2: list[str] = ["", "PART 2: Local Neighborhood Context"]

    if sc:
        ss_label = sc.secondary_structure
        ss_desc = _DSSP_LABELS.get(ss_label, f"Unknown ({ss_label})")
        ss_cat = _SS_CATEGORY.get(ss_label, "Unknown")

        dssp_available = ss_label != "UNKNOWN"
        analysis_method = "DSSP (high confidence)" if dssp_available else "Unavailable"

        p2.append("Secondary structure:")
        p2.append(f"- Position {pos} secondary structure: {ss_desc}")
        p2.append(f"- Analysis method: {analysis_method}")
        p2.append(f"- {ss_cat}")
        p2.append("")

        lin_n = sc.linear_neighbors_n or "(N-terminus)"
        lin_c = sc.linear_neighbors_c or "(C-terminus)"
        p2.append(
            f"Linear neighbors: {len(sc.linear_neighbors_n)} N-terminus neighbor(s): "
            f"{lin_n}, {len(sc.linear_neighbors_c)} C-terminus neighbor(s): {lin_c}"
        )

        if sc.spatial_neighbors:
            n_spatial = len(sc.spatial_neighbors)
            p2.append(
                f"Spatial neighbors from distance matrix within 8.0 Å: {n_spatial} total"
            )
            p2.append("Detailed spatial neighbors sorted by distance:")
            for sn in sc.spatial_neighbors:
                p2.append(f"- Position {sn.position}: {sn.residue} (distance: {sn.distance:.1f}Å)")
            p2.append("")

            # spatial context interpretation
            if n_spatial <= 4:
                spatial_desc = f"Very few spatial contacts ({n_spatial} neighbors) - likely highly exposed surface"
            elif n_spatial <= 8:
                spatial_desc = f"Not so spatially connected ({n_spatial} neighbors) - likely exposed surface"
            elif n_spatial <= 14:
                spatial_desc = f"Moderately connected ({n_spatial} neighbors) - partially buried"
            else:
                spatial_desc = f"Highly connected ({n_spatial} neighbors) - deeply buried core"

            p2.append(f"Spatial context: {spatial_desc}")

            if sc.contact_density <= 6:
                p2.append(
                    "Mutate or keep the current residue according to the design goal; "
                    "might favor polar, charged, or functionally relevant residues."
                )
            elif sc.contact_density >= 12:
                p2.append(
                    "Deeply buried — strongly prefer hydrophobic residues for core packing. "
                    "Avoid charged/polar unless the design goal requires it."
                )
            else:
                p2.append(
                    "Semi-buried position — balance between core packing and surface "
                    "accessibility based on the design goal."
                )

        p2.append("")
        p2.append("Detailed structural context from distance matrix analysis:")
        p2.append("Structural summary:")

        # compactness
        if sc.avg_local_distance < 6.0:
            compact_desc = (
                "Very compact and tightly packed environment, "
                "average distance smaller than 6 Å."
            )
        elif sc.avg_local_distance < 8.0:
            compact_desc = (
                f"Moderately compact environment, "
                f"average distance {sc.avg_local_distance:.1f} Å."
            )
        else:
            compact_desc = (
                f"Extended/loose environment, "
                f"average distance {sc.avg_local_distance:.1f} Å."
            )
        p2.append(
            f"- Compactness measured by average distance to 5 residues to the "
            f"N-terminus and 5 residues to the C-terminus: {compact_desc}"
        )

        # contact density
        cd = sc.contact_density
        if cd <= 3:
            cd_desc = f"Low contact density ({cd} neighbors) - few constraints, exposed."
        elif cd <= 8:
            cd_desc = f"Moderate contact density ({cd} neighbors) - moderately constrained, 4-8 neighbors."
        else:
            cd_desc = f"High contact density ({cd} neighbors) - highly constrained, tightly packed."
        p2.append(
            f"- Contact density measured by number of neighbors within 8.0 Å: {cd_desc}"
        )

        # flexibility
        std = sc.std_local_distance
        if std < 1.0:
            flex_desc = f"Rigid position, standard deviation of local distances < 1.0 Å."
        elif std < 2.0:
            flex_desc = f"Moderately flexible position, standard deviation of local distances {std:.1f} Å."
        else:
            flex_desc = f"Highly flexible position, standard deviation of local distances {std:.1f} Å."
        p2.append(
            f"- Structural flexibility measured by standard deviation of local distances: {flex_desc}"
        )

        # region prediction
        region = sc.region_guess
        region_map = {
            "helical": "Local geometry potentially suggests helical environment.",
            "sheet/strand": "Local geometry suggests extended beta-strand environment.",
            "buried": "Position appears deeply buried in the protein core.",
            "surface-exposed": "Position appears surface-exposed.",
            "loop-like": "Local geometry suggests loop or turn region.",
            "intermediate": "Intermediate structural environment.",
        }
        region_desc = region_map.get(region, f"Region classification: {region}")
        p2.append(f"- Secondary structure region predicted based on distance patterns: {region_desc}")

    else:
        w = agent_input.neighbourhood_window
        start = max(0, pos - w)
        end = min(len(seq), pos + w + 1)
        local = seq[start:end]
        p2.append(f"Linear window [{start}:{end}]: {local}")
        p2.append("(No structural context available — no PDB yet)")

    parts.append("\n".join(p2))

    # ═════════════════════════════════════════════════════════════════════
    # PART 3: Memory History and Analysis
    # ═════════════════════════════════════════════════════════════════════
    p3: list[str] = ["", "PART 3: Memory History and Analysis"]

    # global patterns
    p3.append("Global Patterns from All Agents from Memory:")
    if gm and gm.total_iterations > 0:
        p3.append(f"- Total iterations completed: {gm.total_iterations}")
        p3.append(f"- Overall acceptance rate: {gm.acceptance_rate:.1%}")
        p3.append(f"- Recent acceptance rate in the last 5 iterations: {gm.recent_acceptance_rate:.1%}")
        p3.append(f"- Energy trend: {gm.energy_trend}")
    else:
        p3.append("- No iterations completed yet.")
    p3.append("")

    # previous iteration outcomes (accepted/rejected with energies)
    if gm and gm.total_iterations > 0:
        p3.append("Previous iteration outcomes:")
        # get from global stats recent_scores and iteration data passed via events
        acc_strs: list[str] = []
        rej_strs: list[str] = []
        # reconstruct from position + neighborhood history since we have rosetta scores
        all_events = list(pos_history) + list(nbr_history)
        seen_iters: dict[int, tuple[bool, float | None]] = {}
        for ev in all_events:
            if ev.iteration not in seen_iters:
                energy_str = f"{ev.rosetta_total_score:.1f}" if ev.rosetta_total_score is not None else "N/A"
                seen_iters[ev.iteration] = (ev.accepted, ev.rosetta_total_score)
                label = f"Iter{ev.iteration}({energy_str})"
                if ev.accepted:
                    acc_strs.append(label)
                else:
                    rej_strs.append(label)
        if acc_strs:
            p3.append(f"- Accepted iterations and energies: {acc_strs}")
        if rej_strs:
            p3.append(f"- Rejected iterations and energies: {rej_strs}")
        p3.append("")

    # personal mutations at this position
    if pos_history:
        p3.append(
            f"Personal Mutations at Position {pos} and Local Neighborhood "
            "Mutations from Memory:"
        )
        for ev in reversed(pos_history[-10:]):
            status = "ACCEPTED" if ev.accepted else "REJECTED"
            reason_short = ev.reason[:60] if ev.reason else "N/A"
            energy_str = f"{ev.rosetta_total_score:.1f}" if ev.rosetta_total_score is not None else "N/A"
            score_str = f"{ev.design_goal_score:.1f}" if ev.design_goal_score is not None else "N/A"
            n_mut = ev.num_mutations_in_iteration or 0

            p3.append(f"Iter {ev.iteration}: {ev.from_res}->{ev.to_res} {status} for reason of {reason_short}")
            p3.append(f" Iteration details: Energy: {energy_str}, Score: {score_str}, {n_mut} mutations")
            p3.append(f" Total energy: {energy_str}")
        p3.append("")

    # local neighborhood mutations analysis
    if nbr_history:
        w = agent_input.neighbourhood_window
        start = max(0, pos - w)
        end = min(len(seq), pos + w + 1)
        local_context = seq[start:end]

        p3.append("Local Neighborhood Mutations Analysis:")
        p3.append(f"- Current local context: {local_context}")

        # neighborhood events summary
        p3.append(f"- Local mutation patterns near position {pos}:")
        for ev in nbr_history[:5]:
            status = "ACC" if ev.accepted else "REJ"
            energy_str = f"{ev.rosetta_total_score:.1f}" if ev.rosetta_total_score is not None else "N/A"
            p3.append(
                f"  pos {ev.position}: {ev.from_res}->{ev.to_res} "
                f"({status}, iter {ev.iteration}, energy: {energy_str})"
            )
        p3.append("")

    # personal mutations analysis (statistics)
    if pos_history:
        p3.append(f"Personal Mutations Analysis at Position {pos}:")

        # compute analysis from events
        events = pos_history
        total_ev = len(events)
        accepted_ev = [e for e in events if e.accepted]
        rejected_ev = [e for e in events if not e.accepted]

        # acceptance trend (first half vs second half)
        if total_ev >= 4:
            mid = total_ev // 2
            first_rate = sum(1 for e in events[:mid] if e.accepted) / mid
            second_rate = sum(1 for e in events[mid:] if e.accepted) / (total_ev - mid)
            if second_rate > first_rate + 0.1:
                trend = "improving"
            elif second_rate < first_rate - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient data"
        p3.append(f"- Acceptance trend (first half vs second half): {trend}")

        # rejected patterns
        if rejected_ev:
            pair_counts: dict[str, int] = defaultdict(int)
            for e in rejected_ev:
                pair_counts[f"{e.from_res}->{e.to_res}"] += 1
            patterns = [
                f"'{pair} ({count / len(rejected_ev) * 100:.1f}%)'"
                for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1])
            ]
            p3.append(f"- Your rejected patterns: [{', '.join(patterns[:5])}]")

        # recent trend last 10
        recent = events[-10:]
        recent_acc = sum(1 for e in recent if e.accepted)
        if len(recent) >= 3:
            r_rate = recent_acc / len(recent)
            r_trend = "improving" if r_rate > 0.6 else ("declining" if r_rate < 0.3 else "mixed")
        else:
            r_trend = "insufficient data"
        p3.append(f"- Recent trend in the last 10 iterations: {r_trend}")

        # residues accepted/rejected with counts
        acc_counts: dict[str, int] = defaultdict(int)
        for e in accepted_ev:
            acc_counts[e.to_res] += 1
        rej_counts: dict[str, int] = defaultdict(int)
        for e in rejected_ev:
            rej_counts[e.to_res] += 1

        if acc_counts:
            p3.append(
                f"- Residues that were successfully accepted at this position "
                f"and the number of times they were accepted: {dict(acc_counts)}"
            )
        if rej_counts:
            p3.append(
                f"- Residues that were rejected at this position "
                f"and the number of times they were rejected: {dict(rej_counts)}"
            )

        # current residue performance
        cur_acc = sum(1 for e in events if e.to_res == current and e.accepted)
        cur_tot = sum(1 for e in events if e.to_res == current)
        if cur_tot > 0:
            p3.append(f"- Current residue {current} performance: {cur_acc}/{cur_tot} accepted")

        # avg energy with current
        cur_energies = [
            e.rosetta_total_score for e in events
            if e.to_res == current and e.rosetta_total_score is not None
        ]
        if cur_energies:
            avg_e = sum(cur_energies) / len(cur_energies)
            p3.append(f"- Average total energy with {current}: {avg_e:.2f}")

        p3.append("")

    # mutation recommendations
    if pos_history and len(pos_history) >= 2:
        p3.append(f"Mutation Recommendations for {current} at Position {pos} from Memory:")

        residue_stats: dict[str, tuple[int, int]] = {}
        for e in pos_history:
            if e.to_res == current:
                continue
            acc, tot = residue_stats.get(e.to_res, (0, 0))
            residue_stats[e.to_res] = (acc + (1 if e.accepted else 0), tot + 1)

        highly: list[str] = []
        recommended: list[str] = []
        avoid: list[str] = []
        for res, (a, t) in sorted(residue_stats.items(), key=lambda x: -x[1][0] / max(x[1][1], 1)):
            rate = a / t if t > 0 else 0
            label = f"{res} ({rate:.2f} ({a}/{t}))"
            if rate >= 0.7 and t >= 1:
                highly.append(label)
            elif rate >= 0.3:
                recommended.append(label)
            else:
                avoid.append(label)

        if highly:
            p3.append(f"- Highly recommended (and past success rate): {highly}")
        if recommended:
            p3.append(f"- Recommended (and past success rate): {recommended}")
        if avoid:
            p3.append(f"- Avoid (and past success rate): {avoid}")
        p3.append("")

    # one-line summary
    if pos_history:
        recent_5 = pos_history[-5:]
        recent_rej = sum(1 for e in recent_5 if not e.accepted)
        if recent_rej >= 4:
            summary = "Multiple recent rejections - strongly prefer conservative mutations or no-op"
        elif recent_rej >= 2:
            summary = "Recent rejections detected - consider energy-favorable mutations"
        elif recent_rej == 0 and len(recent_5) >= 2:
            summary = "Recent mutations accepted - moderate exploration acceptable"
        else:
            summary = "Mixed recent results - balance exploration with stability"
        p3.append(f"One-line Summary from Memory:")
        p3.append(summary)
        p3.append("")

    # past experiences and energy changes
    if pos_history:
        p3.append(f"Your Relevant Past Experiences and Energy Changes at Position {pos}:")
        for ev in reversed(pos_history[-5:]):
            status = "accepted" if ev.accepted else "rejected"
            if ev.rosetta_total_score is not None:
                p3.append(
                    f"- Iter {ev.iteration}: {ev.from_res}->{ev.to_res} → {status} "
                    f"(ΔE: {ev.rosetta_total_score:.1f})"
                )
            else:
                p3.append(f"- Iter {ev.iteration}: {ev.from_res}->{ev.to_res} → {status}")

    parts.append("\n".join(p3))

    # ═════════════════════════════════════════════════════════════════════
    # PART 4: Design Goal and Energy Analysis
    # ═════════════════════════════════════════════════════════════════════
    p4: list[str] = ["", "PART 4: Design Goal and Energy Analysis"]

    if goal_eval:
        rating_desc = {
            "POOR": "Limited match to design goal",
            "OK": "Moderate match to design goal",
            "GOOD": "Strong match to design goal",
        }.get(goal_eval.rating, "")

        p4.append("Design Goal Evaluation:")
        p4.append(
            f"- Current design goal score: {goal_eval.goal_score:.1f}/100, "
            f"{goal_eval.rating}: {rating_desc}"
        )

        if goal_eval.key_aspects:
            p4.append("- Key aspects:")
            for k, v in goal_eval.key_aspects.items():
                if k == "helix_propensity_%":
                    helix_frac = v
                    p4.append(
                        f"  • helix_structure: "
                        f"{'Strong' if helix_frac > 50 else 'Limited'} helix content: "
                        f"{helix_frac:.1f}% helix-forming residues"
                    )
                elif k == "diversity":
                    p4.append(
                        f"  • sequence_diversity: "
                        f"{'Good' if v > 50 else 'Limited'} diversity: {v:.0f}% unique amino acids"
                    )
                elif k == "dssp_helix_%":
                    p4.append(f"  • dssp_helix_content: {v:.1f}% assigned helix by DSSP")
                elif k == "dssp_sheet_%":
                    p4.append(f"  • dssp_sheet_content: {v:.1f}% assigned sheet by DSSP")
                elif k == "hydrophobic_%":
                    p4.append(f"  • hydrophobic_content: {v:.1f}%")
                elif k == "longest_repeat":
                    p4.append(f"  • longest_repeat_stretch: {int(v)} residues")
                else:
                    p4.append(f"  • {k}: {v}")
        p4.append("")

        if goal_eval.recommendations:
            p4.append(
                f"Structure and Design Goal Score Recommendations for "
                f"{current} at Position {pos}:"
            )
            for r in goal_eval.recommendations:
                p4.append(f"- {r}")
    else:
        p4.append("(No goal evaluation available yet)")

    parts.append("\n".join(p4))

    # ═════════════════════════════════════════════════════════════════════
    # Output instructions
    # ═════════════════════════════════════════════════════════════════════
    output = (
        "\n\nBased on the above analysis, choose the best amino acid for "
        f"position {pos} (currently '{current}').\n"
        "If no beneficial mutation exists, propose the CURRENT residue "
        "with confidence 0.0.\n\n"
        "Respond with ONLY a JSON object (no markdown, no explanation):\n"
        '{"position": <int>, "proposed_residue": "<single AA letter>", '
        '"confidence": <float 0.0-1.0>, "reason": "<brief explanation>"}'
    )
    parts.append(output)

    return "\n".join(parts)


# ── Prompt dump for debugging ────────────────────────────────────────────────

def _dump_prompt(iteration: int, position: int, prompt: str) -> None:
    try:
        out_dir = Path("outputs/debug/prompts")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"iter_{iteration}_pos_{position}.txt"
        path.write_text(prompt)
    except Exception as e:
        logger.warning("Failed to dump prompt: %s", e)


# ── Heuristic helpers ─────────────────────────────────────────────────────────

def _build_rng(seed: int | None, position: int) -> random.Random:
    if seed is not None:
        combined = int(hashlib.sha256(f"{seed}-{position}".encode()).hexdigest(), 16)
        return random.Random(combined)
    return random.Random()


def _extract_neighbourhood(seq: str, pos: int, window: int) -> str:
    start = max(0, pos - window)
    end = min(len(seq), pos + window + 1)
    return seq[start:end]


def _no_op(pos: int, current: str, reason: str) -> MutationProposal:
    return MutationProposal(
        position=pos,
        current_residue=current,
        proposed_residue=current,
        confidence=0.0,
        reason=reason,
    )


def _rank_candidates(
    current: str,
    neighbourhood: str,
    objective: ObjectiveSpec | None,
    memory: PositionMemorySummary | None,
) -> list[tuple[str, float, str]]:
    scored: list[tuple[str, float, str]] = []

    for aa in AMINO_ACIDS:
        if aa == current:
            continue
        score = 0.0
        reasons: list[str] = []

        if objective and objective.favour_helix and aa in HELIX_FAVORING:
            score += 0.4
            reasons.append("helix-favoring")
        if objective and objective.favour_sheet and aa in SHEET_FAVORING:
            score += 0.35
            reasons.append("sheet-favoring")
        if objective and objective.favour_stability:
            if current in POLAR and aa in HYDROPHOBIC:
                score += 0.25
                reasons.append("hydrophobic core packing")
            if current in HYDROPHOBIC and aa in HYDROPHOBIC:
                score += 0.15
                reasons.append("conserved hydrophobicity")
        if aa in neighbourhood:
            score -= 0.2
            reasons.append("already in neighbourhood — penalised")
        if memory:
            if aa in memory.accepted_residues:
                score += 0.2
                reasons.append("previously accepted")
            if aa in memory.rejected_residues:
                score -= 0.15
                reasons.append("previously rejected")

        if score > 0:
            scored.append((aa, min(score, 1.0), "; ".join(reasons)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
