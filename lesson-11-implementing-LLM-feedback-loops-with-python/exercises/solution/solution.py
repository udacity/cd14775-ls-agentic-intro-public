#!/usr/bin/env python3
"""
Self-Correcting Trial Synopsis Generator (Life Sciences Agentic AI – Course 1)

Implements an LLM feedback loop where gpt-4o-mini iteratively revises a
regulatory-style clinical trial synopsis based on programmatic evaluation of
required metadata (PICO, phase, enrollment, and primary endpoint).

Uses:
- OpenAI: model="gpt-4o-mini", base_url="https://openai.vocareum.com/v1"
- OPENAI_API_KEY loaded via dotenv from ".env"
- ClinicalTrials.gov access via ls_action_space.action_space.query_clinicaltrials
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Action space (provided in the workspace)
from ls_action_space.action_space import query_clinicaltrials


# -----------------------------
# OpenAI client + helpers
# -----------------------------
def get_client() -> OpenAI:
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Ensure you have a .env file with OPENAI_API_KEY=... "
            "or set the environment variable."
        )
    return OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")


def chat_complete(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# -----------------------------
# Trial data normalization
# -----------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "this",
    "that", "these", "those", "measure", "measures", "outcome", "outcomes",
    "primary", "secondary", "endpoint", "endpoints", "trial", "study",
}

@dataclass
class TrialBrief:
    nct_id: Optional[str]
    title: Optional[str]
    phase: Optional[str]
    enrollment: Optional[int]
    primary_endpoint: Optional[str]
    conditions: Optional[List[str]]
    interventions: Optional[List[str]]
    comparator_hint: Optional[str]
    population_hint: Optional[str]

    def to_prompt_json(self) -> Dict[str, Any]:
        # Keep prompts small and audit-friendly: only include what the model needs.
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "phase": self.phase,
            "enrollment": self.enrollment,
            "primary_endpoint": self.primary_endpoint,
            "conditions": self.conditions,
            "interventions": self.interventions,
            "comparator_hint": self.comparator_hint,
            "population_hint": self.population_hint,
        }


def _safe_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _first_str(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _first_int(*vals: Any) -> Optional[int]:
    for v in vals:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            m = re.search(r"\d+", v.replace(",", ""))
            if m:
                try:
                    return int(m.group(0))
                except ValueError:
                    pass
    return None


def _as_str_list(v: Any) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, list):
        out = [str(x).strip() for x in v if str(x).strip()]
        return out or None
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return None


def _extract_study_payload(trial_resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    query_clinicaltrials(...) is best-effort and may return different shapes.
    We defensively select the first study-like object.
    """
    if not isinstance(trial_resp, dict):
        return {}
    # Common: {"count": ..., "studies": [ ... ]}
    studies = trial_resp.get("studies")
    if isinstance(studies, list) and studies:
        if isinstance(studies[0], dict):
            return studies[0]
    # Sometimes already flattened single record
    return trial_resp


def build_trial_brief(trial_resp: Dict[str, Any]) -> TrialBrief:
    study = _extract_study_payload(trial_resp)

    # Try multiple common key patterns (flattened or nested CT.gov v2-like)
    nct_id = _first_str(
        study.get("nctId"), study.get("NCTId"), study.get("nct_id"), study.get("nct_id".upper()),
        _safe_get(study, "protocolSection", "identificationModule", "nctId"),
    )
    title = _first_str(
        study.get("briefTitle"), study.get("officialTitle"), study.get("title"),
        _safe_get(study, "protocolSection", "identificationModule", "briefTitle"),
        _safe_get(study, "protocolSection", "identificationModule", "officialTitle"),
    )
    phase = _first_str(
        study.get("phase"), study.get("phases"),
        _safe_get(study, "protocolSection", "designModule", "phases"),
        _safe_get(study, "protocolSection", "designModule", "phase"),
    )
    enrollment = _first_int(
        study.get("enrollment"), study.get("enrollmentCount"),
        _safe_get(study, "protocolSection", "designModule", "enrollmentInfo", "enrollmentCount"),
        _safe_get(study, "protocolSection", "designModule", "enrollmentInfo", "count"),
        _safe_get(study, "protocolSection", "designModule", "enrollmentCount"),
    )

    # Primary endpoint / primary outcome measure
    primary_endpoint = _first_str(
        study.get("primary_outcome"),
        study.get("primaryOutcome"),
        study.get("primaryEndpoint"),
        _safe_get(study, "protocolSection", "outcomesModule", "primaryOutcomeMeasures", 0, "measure"),
        _safe_get(study, "protocolSection", "outcomesModule", "primaryOutcomeMeasures", 0, "title"),
        _safe_get(study, "protocolSection", "outcomesModule", "primaryOutcomeMeasures", 0, "description"),
    )

    conditions = _as_str_list(
        study.get("conditions") or _safe_get(study, "protocolSection", "conditionsModule", "conditions")
    )

    # Interventions often nested; accept strings, dicts, or list of either.
    interventions_raw = (
        study.get("interventions")
        or _safe_get(study, "protocolSection", "armsInterventionsModule", "interventions")
        or _safe_get(study, "protocolSection", "armsInterventionsModule", "armGroups")
    )
    interventions: Optional[List[str]] = None
    if isinstance(interventions_raw, list):
        parts: List[str] = []
        for item in interventions_raw:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                # Common fields
                name = _first_str(item.get("name"), item.get("interventionName"), item.get("label"), item.get("type"))
                desc = _first_str(item.get("description"), item.get("interventionDescription"))
                if name and desc:
                    parts.append(f"{name} — {desc}")
                elif name:
                    parts.append(name)
        interventions = parts or None
    elif isinstance(interventions_raw, str) and interventions_raw.strip():
        interventions = [interventions_raw.strip()]

    # Eligibility criteria can be huge; we only use as a "population hint" if present.
    population_hint = _first_str(
        study.get("eligibilityCriteria"),
        _safe_get(study, "protocolSection", "eligibilityModule", "eligibilityCriteria"),
        _safe_get(study, "protocolSection", "eligibilityModule", "studyPopulation"),
    )
    if population_hint and len(population_hint) > 800:
        population_hint = population_hint[:800] + "…"

    # Comparator hint: sometimes trial record includes placebo/control/standard-of-care in arms or description.
    text_blob = " ".join(
        [x for x in [
            title,
            _first_str(study.get("briefSummary"), _safe_get(study, "protocolSection", "descriptionModule", "briefSummary")),
            _first_str(study.get("detailedDescription"), _safe_get(study, "protocolSection", "descriptionModule", "detailedDescription")),
            " ".join(interventions or []),
        ] if x]
    ).lower()
    comparator_hint = None
    if any(w in text_blob for w in ["placebo", "control", "standard of care", "versus", "vs. ", " vs "]):
        # Provide a light hint; LLM must still keep to "Not reported" if not explicit.
        comparator_hint = "Record mentions a control/comparator (e.g., placebo/control/standard-of-care). Use only if explicit."

    return TrialBrief(
        nct_id=nct_id,
        title=title,
        phase=phase,
        enrollment=enrollment,
        primary_endpoint=primary_endpoint,
        conditions=conditions,
        interventions=interventions,
        comparator_hint=comparator_hint,
        population_hint=population_hint,
    )


# -----------------------------
# Programmatic evaluation
# -----------------------------
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_label_value(text: str, label: str) -> Optional[str]:
    # Supports "Label: value" lines, case-insensitive, multiline.
    pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
    m = re.search(pattern, text)
    if m:
        v = m.group(1).strip()
        return v if v else None
    return None


def _extract_first_int(text: str) -> Optional[int]:
    m = re.search(r"(?<!\d)(\d{2,})(?!\d)", text.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _phase_tokens(phase: str) -> List[str]:
    # Make matching tolerant: "Phase 3", "Phase III", "PHASE3", etc.
    p = phase.strip()
    toks = []
    toks.append(_norm(p).replace(" ", ""))
    # also keep roman/arabic extracted
    m = re.search(r"(?i)\bphase\s*([0-9]+|i{1,4}v?)\b", p)
    if m:
        toks.append("phase" + _norm(m.group(1)).replace(" ", ""))
    return list(dict.fromkeys([t for t in toks if t]))


def _endpoint_keywords(endpoint: str) -> List[str]:
    words = [w for w in _norm(endpoint).split() if len(w) >= 4 and w not in STOPWORDS]
    # Cap to avoid overly strict matching
    return words[:8] if words else []


def evaluate_synopsis(synopsis: str, brief: TrialBrief) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """
    Checks:
      - PICO labels present (Population/Intervention/Comparison/Outcome)
      - Phase present and matches expected when available
      - Enrollment present and matches expected when available
      - Primary endpoint present and aligns with expected when available
    """
    details: Dict[str, Any] = {}
    text_norm = _norm(synopsis)

    pop = _extract_label_value(synopsis, "Population")
    itv = _extract_label_value(synopsis, "Intervention")
    cmp_ = _extract_label_value(synopsis, "Comparison")
    out = _extract_label_value(synopsis, "Outcome")
    phase_line = _extract_label_value(synopsis, "Phase")
    enrollment_line = _extract_label_value(synopsis, "Enrollment")
    endpoint_line = _extract_label_value(synopsis, "Primary endpoint")

    # Basic presence checks
    has_population = bool(pop)
    has_intervention = bool(itv)
    has_comparison = bool(cmp_)
    has_outcome = bool(out)

    # Phase check (required if source has a phase)
    phase_required = bool(brief.phase)
    has_phase = bool(phase_line) if phase_required else True
    phase_matches = True
    if phase_required and brief.phase:
        expected_tokens = _phase_tokens(brief.phase)
        synopsis_compact = text_norm.replace(" ", "")
        phase_matches = any(t in synopsis_compact for t in expected_tokens) or (
            phase_line is not None and any(t in _norm(phase_line).replace(" ", "") for t in expected_tokens)
        )

    # Enrollment check (required if source has enrollment)
    enrollment_required = brief.enrollment is not None
    has_enrollment = bool(enrollment_line) if enrollment_required else True
    enrollment_matches = True
    parsed_enroll = None
    if enrollment_required and brief.enrollment is not None:
        parsed_enroll = _first_int(enrollment_line) if enrollment_line else _extract_first_int(synopsis)
        if parsed_enroll is None:
            enrollment_matches = False
        else:
            # Accept exact, or within ±10% for safety against rounding language like "approximately"
            target = brief.enrollment
            enrollment_matches = (parsed_enroll == target) or (abs(parsed_enroll - target) <= max(3, int(0.10 * target)))

    # Primary endpoint check (required if source has it)
    endpoint_required = bool(brief.primary_endpoint)
    has_primary_endpoint = bool(endpoint_line) if endpoint_required else True
    endpoint_matches = True
    if endpoint_required and brief.primary_endpoint:
        kws = _endpoint_keywords(brief.primary_endpoint)
        # Match if any key terms appear, or the endpoint is explicitly "Not reported" (counts as fail if required)
        if endpoint_line and _norm(endpoint_line) == "not reported":
            endpoint_matches = False
        elif kws:
            endpoint_matches = sum(1 for w in kws if w in text_norm) >= max(1, min(2, len(kws)))
        else:
            endpoint_matches = True  # if no useful keywords, don't overconstrain

    checks = {
        "has_population": has_population,
        "has_intervention": has_intervention,
        "has_comparison": has_comparison,
        "has_outcome": has_outcome,
        "has_phase": has_phase,
        "phase_matches_source": phase_matches,
        "has_enrollment": has_enrollment,
        "enrollment_matches_source": enrollment_matches,
        "has_primary_endpoint": has_primary_endpoint,
        "primary_endpoint_matches_source": endpoint_matches,
    }

    details.update(
        {
            "extracted": {
                "Population": pop,
                "Intervention": itv,
                "Comparison": cmp_,
                "Outcome": out,
                "Phase": phase_line,
                "Enrollment": enrollment_line,
                "Primary endpoint": endpoint_line,
                "parsed_enrollment": parsed_enroll,
            },
            "expected": {
                "phase": brief.phase,
                "enrollment": brief.enrollment,
                "primary_endpoint": brief.primary_endpoint,
            },
        }
    )
    return checks, details


def generate_feedback(checks: Dict[str, bool], brief: TrialBrief, eval_details: Dict[str, Any]) -> Optional[str]:
    missing: List[str] = []
    mismatched: List[str] = []

    # PICO
    if not checks["has_population"]:
        missing.append("Population (include a 'Population:' line)")
    if not checks["has_intervention"]:
        missing.append("Intervention (include an 'Intervention:' line)")
    if not checks["has_comparison"]:
        missing.append("Comparison (include a 'Comparison:' line; use 'Not reported' ONLY if absent in source)")
    if not checks["has_outcome"]:
        missing.append("Outcome (include an 'Outcome:' line)")

    # Phase / Enrollment / Endpoint (only if expected exists)
    if brief.phase:
        if not checks["has_phase"]:
            missing.append("Phase (include a 'Phase:' line)")
        elif not checks["phase_matches_source"]:
            mismatched.append(f"Phase does not match source (expected: {brief.phase})")

    if brief.enrollment is not None:
        if not checks["has_enrollment"]:
            missing.append("Enrollment (include an 'Enrollment:' line with a number)")
        elif not checks["enrollment_matches_source"]:
            exp = brief.enrollment
            got = eval_details.get("extracted", {}).get("parsed_enrollment")
            mismatched.append(f"Enrollment does not match source (expected ≈ {exp}; extracted: {got})")

    if brief.primary_endpoint:
        if not checks["has_primary_endpoint"]:
            missing.append("Primary endpoint (include a 'Primary endpoint:' line)")
        elif not checks["primary_endpoint_matches_source"]:
            mismatched.append("Primary endpoint does not appear to align with the source text")

    if not missing and not mismatched:
        return None

    # Provide concise, constructive guidance and anti-hallucination guardrails.
    parts: List[str] = []
    if missing:
        parts.append("Missing required elements:\n- " + "\n- ".join(missing))
    if mismatched:
        parts.append("Items needing correction:\n- " + "\n- ".join(mismatched))

    guardrails = (
        "Revision rules:\n"
        "1) Use ONLY the provided trial record fields; do not invent details.\n"
        "2) Keep the same labeled format lines (Population/Intervention/Comparison/Outcome/Phase/Enrollment/Primary endpoint).\n"
        "3) If a PICO element is truly absent in the source, write 'Not reported' explicitly.\n"
        "4) Keep it concise (≤ 180 words)."
    )
    return "\n\n".join(parts + [guardrails])


# -----------------------------
# LLM prompting
# -----------------------------
def build_initial_prompt(brief: TrialBrief) -> str:
    trial_json = json.dumps(brief.to_prompt_json(), indent=2)
    return (
        "You are a regulatory medical writer creating an audit-friendly trial synopsis.\n"
        "Write a concise synopsis (≤ 180 words) grounded ONLY in the provided trial record JSON.\n\n"
        "Output format (must include these labels on their own lines):\n"
        "Trial ID: ...\n"
        "Title: ...\n"
        "Phase: ...\n"
        "Enrollment: ...\n"
        "Population: ...\n"
        "Intervention: ...\n"
        "Comparison: ...\n"
        "Outcome: ...\n"
        "Primary endpoint: ...\n\n"
        "Rules:\n"
        "- Do NOT include patient-identifiable data (PHI). Do NOT add citations.\n"
        "- If something is not in the record, write 'Not reported'.\n"
        "- Keep language factual and minimal.\n\n"
        f"TRIAL RECORD JSON:\n{trial_json}\n"
    )


def build_revision_prompt(brief: TrialBrief, prior_synopsis: str, feedback: str) -> str:
    trial_json = json.dumps(brief.to_prompt_json(), indent=2)
    return (
        "You are a regulatory medical writer revising a trial synopsis.\n"
        "Revise the synopsis to address the evaluation feedback.\n\n"
        "Constraints:\n"
        "- Ground the revision ONLY in the provided trial record JSON.\n"
        "- Preserve the required labeled lines:\n"
        "  Trial ID, Title, Phase, Enrollment, Population, Intervention, Comparison, Outcome, Primary endpoint\n"
        "- If something is not in the record, write 'Not reported' (do not guess).\n"
        "- ≤ 180 words. No citations. No PHI.\n\n"
        f"EVALUATION FEEDBACK:\n{feedback}\n\n"
        f"PRIOR SYNOPSIS:\n{prior_synopsis}\n\n"
        f"TRIAL RECORD JSON:\n{trial_json}\n"
    )


# -----------------------------
# Self-correcting loop
# -----------------------------
def self_correcting_loop(
    client: OpenAI,
    brief: TrialBrief,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = []
    feedback: Optional[str] = None
    synopsis: str = ""

    print("=" * 88)
    print("Self-Correcting Trial Synopsis Generator")
    print("=" * 88)
    print()

    for i in range(1, max_iterations + 1):
        print(f"\n{'─' * 88}\nITERATION {i}\n{'─' * 88}")

        if i == 1:
            prompt = build_initial_prompt(brief)
        else:
            assert feedback is not None
            prompt = build_revision_prompt(brief, synopsis, feedback)

        synopsis = chat_complete(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.2,
        )

        print("\nGenerated Synopsis:\n")
        print(synopsis)

        checks, eval_details = evaluate_synopsis(synopsis, brief)

        print("\nEvaluation Results:")
        for k in [
            "has_population", "has_intervention", "has_comparison", "has_outcome",
            "has_phase", "phase_matches_source",
            "has_enrollment", "enrollment_matches_source",
            "has_primary_endpoint", "primary_endpoint_matches_source",
        ]:
            status = "✓" if checks.get(k) else "✗"
            print(f"  {status} {k.replace('_', ' ')}")

        feedback = generate_feedback(checks, brief, eval_details)

        history.append(
            {
                "iteration": i,
                "synopsis": synopsis,
                "checks": checks,
                "eval_details": eval_details,
                "feedback": feedback,
                "all_passed": feedback is None,
            }
        )

        if feedback is None:
            print("\n" + "=" * 88)
            print("✓ SUCCESS: Synopsis meets required metadata checks.")
            print("=" * 88)
            break

        print("\nFeedback for revision:\n")
        print(feedback)

    return {
        "final_synopsis": synopsis,
        "iterations": len(history),
        "history": history,
        "success": history[-1]["all_passed"] if history else False,
    }


# -----------------------------
# Main
# -----------------------------
def fetch_trial_record(trial_query: str) -> Dict[str, Any]:
    """
    The action_space tool expects an expression (best-effort).
    Use an NCT ID (e.g., 'NCT01234567') or a general query string.
    """
    resp = query_clinicaltrials(trial_query, max_results=1)
    if not isinstance(resp, dict):
        return {"error": "query_clinicaltrials returned non-dict response"}
    if resp.get("count") in (0, None) and not resp.get("studies"):
        # Best-effort: still return resp for debugging
        return resp
    return resp


def main() -> None:
    client = get_client()

    # Prefer exact NCT ID, fallback to general query.
    trial_query = os.environ.get("NCT_ID") or os.environ.get("TRIAL_QUERY") or "NCT00000000"
    print("=" * 88)
    print("DEMO: Self-Correcting Trial Synopsis Generator")
    print("Module: Implementing LLM Feedback Loops with Python")
    print("=" * 88)
    print(f"\nQuerying ClinicalTrials.gov record for: {trial_query}\n")

    trial_resp = fetch_trial_record(trial_query)
    brief = build_trial_brief(trial_resp)

    if not any([brief.title, brief.phase, brief.enrollment, brief.primary_endpoint, brief.nct_id]):
        print("⚠ Warning: Trial record appears sparse or unrecognized. Raw response snippet:\n")
        print(json.dumps(trial_resp, indent=2)[:1200] + ("…" if len(json.dumps(trial_resp)) > 1200 else ""))
        print("\nProceeding anyway; checks will adapt based on available fields.\n")

    print("Parsed Trial Brief (used for prompting/evaluation):")
    print(json.dumps(brief.to_prompt_json(), indent=2))

    result = self_correcting_loop(client, brief, max_iterations=int(os.environ.get("MAX_ITERATIONS", "5")))

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"Iterations: {result['iterations']}")
    print(f"Success: {result['success']}")
    print("\nFinal Synopsis:\n")
    print(result["final_synopsis"])

    output_file = os.environ.get("OUTPUT_FILE", "synopsis_result.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trial_query": trial_query,
                "trial_brief": brief.to_prompt_json(),
                "final_synopsis": result["final_synopsis"],
                "iterations": result["iterations"],
                "success": result["success"],
                "history": result["history"],
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 88)


if __name__ == "__main__":
    main()

