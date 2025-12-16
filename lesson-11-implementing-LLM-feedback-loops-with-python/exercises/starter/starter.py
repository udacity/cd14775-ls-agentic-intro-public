#!/usr/bin/env python3
"""
Self-Correcting Trial Synopsis Generator — STUDENT SCAFFOLD (Course 1)

Module: Implementing LLM Feedback Loops with Python
Learning objective:
  Implement a self-correcting feedback loop in Python where an LLM iteratively
  revises its own output based on programmatic evaluation.

✅ Everything is provided EXCEPT a few small TODOs that are *directly* about:
  (1) programmatic evaluation gates
  (2) generating feedback from those gates
  (3) wiring the revise-and-retry loop

Guardrails:
  - Use ONLY TRIAL RECORD JSON (no invented details)
  - If missing in record: write "Not reported"
  - No PHI, no citations
  - ≤ 180 words
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Provided in the workspace
from ls_action_space.action_space import query_clinicaltrials


# =============================================================================
# OpenAI client + helper (provided)
# =============================================================================
def get_client() -> OpenAI:
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to .env (OPENAI_API_KEY=...) or set it as an env var."
        )
    return OpenAI(api_key=api_key, base_url="https://openai.vocareum.com/v1")


def chat_complete(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# =============================================================================
# Trial record -> minimal prompt brief (provided; intentionally simple)
# =============================================================================
@dataclass
class TrialBrief:
    nct_id: Optional[str]
    title: Optional[str]
    phase: Optional[str]
    enrollment: Optional[int]
    primary_endpoint: Optional[str]

    def to_prompt_json(self) -> Dict[str, Any]:
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "phase": self.phase,
            "enrollment": self.enrollment,
            "primary_endpoint": self.primary_endpoint,
        }


def fetch_trial_record(trial_query: str) -> Dict[str, Any]:
    """Action space: use NCT ID (preferred) or a query string."""
    resp = query_clinicaltrials(trial_query, max_results=1)
    return resp if isinstance(resp, dict) else {"error": "query_clinicaltrials returned non-dict response"}


def _first_study(trial_resp: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(trial_resp, dict):
        return {}
    studies = trial_resp.get("studies")
    if isinstance(studies, list) and studies and isinstance(studies[0], dict):
        return studies[0]
    return trial_resp


def _parse_int_maybe(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"\d+", str(v).replace(",", ""))
    return int(m.group(0)) if m else None


def build_trial_brief(trial_resp: Dict[str, Any]) -> TrialBrief:
    """
    Keep parsing lightweight for Course 1—just enough for the feedback loop demo.
    """
    study = _first_study(trial_resp)

    # Try a few common key variants
    nct_id = study.get("nctId") or study.get("NCTId") or study.get("nct_id")
    title = study.get("briefTitle") or study.get("officialTitle") or study.get("title")
    phase = study.get("phase") or study.get("phases")
    enrollment = _parse_int_maybe(study.get("enrollment") or study.get("enrollmentCount"))
    primary_endpoint = (
        study.get("primaryEndpoint")
        or study.get("primary_outcome")
        or study.get("primaryOutcome")
        or study.get("primary_outcome_measure")
    )

    return TrialBrief(
        nct_id=nct_id,
        title=title,
        phase=phase if isinstance(phase, str) else (phase[0] if isinstance(phase, list) and phase else None),
        enrollment=enrollment,
        primary_endpoint=primary_endpoint if isinstance(primary_endpoint, str) else None,
    )


# =============================================================================
# Prompting (provided)
# =============================================================================
REQUIRED_LABELS = [
    "Trial ID",
    "Title",
    "Phase",
    "Enrollment",
    "Population",
    "Intervention",
    "Comparison",
    "Outcome",
    "Primary endpoint",
]


def build_initial_prompt(brief: TrialBrief) -> str:
    trial_json = json.dumps(brief.to_prompt_json(), indent=2)
    return (
        "You are a regulatory medical writer creating an audit-friendly trial synopsis.\n"
        "Write a concise synopsis (≤ 180 words) grounded ONLY in the TRIAL RECORD JSON.\n"
        "If something is not in the record, write 'Not reported'. No citations. No PHI.\n\n"
        "Output format (each label on its own line):\n"
        + "\n".join([f"{lab}: ..." for lab in REQUIRED_LABELS])
        + "\n\n"
        f"TRIAL RECORD JSON:\n{trial_json}\n"
    )


def build_revision_prompt(brief: TrialBrief, prior_synopsis: str, feedback: str) -> str:
    trial_json = json.dumps(brief.to_prompt_json(), indent=2)
    return (
        "You are a regulatory medical writer revising a trial synopsis.\n"
        "Revise the synopsis to address the feedback.\n"
        "Ground changes ONLY in the TRIAL RECORD JSON. Do not invent details.\n"
        "Keep the same labeled format. ≤ 180 words. No citations. No PHI.\n\n"
        f"FEEDBACK:\n{feedback}\n\n"
        f"PRIOR SYNOPSIS:\n{prior_synopsis}\n\n"
        f"TRIAL RECORD JSON:\n{trial_json}\n"
    )


# =============================================================================
# Programmatic evaluation gates (STUDENT TODO: small)
# =============================================================================
def _line_value(text: str, label: str) -> Optional[str]:
    m = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text)
    return m.group(1).strip() if m and m.group(1).strip() else None


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _phase_match(expected: str, observed: str) -> bool:
    """
    Very forgiving phase match: checks that the key token appears.
    Examples: "Phase 2" matches "Phase II" if both contain "phase" and some 2/ii token.
    """
    e = _norm(expected).replace(" ", "")
    o = _norm(observed).replace(" ", "")
    if "phase" not in e or "phase" not in o:
        return False
    # Accept if the compact expected appears, or key number/roman token overlaps
    return (e in o) or any(tok in o for tok in ["phase2", "phaseii", "phase3", "phaseiii", "phase1", "phasei"] if tok in e)


def _first_int_in_text(text: str) -> Optional[int]:
    m = re.search(r"(?<!\d)(\d{2,})(?!\d)", text.replace(",", ""))
    return int(m.group(1)) if m else None


def _endpoint_keywords(endpoint: str) -> List[str]:
    words = [w for w in _norm(endpoint).split() if len(w) >= 4]
    return words[:6]


def evaluate_synopsis(synopsis: str, brief: TrialBrief) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """
    Gates:
      - All PICO labels present (Population/Intervention/Comparison/Outcome)
      - If brief.phase exists: Phase present + matches (forgiving)
      - If brief.enrollment exists: Enrollment present + matches approx (±10%)
      - If brief.primary_endpoint exists: Primary endpoint present + not 'Not reported' + keyword overlap

    STUDENT TODOs are intentionally tiny and directly tied to the learning objective.
    """
    extracted = {lab: _line_value(synopsis, lab) for lab in REQUIRED_LABELS}

    checks: Dict[str, bool] = {
        "has_population": bool(extracted["Population"]),
        "has_intervention": bool(extracted["Intervention"]),
        "has_comparison": bool(extracted["Comparison"]),
        "has_outcome": bool(extracted["Outcome"]),
        "has_phase": True,                   # may become required
        "phase_matches_source": True,         # may become required
        "has_enrollment": True,               # may become required
        "enrollment_matches_source": True,    # may become required
        "has_primary_endpoint": True,         # may become required
        "primary_endpoint_matches_source": True,  # may become required
    }

    # ---- TODO 1 (Phase): if brief.phase exists, require Phase line and match it ----
    # Hint: use extracted["Phase"] and _phase_match(brief.phase, extracted["Phase"])
    if brief.phase:
        # TODO: set checks["has_phase"]
        # TODO: set checks["phase_matches_source"]
        pass

    # ---- TODO 2 (Enrollment): if brief.enrollment exists, require a number and match ±10% ----
    parsed_enrollment = None
    if brief.enrollment is not None:
        # TODO: set checks["has_enrollment"]
        # TODO: parse enrollment number into parsed_enrollment (use _first_int_in_text)
        # TODO: set checks["enrollment_matches_source"] using ±10% tolerance
        pass

    # ---- TODO 3 (Primary endpoint): if brief.primary_endpoint exists, require line + keyword overlap ----
    if brief.primary_endpoint:
        # TODO: set checks["has_primary_endpoint"]
        # TODO: set checks["primary_endpoint_matches_source"]
        #   - fail if extracted["Primary endpoint"] is exactly "Not reported"
        #   - else pass if >=1 keyword from brief.primary_endpoint appears anywhere in synopsis
        pass

    details = {
        "extracted": extracted,
        "parsed_enrollment": parsed_enrollment,
        "expected": brief.to_prompt_json(),
    }
    return checks, details


# =============================================================================
# Feedback generation (STUDENT TODO: small)
# =============================================================================
def generate_feedback(checks: Dict[str, bool], brief: TrialBrief, details: Dict[str, Any]) -> Optional[str]:
    """
    Return:
      - None if all gates pass (stop condition)
      - otherwise a short, actionable feedback string the LLM can follow

    STUDENT TODO: build feedback from failed checks (a few bullet points).
    """
    failed: List[str] = []

    # PICO
    if not checks["has_population"]:
        failed.append("Add 'Population:' line (or 'Not reported' if absent).")
    if not checks["has_intervention"]:
        failed.append("Add 'Intervention:' line (or 'Not reported' if absent).")
    if not checks["has_comparison"]:
        failed.append("Add 'Comparison:' line (use 'Not reported' ONLY if absent in record).")
    if not checks["has_outcome"]:
        failed.append("Add 'Outcome:' line (or 'Not reported' if absent).")

    # ---- TODO 4: add conditional feedback for phase/enrollment/endpoint failures (only if expected exists) ----
    # Examples:
    #  - if brief.phase and (missing or mismatch): tell them to match expected phase
    #  - if brief.enrollment is not None and mismatch: include expected enrollment value
    #  - if brief.primary_endpoint and mismatch: ask to align with record (no guessing)
    # Keep it short.
    # TODO: implement

    if not failed:
        return None

    rules = (
        "Rules: Use ONLY the TRIAL RECORD JSON (no invented details). "
        "Keep the exact labeled format. If truly missing, write 'Not reported'. "
        "≤ 180 words."
    )
    return "- " + "\n- ".join(failed) + "\n\n" + rules


# =============================================================================
# Self-correcting loop (STUDENT TODO: tiny)
# =============================================================================
def self_correcting_loop(
    client: OpenAI,
    brief: TrialBrief,
    max_iterations: int = 4,
) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = []
    synopsis = ""
    feedback: Optional[str] = None

    for i in range(1, max_iterations + 1):
        prompt = build_initial_prompt(brief) if i == 1 else build_revision_prompt(brief, synopsis, feedback or "")

        synopsis = chat_complete(client, prompt)

        checks, details = evaluate_synopsis(synopsis, brief)
        feedback = generate_feedback(checks, brief, details)

        history.append(
            {
                "iteration": i,
                "synopsis": synopsis,
                "checks": checks,
                "details": details,
                "feedback": feedback,
                "passed": feedback is None,
            }
        )

        # ---- TODO 5: stop early when the synopsis passes all gates ----
        # TODO: if feedback is None: break
        # (This is the “feedback loop” stopping condition.)
        # TODO: implement

    return {
        "final_synopsis": synopsis,
        "success": history[-1]["passed"] if history else False,
        "iterations": len(history),
        "history": history,
    }


# =============================================================================
# Main (provided)
# =============================================================================
def main() -> None:
    client = get_client()

    trial_query = os.environ.get("NCT_ID") or os.environ.get("TRIAL_QUERY") or "NCT00000000"
    print("=" * 88)
    print("DEMO (STUDENT SCAFFOLD): Self-Correcting Trial Synopsis Generator")
    print("Module: Implementing LLM Feedback Loops with Python")
    print("=" * 88)
    print(f"\nQuerying record for: {trial_query}\n")

    trial_resp = fetch_trial_record(trial_query)
    brief = build_trial_brief(trial_resp)

    print("TrialBrief used for prompting/eval:")
    print(json.dumps(brief.to_prompt_json(), indent=2))

    result = self_correcting_loop(client, brief, max_iterations=int(os.environ.get("MAX_ITERATIONS", "4")))

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print("Success:", result["success"])
    print("Iterations:", result["iterations"])
    print("\nFinal Synopsis:\n")
    print(result["final_synopsis"])

    out_path = os.environ.get("OUTPUT_FILE", "synopsis_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trial_query": trial_query,
                "trial_brief": brief.to_prompt_json(),
                "result": result,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to: {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()

