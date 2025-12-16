#!/usr/bin/env python3
"""
Pharmacovigilance Signal Reporter (STARTER SCAFFOLD)
LangChain Tools + Programmatic Gate Checks

Learning objective:
- Implement a multi-step chain with programmatic gate checks between steps.
- Learn how to define/attach tools and force an agent to call them.
- Learn how to validate downstream LLM outputs so tool results aren't distorted.

Flow:
  A) Statistical Agent (TOOLS REQUIRED)
     - must call get_faers_counts(drug,event)  -> {a,b,c,d}   [TOOL PROVIDED]
     - must call calculate_drug_event_statistics(counts) -> stats JSON [TOOL PROVIDED]
     - Gate 1 validates stats sanity (provided)

  B) Draft Writer (NO TOOLS)
     - consumes ONLY the returned stats JSON
     - produces STRICT JSON brief (Gate 2: TODO)

  C) Final Writer (NO TOOLS)
     - converts JSON brief -> regulator-ready Markdown (Gate 3: TODO)

Your tasks (all quick):
  1) Write the prompts for Step B (draft JSON)  [~5–10 mins]
  2) Implement Gate 2 validator (required keys + numeric allowlist) [~10 mins]
  3) Implement Gate 3 validator (required headings + numeric allowlist) [~10 mins]

Run:
  pip install requests python-dotenv langchain-openai langchain-core
  export OPENAI_API_KEY="..."
  export OPENAI_BASE_URL="https://openai.vocareum.com/v1"   # if needed
  export DRUG_NAME="warfarin"
  export ADVERSE_EVENT="bleeding"
  python pv_signal_reporter_starter.py
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv(".env")

OPENFDA_ENDPOINT = "https://api.fda.gov/drug/event.json"


# =============================================================================
# openFDA helpers (PROVIDED)
# =============================================================================
def _openfda_total_query() -> str:
    return "receivedate:[20040101 TO 30000101]"


def _query_openfda_total(search_query: str, *, retries: int = 3, timeout_s: int = 20) -> int:
    """Return openFDA meta.results.total for a given search string. Returns 0 on failure."""
    params = {"search": search_query, "limit": 1}
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        try:
            r = requests.get(OPENFDA_ENDPOINT, params=params, timeout=timeout_s)
            if r.status_code in (429, 503):
                time.sleep(1.0 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            return int(data.get("meta", {}).get("results", {}).get("total", 0))
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    print(f"⚠️ openFDA request failed for query={search_query!r}: {last_err}")
    return 0


# =============================================================================
# Tools (PROVIDED)
# =============================================================================
@tool
def get_faers_counts(drug_name: str, adverse_event: str) -> Dict[str, int]:
    """
    Fetch aggregate FAERS 2x2 contingency table counts from openFDA.

    Returns:
      a: drug & event
      b: drug & !event
      c: !drug & event
      d: !drug & !event
    """
    drug_q = f'patient.drug.medicinalproduct:"{drug_name}"'
    event_q = f'patient.reaction.reactionmeddrapt:"{adverse_event}"'

    a = _query_openfda_total(f"({drug_q}) AND ({event_q})")
    total_drug = _query_openfda_total(drug_q)
    b = max(total_drug - a, 0)

    total_event = _query_openfda_total(event_q)
    c = max(total_event - a, 0)

    total_reports = _query_openfda_total(_openfda_total_query())
    d = max(total_reports - a - b - c, 0)

    return {"a": int(a), "b": int(b), "c": int(c), "d": int(d)}


@tool
def calculate_drug_event_statistics(counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Compute ROR, PRR, and 95% CI (ROR) from a 2x2 table.
    Uses Haldane-Anscombe correction (+0.5) to avoid division by zero.
    """
    a = counts["a"] + 0.5
    b = counts["b"] + 0.5
    c = counts["c"] + 0.5
    d = counts["d"] + 0.5

    ror = (a * d) / (b * c)
    prr = (a / (a + b)) / (c / (c + d))

    log_ror = math.log(ror)
    se_log_ror = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_lower = math.exp(log_ror - 1.96 * se_log_ror)
    ci_upper = math.exp(log_ror + 1.96 * se_log_ror)

    return {
        "ror": float(round(ror, 2)),
        "prr": float(round(prr, 2)),
        "ci_lower": float(round(ci_lower, 2)),
        "ci_upper": float(round(ci_upper, 2)),
        "cases": int(counts["a"]),
        "counts": {"a": int(counts["a"]), "b": int(counts["b"]), "c": int(counts["c"]), "d": int(counts["d"])},
        "data_source": "FAERS via openFDA (aggregate counts only)",
    }


# =============================================================================
# Gates + numeric allowlist helpers (MOSTLY PROVIDED)
# =============================================================================
def gate1_validate_statistics(stats: Dict[str, Any]) -> Tuple[bool, str]:
    """Gate 1 (provided): sanity checks on computed stats."""
    if stats.get("ror", 0) <= 0 or stats.get("prr", 0) <= 0:
        return False, "ROR/PRR must be positive."
    if stats.get("ci_lower", 0) <= 0 or stats.get("ci_upper", 0) <= 0:
        return False, "CI bounds must be positive."
    if stats["ci_lower"] > stats["ci_upper"]:
        return False, "CI lower bound exceeds upper bound."
    if int(stats.get("cases", 0)) < 3:
        return False, "Insufficient case count for a basic signal screen (need ≥3)."
    cts = stats.get("counts", {})
    for k in ("a", "b", "c", "d"):
        if k not in cts or int(cts[k]) < 0:
            return False, f"Invalid 2x2 counts: missing/negative {k}."
    return True, "OK"


def _extract_numeric_tokens(text: str) -> List[str]:
    return re.findall(r"(?<![\w/])\d+(?:\.\d+)?(?![\w/])", text)


def _allowed_numeric_tokens(stats: Dict[str, Any]) -> List[str]:
    """Allow ONLY these number tokens to appear in LLM outputs (provided)."""
    allowed = set()
    for key in ("ror", "prr", "ci_lower", "ci_upper"):
        v = stats[key]
        allowed.add(str(v))
        allowed.add(f"{v:.2f}")
        if isinstance(v, float) and v.is_integer():
            allowed.add(str(int(v)))
    allowed.add(str(int(stats["cases"])))
    allowed.add("95")
    for k in ("a", "b", "c", "d"):
        allowed.add(str(int(stats["counts"][k])))
    return sorted(allowed)


def gate_no_hallucinated_numbers(text: str, stats: Dict[str, Any]) -> Tuple[bool, str]:
    allowed = set(_allowed_numeric_tokens(stats))
    found = _extract_numeric_tokens(text)
    extras = [tok for tok in found if tok not in allowed]
    if extras:
        return False, f"Found numeric tokens not in allowlist: {', '.join(extras[:10])}"
    return True, "OK"


# =============================================================================
# LLM setup + tool-call runner (PROVIDED)
# =============================================================================
def _llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "").strip() or None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    return ChatOpenAI(
        model=model,
        temperature=0.2,
        max_completion_tokens=1200,
        api_key=api_key,
        base_url=base_url,
    )


def _run_tool_agent_once(agent: Any, tools: List[Any], messages: List[Any]) -> Tuple[Any, List[Tuple[str, Any]]]:
    """
    Execute a single LangChain tool-using turn:
      invoke -> execute tool calls -> ToolMessages -> invoke again.
    """
    result = agent.invoke(messages)
    tool_outputs: List[Tuple[str, Any]] = []

    if getattr(result, "tool_calls", None):
        messages = messages + [result]
        for tc in result.tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            tool_to_call = next(t for t in tools if t.name == name)
            out = tool_to_call.invoke(args)
            tool_outputs.append((name, out))
            messages.append(ToolMessage(content=json.dumps(out, ensure_ascii=False), tool_call_id=tc["id"]))
        final = agent.invoke(messages)
        return final, tool_outputs

    return result, tool_outputs


# =============================================================================
# Step A: Statistical agent (PROVIDED)
# =============================================================================
def run_statistical_agent(drug: str, event: str) -> Dict[str, Any]:
    """
    Statistical agent must call both tools and we take the stats directly from the
    calculate_drug_event_statistics tool output (source of truth).
    """
    llm = _llm()
    tools = [get_faers_counts, calculate_drug_event_statistics]
    agent = llm.bind_tools(tools)

    system = SystemMessage(
        content=(
            "You are a pharmacovigilance biostatistics assistant. "
            "You MUST use tools for all computations. "
            "First call get_faers_counts, then call calculate_drug_event_statistics using the returned counts. "
            "Do not estimate or invent values."
        )
    )
    user = HumanMessage(
        content=(
            f"Compute disproportionality statistics for:\n"
            f"Drug: {drug}\n"
            f"Adverse Event: {event}\n\n"
            "Required:\n"
            "- Call get_faers_counts(drug_name, adverse_event)\n"
            "- Then call calculate_drug_event_statistics(counts)\n"
        )
    )

    _, tool_outputs = _run_tool_agent_once(agent, tools, [system, user])

    called = [name for name, _ in tool_outputs]
    if "get_faers_counts" not in called:
        raise RuntimeError("Statistical agent did not call get_faers_counts.")
    if "calculate_drug_event_statistics" not in called:
        raise RuntimeError("Statistical agent did not call calculate_drug_event_statistics.")

    stats = next(out for name, out in tool_outputs if name == "calculate_drug_event_statistics")
    if not isinstance(stats, dict):
        raise RuntimeError("Statistics tool returned non-dict output.")
    return stats


# =============================================================================
# Step B: Draft writer (NO TOOLS) -> STRICT JSON  [TODO: prompts]
# =============================================================================
def step2_generate_draft_json(drug: str, event: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    TODO (students):
      - Write system + user prompts so the model returns JSON ONLY.
      - Hard rule: it may use ONLY the numeric values provided in `stats`.
      - Required JSON keys:
          drug, adverse_event, statistics_table, risk_statement,
          caveats, data_source, compliance_notes
    """
    llm = _llm()

    # TODO: Replace with a short, strict system prompt
    system = "TODO"

    # TODO: Replace with a user prompt that:
    #  - includes the provided stats (ror, prr, CI, cases, optionally 2x2 counts)
    #  - lists required JSON keys + what they mean
    #  - forbids adding new numbers and forbids causality/incidence claims
    user = "TODO"

    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return json.loads((msg.content or "").strip())


# =============================================================================
# Gate 2: Validate draft JSON  [TODO]
# =============================================================================
def gate2_validate_draft_json(draft_obj: Dict[str, Any], stats: Dict[str, Any]) -> Tuple[bool, str]:
    """
    TODO (students): implement Gate 2 in ~10–15 lines.

    Requirements:
      1) Draft JSON must include ALL required keys:
           drug, adverse_event, statistics_table, risk_statement,
           caveats, data_source, compliance_notes
      2) Draft must include the required values in its statistics_table:
           ror, prr, ci_lower, ci_upper, cases  (string containment is fine)
      3) Numeric integrity:
           Run gate_no_hallucinated_numbers(json.dumps(draft_obj), stats)
           and fail if any extra numbers are found.
    """
    raise NotImplementedError("TODO: implement gate2_validate_draft_json()")


# =============================================================================
# Step C: Final writer (NO TOOLS) -> Markdown  (PROVIDED)
# =============================================================================
def step3_finalize_markdown_from_json(draft: Dict[str, Any], stats: Dict[str, Any]) -> str:
    llm = _llm()

    system = (
        "You are a regulatory medical writing assistant for pharmacovigilance. "
        "You must be numerically faithful: do not add numbers beyond what is provided. "
        "Write concise Markdown with the specified sections."
    )

    allowed_numbers = _allowed_numeric_tokens(stats)

    user = f"""
Convert the following JSON draft into a final regulator-ready Markdown brief.

Required sections (use these exact headings):
- Signal Summary
- Statistics
- Interpretation
- Caveats
- Data Source

Rules:
- Include a Markdown table in the Statistics section derived from statistics_table.
- Use ONLY the numeric tokens in this allowlist: {allowed_numbers}
- Do NOT add any new numbers (no extra percentages, no ranges, no counts beyond provided).
- Keep it professional and concise.

JSON draft:
{json.dumps(draft, ensure_ascii=False)}
""".strip()

    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return (msg.content or "").strip()


# =============================================================================
# Gate 3: Validate final Markdown  [TODO]
# =============================================================================
def gate3_validate_final_markdown(md: str, stats: Dict[str, Any]) -> Tuple[bool, str]:
    """
    TODO (students): implement Gate 3 in ~10–15 lines.

    Requirements:
      1) Must include required headings (case-insensitive):
           Signal Summary, Statistics, Interpretation, Caveats, Data Source
      2) Must include required stat values somewhere in the text:
           ror, prr, ci_lower, ci_upper, cases
      3) Numeric integrity:
           Run gate_no_hallucinated_numbers(md, stats)
    """
    raise NotImplementedError("TODO: implement gate3_validate_final_markdown()")


# =============================================================================
# Main (PROVIDED)
# =============================================================================
def main() -> int:
    print("=" * 72)
    print("Pharmacovigilance Signal Reporter (STARTER)")
    print("LangChain Tools + Gate Checks")
    print("=" * 72)

    drug = os.getenv("DRUG_NAME", "warfarin").strip()
    event = os.getenv("ADVERSE_EVENT", "bleeding").strip()
    print(f"\n🎯 Drug–Event Pair: {drug} + {event}")

    # Step A: tools -> stats
    print("\n" + "-" * 72)
    print("STEP A: Statistical Agent (MUST use tools)")
    print("-" * 72)
    stats = run_statistical_agent(drug, event)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    ok, msg = gate1_validate_statistics(stats)
    print(f"\n🔍 GATE 1 (stats sanity): {msg}")
    if not ok:
        print("❌ Stopping: invalid stats.")
        return 1

    # Step B: LLM JSON draft + Gate 2
    print("\n" + "-" * 72)
    print("STEP B: Draft Writer (STRICT JSON, no tools)")
    print("-" * 72)
    try:
        draft = step2_generate_draft_json(drug, event, stats)
    except Exception as e:
        print(f"❌ Step B failed: {e}")
        return 1

    try:
        ok, msg = gate2_validate_draft_json(draft, stats)
    except NotImplementedError as e:
        print(f"🛠️ {e}")
        return 1

    print(f"\n🔍 GATE 2 (draft JSON): {msg}")
    if not ok:
        print("❌ Stopping: draft failed validation.")
        print(json.dumps(draft, indent=2, ensure_ascii=False))
        return 1

    # Step C: LLM Markdown + Gate 3
    print("\n" + "-" * 72)
    print("STEP C: Final Writer (Markdown, no tools)")
    print("-" * 72)
    final_md = step3_finalize_markdown_from_json(draft, stats)

    try:
        ok, msg = gate3_validate_final_markdown(final_md, stats)
    except NotImplementedError as e:
        print(f"🛠️ {e}")
        return 1

    print(f"\n🔍 GATE 3 (final Markdown): {msg}")
    if not ok:
        print("❌ Stopping: final brief failed validation.")
        print(final_md)
        return 1

    print("\n" + "=" * 72)
    print("FINAL REGULATOR-READY SIGNAL BRIEF")
    print("=" * 72)
    print(final_md)
    print("\n✅ Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

