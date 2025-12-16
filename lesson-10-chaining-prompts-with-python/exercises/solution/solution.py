#!/usr/bin/env python3
"""
Pharmacovigilance Signal Reporter (TOOL-USING AGENTS + GATES)
LangChain Tools + Programmatic Gate Checks (Course-bridge solution)

Flow:
  A) Statistical Agent (TOOLS REQUIRED)
     - must call get_faers_counts(drug,event)  -> {a,b,c,d}
     - must call calculate_drug_event_statistics(counts) -> stats JSON
     - Gate 1 validates stats sanity (Python)

  B) Draft Writer (NO TOOLS)
     - consumes ONLY the returned stats JSON
     - produces STRICT JSON brief (Gate 2: schema + numeric allowlist)

  C) Final Writer (NO TOOLS)
     - converts JSON brief -> regulator-ready Markdown (Gate 3)

Notes:
- Uses only aggregate openFDA counts (no patient-level data).
- Numeric allowlist gates prevent invented numbers from propagating.
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
# openFDA helpers (live)
# =============================================================================
def _openfda_total_query() -> str:
    # openFDA requires a non-empty search; use a wide date range.
    return "receivedate:[20040101 TO 30000101]"


def _query_openfda_total(search_query: str, *, retries: int = 3, timeout_s: int = 20) -> int:
    """Return openFDA meta.results.total for a given search string. Returns 0 on failure."""
    params = {"search": search_query, "limit": 1}
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        try:
            r = requests.get(OPENFDA_ENDPOINT, params=params, timeout=timeout_s)
            if r.status_code in (429, 503):  # rate limit / temporary unavailable
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
# Tools
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
# Gates (programmatic checks)
# =============================================================================
def gate1_validate_statistics(stats: Dict[str, Any]) -> Tuple[bool, str]:
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
    allowed = set()

    for key in ("ror", "prr", "ci_lower", "ci_upper"):
        v = stats[key]
        allowed.add(str(v))
        allowed.add(f"{v:.2f}")
        if isinstance(v, float) and v.is_integer():
            allowed.add(str(int(v)))

    allowed.add(str(int(stats["cases"])))
    allowed.add("95")  # allow "95% CI"

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


def gate2_validate_draft_json(draft_obj: Dict[str, Any], stats: Dict[str, Any]) -> Tuple[bool, str]:
    required_keys = {
        "drug",
        "adverse_event",
        "statistics_table",
        "risk_statement",
        "caveats",
        "data_source",
        "compliance_notes",
    }
    if not required_keys.issubset(set(draft_obj.keys())):
        missing = sorted(list(required_keys - set(draft_obj.keys())))
        return False, f"Draft missing keys: {missing}"

    # Ensure required values appear somewhere in the statistics table text
    table = draft_obj.get("statistics_table", [])
    table_text = json.dumps(table, ensure_ascii=False)
    must_have = [str(stats["ror"]), str(stats["prr"]), str(stats["ci_lower"]), str(stats["ci_upper"]), str(stats["cases"])]
    missing_vals = [m for m in must_have if m not in table_text]
    if missing_vals:
        return False, f"Draft statistics_table missing required values: {', '.join(missing_vals)}"

    combined = json.dumps(draft_obj, ensure_ascii=False)
    ok, msg = gate_no_hallucinated_numbers(combined, stats)
    if not ok:
        return False, f"Draft contains disallowed numbers. {msg}"

    return True, "OK"


def gate3_validate_final_markdown(md: str, stats: Dict[str, Any]) -> Tuple[bool, str]:
    required_headings = ["Signal Summary", "Statistics", "Interpretation", "Caveats", "Data Source"]
    missing = [h for h in required_headings if h.lower() not in md.lower()]
    if missing:
        return False, f"Final brief missing required section(s): {', '.join(missing)}"

    must_have_values = [str(stats["ror"]), str(stats["prr"]), str(stats["ci_lower"]), str(stats["ci_upper"]), str(stats["cases"])]
    missing_vals = [v for v in must_have_values if v not in md]
    if missing_vals:
        return False, f"Final brief missing required stat values: {', '.join(missing_vals)}"

    ok, msg = gate_no_hallucinated_numbers(md, stats)
    if not ok:
        return False, f"Final brief contains disallowed numbers. {msg}"

    return True, "OK"


# =============================================================================
# LLM setup + tool-call runner
# =============================================================================
def _llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    # Accept either BASE_URL or OPENAI_BASE_URL (common in course environments)
    base_url = (os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip() or None
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
      - invoke agent -> if tool_calls, execute them -> provide ToolMessages -> invoke again
    Returns:
      final AI message, plus list of (tool_name, tool_output) in the order executed.
    """
    result = agent.invoke(messages)
    tool_outputs: List[Tuple[str, Any]] = []

    # If tool calls exist, execute and re-invoke
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
# Step A: Statistical agent (TOOLS REQUIRED)
# =============================================================================
def run_statistical_agent(drug: str, event: str) -> Dict[str, Any]:
    """
    Statistical agent must:
      1) call get_faers_counts(drug,event)
      2) call calculate_drug_event_statistics(counts)
    We trust the tool output as the stats source-of-truth.
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
            "- After tool calls, briefly confirm completion (no new numbers).\n"
        )
    )

    final_msg, tool_outputs = _run_tool_agent_once(agent, tools, [system, user])

    # Enforce that required tools were called
    called = [name for name, _ in tool_outputs]
    if "get_faers_counts" not in called:
        raise RuntimeError("Statistical agent did not call get_faers_counts.")
    if "calculate_drug_event_statistics" not in called:
        raise RuntimeError("Statistical agent did not call calculate_drug_event_statistics.")

    # Take the stats directly from the calculate tool output (source of truth)
    stats = next(out for name, out in tool_outputs if name == "calculate_drug_event_statistics")
    if not isinstance(stats, dict):
        raise RuntimeError("Statistics tool returned non-dict output.")

    return stats


# =============================================================================
# Step B: Draft-writing agent (NO TOOLS) -> strict JSON brief
# =============================================================================
def step2_generate_draft_json(drug: str, event: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    llm = _llm()

    system = (
        "You are a pharmacovigilance signal reporting assistant. "
        "Output MUST be valid JSON only (no markdown, no extra text). "
        "NEVER invent numbers; use ONLY the provided numbers verbatim."
    )

    user = f"""
Create a draft pharmacovigilance signal brief as JSON for regulatory review.

Drug: {drug}
Adverse Event: {event}

Provided statistics (the ONLY numbers you may use anywhere):
- ROR: {stats["ror"]}
- PRR: {stats["prr"]}
- 95% CI (ROR): [{stats["ci_lower"]}, {stats["ci_upper"]}]
- Cases (a): {stats["cases"]}
- 2x2 counts: a={stats["counts"]["a"]}, b={stats["counts"]["b"]}, c={stats["counts"]["c"]}, d={stats["counts"]["d"]}

JSON schema requirements:
- drug (string)
- adverse_event (string)
- statistics_table (array of objects: {{metric: string, value: string}})
  Include at least: Drug, Adverse Event, ROR, PRR, 95% CI, Cases (and optionally 2x2 counts).
- risk_statement (string): plain-English interpretation in FAERS context (association, not causation).
- caveats (array of strings): include confounding and under-reporting.
- data_source (string): mention FAERS/openFDA.
- compliance_notes (array of strings): include "no patient-level data" and "numbers copied from provided stats".

Hard rules:
- Use ONLY the numbers provided above. Do NOT include any other numbers anywhere (including ranges or percentages besides "95").
- Do NOT claim incidence, prevalence, or causality.
- Keep language professional and concise.

Return ONLY valid JSON.
""".strip()

    # Ask for JSON; parse robustly
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = (msg.content or "").strip()
    return json.loads(text)


# =============================================================================
# Step C: Final-writing agent (NO TOOLS) -> Markdown
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
# Main
# =============================================================================
def main() -> int:
    print("=" * 72)
    print("Pharmacovigilance Signal Reporter")
    print("Tool-Using Agents + Gate Checks (LangChain)")
    print("=" * 72)

    drug = os.getenv("DRUG_NAME", "warfarin").strip()
    event = os.getenv("ADVERSE_EVENT", "bleeding").strip()
    print(f"\n🎯 Drug–Event Pair: {drug} + {event}")

    # A) Statistical agent -> tool-derived stats
    print("\n" + "-" * 72)
    print("STEP A: Statistical Agent (MUST use tools)")
    print("-" * 72)
    stats = run_statistical_agent(drug, event)

    print("Stats (tool output):")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    ok, msg = gate1_validate_statistics(stats)
    print(f"\n🔍 GATE 1 (stats sanity): {msg}")
    if not ok:
        print("❌ Stopping: invalid stats.")
        return 1

    # B) Draft JSON brief (no tools) + Gate 2
    print("\n" + "-" * 72)
    print("STEP B: Draft Writer (STRICT JSON, no tools)")
    print("-" * 72)
    try:
        draft = step2_generate_draft_json(drug, event, stats)
    except Exception as e:
        print(f"❌ Step B failed (JSON draft): {e}")
        return 1

    ok, msg = gate2_validate_draft_json(draft, stats)
    print(f"\n🔍 GATE 2 (draft JSON): {msg}")
    if not ok:
        print("❌ Stopping: draft failed validation.")
        print(json.dumps(draft, indent=2, ensure_ascii=False))
        return 1

    # C) Final Markdown (no tools) + Gate 3
    print("\n" + "-" * 72)
    print("STEP C: Final Writer (Markdown, no tools)")
    print("-" * 72)
    try:
        final_md = step3_finalize_markdown_from_json(draft, stats)
    except Exception as e:
        print(f"❌ Step C failed (Markdown): {e}")
        return 1

    ok, msg = gate3_validate_final_markdown(final_md, stats)
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

