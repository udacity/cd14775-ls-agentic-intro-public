#!/usr/bin/env python3
"""
Pathogenic-or-Benign Variant Analyst (CoT + ReAct) — STUDENT SCAFFOLD (LOW EFFORT)
Udacity Life Sciences Agentic AI Nanodegree — Course 1
Module: Applying CoT and ReACT Prompting with Python

Student work (VERY small, < 15 minutes)
- TODO #1: Fill in the CoT SYSTEM prompt (guardrails + “JSON only”)
- TODO #2: Fill in the CoT USER prompt (schema + evidence + “JSON only”)
- TODO #3: Fill in the ReAct SYSTEM prompt (format + tool list + guardrails)

Everything else is provided and runnable (with real tools if available, otherwise stubs).

Safety (must keep)
- No PHI / patient identity inference.
- No clinical decision-making instructions. Provide research/curation next steps only.
"""

import os
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


# =============================================================================
# Tools (action space)
# =============================================================================

USING_STUBS = False
try:
    from ls_action_space.action_space import query_clinvar, query_pubmed
except Exception:
    USING_STUBS = True

    def query_clinvar(variant: str) -> Dict[str, Any]:
        return {
            "query": variant,
            "gene": "BRCA1",
            "clinical_significance": "Uncertain significance",
            "review_status": "criteria provided, single submitter",
            "conditions": ["Example condition"],
            "accessions": {"VCV": ["VCV000000001"], "RCV": ["RCV000000001"]},
            "pubmed_pmids": ["12345678", "87654321"],
            "allele_frequencies": {"gnomAD": 0.00001},
        }

    def query_pubmed(query: str, max_results: int = 3, **kwargs) -> List[Dict[str, Any]]:
        return [
            {
                "pmid": "12345678",
                "title": f"Functional analysis relevant to {query}",
                "abstract": "Functional assays suggest reduced activity.",
                "year": 2023,
            },
            {
                "pmid": "87654321",
                "title": f"Clinical characterization relevant to {query}",
                "abstract": "Limited segregation/phenotype evidence described.",
                "year": 2022,
            },
        ][:max_results]


# =============================================================================
# OpenAI client
# =============================================================================

load_dotenv(".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openai.vocareum.com/v1",
)
MODEL = "gpt-4o-mini"


# =============================================================================
# Helpers
# =============================================================================

def pretty(x: Any) -> str:
    return json.dumps(x, indent=2, ensure_ascii=False)


def clip(text: str, n: int = 700) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 20] + " …[truncated]…"


def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}\s*$", text.strip(), flags=re.DOTALL)
    return safe_json_loads(m.group(0)) if m else None


def chat(messages: List[Dict[str, str]], max_output_tokens: int = 600) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=max_output_tokens,
    )
    return resp.choices[0].message.content or ""


# =============================================================================
# Shared schema (used by BOTH CoT and ReAct final output)
# =============================================================================

VERDICT_SCHEMA: Dict[str, Any] = {
    "verdict": "Pathogenic | Likely pathogenic | VUS | Likely benign | Benign",
    "confidence": "Low | Moderate | High",
    "summary": ["1-3 short bullets"],
    "citations": ["PMID:xxxxxxx and/or ClinVar:VCV/RCV"],
    "recommended_next_steps": ["1-3 research/curation steps (NOT clinical advice)"],
}


# =============================================================================
# Evidence collection (already complete)
# =============================================================================

def collect_evidence(variant_hgvs: str) -> Dict[str, Any]:
    clinvar = query_clinvar(variant_hgvs)
    gene = clinvar.get("gene") or ""
    q = f'{gene} "{variant_hgvs}"' if gene else variant_hgvs

    papers = query_pubmed(q, max_results=3)
    for p in papers:
        p["abstract"] = clip(p.get("abstract", ""), 700)

    return {
        "variant": variant_hgvs,
        "clinvar": clinvar,
        "pubmed": papers,
        "notes": {
            "phi_policy": "No patient-identifiable info is provided or requested.",
            "intended_use": "Educational demo only; not medical advice.",
        },
    }


# =============================================================================
# Part I — CoT Prompting (single call)
# =============================================================================

def cot_system_prompt() -> str:
    # TODO #1 (CoT): Write a short SYSTEM prompt that enforces:
    # - Reasoning allowed privately, but do NOT reveal chain-of-thought
    # - Output MUST be valid JSON only
    # - Safety: no PHI; no clinical decision instructions; cautious language
    return (
        "TODO: Replace this with your CoT system prompt.\n"
        "Must require: JSON only, no chain-of-thought, no PHI, no clinical advice.\n"
    )


def cot_user_prompt(evidence: Dict[str, Any]) -> str:
    # TODO #2 (CoT): Write the USER prompt that includes:
    # - The schema (VERDICT_SCHEMA)
    # - The evidence JSON
    # - Clear instruction: 'Return VALID JSON ONLY (no markdown)'
    return (
        "TODO: Replace this with your CoT user prompt.\n"
        "Include schema + evidence. Require: JSON only.\n"
    )


def cot_classify(evidence: Dict[str, Any]) -> Dict[str, Any]:
    raw = chat(
        [
            {"role": "system", "content": cot_system_prompt()},
            {"role": "user", "content": cot_user_prompt(evidence)},
        ],
        max_output_tokens=650,
    )

    parsed = safe_json_loads(raw) or extract_last_json_object(raw)
    if parsed is None:
        return {
            "verdict": "VUS",
            "confidence": "Low",
            "summary": ["Could not parse model output as JSON."],
            "citations": [],
            "recommended_next_steps": ["Re-run with stricter 'JSON only' instruction."],
            "raw_model_output": raw,
        }

    return parsed


# =============================================================================
# Part II — ReAct Prompting (iterative tool use)
# =============================================================================

TOOL_SPECS = {
    "query_clinvar": {
        "args": {"variant": "HGVS string"},
        "desc": "Fetch ClinVar summary (gene, clinical significance, review status, accessions, PMIDs).",
    },
    "query_pubmed": {
        "args": {"query": "string", "max_results": 3},
        "desc": "Fetch top PubMed papers (title/abstract/PMID).",
    },
    "final_answer": {
        "args": {"verdict_json": "JSON object matching VERDICT_SCHEMA"},
        "desc": "Finish by returning the final verdict JSON.",
    },
}


def react_system_prompt() -> str:
    # TODO #3 (ReAct): Write a short SYSTEM prompt that enforces:
    # - Strict format:
    #   Thought: <one sentence>
    #   Action: <query_clinvar|query_pubmed|final_answer>
    #   Action Input: <valid JSON>
    # - List the available tools (names + args)
    # - Safety: no PHI; no clinical advice
    # - The final_answer must contain verdict_json that matches VERDICT_SCHEMA
    tool_list = "\n".join(
        f"- {name}(args: {json.dumps(spec['args'])}): {spec['desc']}"
        for name, spec in TOOL_SPECS.items()
    )

    return (
        "TODO: Replace this with your ReAct system prompt.\n"
        "Must include strict format + tools + safety + final JSON requirement.\n\n"
        f"Available tools:\n{tool_list}\n"
    )


def parse_react_action(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    m_tool = re.search(r"^Action:\s*([a-zA-Z0-9_]+)\s*$", text, flags=re.MULTILINE)
    m_args = re.search(r"^Action Input:\s*(\{.*\})\s*$", text, flags=re.MULTILINE | re.DOTALL)
    if not m_tool or not m_args:
        return None
    tool = m_tool.group(1).strip()
    args = safe_json_loads(m_args.group(1).strip())
    if not isinstance(args, dict):
        return None
    return tool, args


def call_tool(tool: str, args: Dict[str, Any]) -> Any:
    if tool == "query_clinvar":
        return query_clinvar(args["variant"])
    if tool == "query_pubmed":
        return query_pubmed(args["query"], max_results=int(args.get("max_results", 3)))
    raise ValueError(f"Unknown tool: {tool}")


def run_react(variant_hgvs: str, max_turns: int = 4, verbose: bool = True) -> Dict[str, Any]:
    history: List[Dict[str, str]] = [
        {"role": "system", "content": react_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Analyze variant {variant_hgvs}. "
                "Start by calling query_clinvar. Then call query_pubmed. "
                "Then finish with final_answer."
            ),
        },
    ]

    for step in range(1, max_turns + 1):
        assistant = chat(history, max_output_tokens=500)
        history.append({"role": "assistant", "content": assistant})

        if verbose:
            print("\n" + "=" * 80)
            print(f"ReAct step {step}")
            print("=" * 80)
            print(assistant.strip())

        parsed = parse_react_action(assistant)
        if not parsed:
            history.append(
                {
                    "role": "user",
                    "content": (
                        "Format error. Reply using EXACTLY:\n"
                        "Thought: <one sentence>\n"
                        "Action: <query_clinvar|query_pubmed|final_answer>\n"
                        "Action Input: <valid JSON>\n"
                    ),
                }
            )
            continue

        tool, args = parsed

        if tool == "final_answer":
            verdict_json = args.get("verdict_json")
            if isinstance(verdict_json, dict):
                return verdict_json
            # If model returned it as string, try parse:
            if isinstance(verdict_json, str):
                return safe_json_loads(verdict_json) or {"raw_verdict_json": verdict_json}
            return {"error": "final_answer missing verdict_json", "raw": args}

        try:
            obs = call_tool(tool, args)
        except Exception as e:
            obs = {"error": f"{type(e).__name__}: {e}"}

        # Keep observations small
        if isinstance(obs, list):
            trimmed = []
            for it in obs[: int(args.get("max_results", 3))]:
                d = dict(it)
                if "abstract" in d:
                    d["abstract"] = clip(d.get("abstract", ""), 700)
                trimmed.append(d)
            obs = trimmed

        history.append({"role": "user", "content": f"Observation:\n{pretty(obs)}"})
        time.sleep(0.1)

    return {"error": "Reached max_turns without final_answer."}


# =============================================================================
# Main (already complete)
# =============================================================================

def main() -> None:
    variant = os.getenv("VARIANT", "NM_007294.3:c.5266dupC")

    print("=" * 80)
    print("Variant Analyst — SIMPLE SCAFFOLD (CoT + ReAct)")
    print("=" * 80)
    print(f"Variant: {variant}")
    if USING_STUBS:
        print("⚠️ Using stub tools (ls_action_space not found).")

    evidence = collect_evidence(variant)

    print("\n--- Evidence snapshot ---")
    print(pretty({"variant": evidence["variant"], "clinvar": evidence["clinvar"]}))
    print(f"PubMed papers: {len(evidence.get('pubmed', []))}")

    print("\n" + "-" * 80)
    print("CoT (single-shot)")
    print("-" * 80)
    cot_result = cot_classify(evidence)
    print(pretty(cot_result))

    print("\n" + "-" * 80)
    print("ReAct (tool-using loop)")
    print("-" * 80)
    react_result = run_react(variant, max_turns=4, verbose=True)
    print("\nFINAL ReAct result:")
    print(pretty(react_result))

    print("\n" + "-" * 80)
    print("Comparison")
    print("-" * 80)
    print(
        pretty(
            {
                "variant": variant,
                "cot_verdict": cot_result.get("verdict"),
                "cot_confidence": cot_result.get("confidence"),
                "react_verdict": react_result.get("verdict"),
                "react_confidence": react_result.get("confidence"),
            }
        )
    )


if __name__ == "__main__":
    main()

