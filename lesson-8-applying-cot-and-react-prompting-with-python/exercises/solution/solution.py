#!/usr/bin/env python3
"""
Pathogenic-or-Benign Variant Analyst (CoT + ReAct) — Solution Script
Udacity Life Sciences Agentic AI Nanodegree — Course 1
Module: Applying COT and ReACT Prompting with Python

What this script demonstrates
- CoT-style prompting (model reasons privately; outputs structured verdict + citations)
- ReAct-style prompting (iterative tool use via Thought/Action/Observation loop)
- Life-sciences-safe guardrails (no PHI; no clinical decision-making claims)

Requirements
- pip install openai python-dotenv
- OPENAI_API_KEY in .env (loaded from dotenv.load_dotenv(".env"))
- ls_action_space available in workspace:
    from ls_action_space.action_space import *
"""

import os
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# ---- Action Space ----
USING_STUBS = False
try:
    from ls_action_space.action_space import (
        query_clinvar,
        query_pubmed,
        extract_pdf_content,
        extract_url_content,
    )
except Exception:
    USING_STUBS = True

    def query_clinvar(variant: str) -> Dict[str, Any]:
        return {
            "query": variant,
            "title": f"ClinVar record for {variant}",
            "clinical_significance": "Uncertain significance",
            "review_status": "criteria provided, single submitter",
            "conditions": ["Example condition"],
            "gene": "BRCA1",
            "accessions": {"RCV": ["RCV000000001"], "VCV": ["VCV000000001"]},
            "hgvs": [variant],
            "pubmed_pmids": ["12345678", "87654321"],
            "allele_frequencies": {"gnomAD": 0.00001},
        }


    def query_pubmed(query: str, max_results: int = 20, include_mesh: bool = True, include_citations: bool = False) -> List[Dict[str, Any]]:
        return [
            {
                "pmid": "12345678",
                "title": f"Functional analysis relevant to {query}",
                "abstract": "Functional assays suggest reduced activity.",
                "journal": "Example Journal",
                "year": 2023,
                "authors": ["A. Author", "B. Author"],
                "doi": None,
                "pmcid": None,
                "mesh_terms": ["DNA Repair"] if include_mesh else [],
            },
            {
                "pmid": "87654321",
                "title": f"Clinical characterization relevant to {query}",
                "abstract": "Clinical observations described; limited segregation data.",
                "journal": "Example Journal 2",
                "year": 2022,
                "authors": ["C. Author"],
                "doi": None,
                "pmcid": None,
                "mesh_terms": ["Genetic Variation"] if include_mesh else [],
            },
        ]

    def extract_pdf_content(doi_or_url: str, *, max_pages: Optional[int] = None) -> Dict[str, Any]:
        return {"text": "Stubbed PDF text content.", "meta": {"pages": 1, "bytes": 1000}}

    def extract_url_content(url: str) -> Dict[str, Any]:
        return {"title": "Stubbed page", "text": "Stubbed URL text content."}


# ---- OpenAI Client Setup ----
load_dotenv(".env")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openai.vocareum.com/v1",
)

MODEL = "gpt-4o-mini"


# ---- Utilities ----
def clip(text: str, max_chars: int = 2500) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 50] + "\n...[truncated]..."


def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_output_tokens: int = 900) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    return resp.choices[0].message.content or ""


# ---- Evidence Collection (shared by both CoT and ReAct) ----
class EvidenceCollector:
    def __init__(self, variant_hgvs: str):
        self.variant = variant_hgvs

    def collect_baseline(self, pubmed_max: int = 8) -> Dict[str, Any]:
        clinvar = query_clinvar(self.variant)

        gene = clinvar.get("gene") or ""

        # Literature query strategy: variant + gene
        q = self.variant if not gene else f'{gene} "{self.variant}"'
        papers = query_pubmed(q, max_results=pubmed_max, include_mesh=True, include_citations=False)

        # Trim abstracts to keep prompts small
        for p in papers:
            p["abstract"] = clip(p.get("abstract", ""), 800)

        return {
            "variant": self.variant,
            "clinvar": clinvar,
            "pubmed": papers,
            "notes": {
                "phi_policy": "No patient-identifiable info is provided or requested.",
                "intended_use": "Educational demo only; not medical advice.",
            },
        }


# ---- Part I: CoT Prompting (single-shot, structured output) ----
class CoTVariantClassifier:
    def __init__(self, variant_hgvs: str):
        self.variant = variant_hgvs

    def classify(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You are a life-sciences variant interpretation assistant.\n"
            "You can apply high-level ACMG/AMP-style reasoning, but you must NOT provide a full chain-of-thought.\n"
            "Reason step-by-step privately; output ONLY the requested structured result.\n"
            "Safety:\n"
            "- Do not request or infer patient identity.\n"
            "- Do not provide clinical decision-making instructions.\n"
            "- Use cautious language and acknowledge uncertainty.\n"
        )

        schema = {
            "verdict": "Pathogenic | Likely pathogenic | VUS | Likely benign | Benign",
            "confidence": "Low | Moderate | High",
            "summary": "2-5 bullet points",
            "key_evidence": [
                {
                    "type": "ClinVar | Literature",
                    "detail": "short",
                    "citation": "PMID:xxxxxxx or DOI or ClinVar:VCV/RCV",
                }
            ],
            "limitations": ["short bullet points"],
            "recommended_next_steps": ["short bullet points (research/curation steps, not clinical)"],
        }

        user = (
            "Classify this HGVS variant using the provided evidence.\n"
            "Return VALID JSON ONLY matching this schema (no markdown):\n"
            f"{pretty(schema)}\n\n"
            "Evidence (JSON):\n"
            f"{pretty(evidence)}\n"
        )

        content = chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_output_tokens=900,
        )

        parsed = safe_json_loads(content)
        if parsed is None:
            # robust fallback: extract JSON block if the model wrapped it
            m = re.search(r"\{.*\}\s*$", content.strip(), flags=re.DOTALL)
            parsed = safe_json_loads(m.group(0)) if m else None

        if parsed is None:
            return {
                "verdict": "VUS",
                "confidence": "Low",
                "summary": ["Model returned non-JSON output; falling back."],
                "key_evidence": [],
                "limitations": ["Parsing failure."],
                "recommended_next_steps": ["Re-run; ensure evidence is not too large."],
                "raw_model_output": content,
            }

        return parsed


# ---- Part II: ReAct Prompting (iterative tool use) ----
TOOL_SPECS = {
    "query_clinvar": {
        "args": {"variant": "HGVS string"},
        "desc": "Look up ClinVar summary for a variant (clinical significance, review status, gene, PMIDs, accessions).",
    },
    "query_pubmed": {
        "args": {"query": "search string", "max_results": 5},
        "desc": "Search PubMed for relevant papers. Returns titles/abstracts and identifiers.",
    },
    "extract_pdf_content": {
        "args": {"doi_or_url": "DOI or PDF URL", "max_pages": 3},
        "desc": "Pull full-text from an open-access PDF for deeper evidence (truncated).",
    },
    "extract_url_content": {
        "args": {"url": "web URL"},
        "desc": "Extract readable text from a URL (guidelines, databases, etc.).",
    },
    "final_answer": {
        "args": {"verdict_json": "final structured JSON"},
        "desc": "Finish and output the final structured verdict as JSON.",
    },
}


def call_tool(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "query_clinvar":
        return query_clinvar(tool_args["variant"])
    if tool_name == "query_pubmed":
        return query_pubmed(
            tool_args["query"],
            max_results=int(tool_args.get("max_results", 5)),
            include_mesh=True,
            include_citations=False,
        )
    if tool_name == "extract_pdf_content":
        return extract_pdf_content(tool_args["doi_or_url"], max_pages=tool_args.get("max_pages", 3))
    if tool_name == "extract_url_content":
        return extract_url_content(tool_args["url"])
    raise ValueError(f"Unknown tool: {tool_name}")


class ReActVariantAgent:
    """
    Text-based ReAct loop:
      - Model chooses ACTION + JSON args
      - Python executes tool
      - Observation returned to model
      - Repeat until FINAL
    """

    def __init__(self, variant_hgvs: str, max_iters: int = 6, verbose: bool = True):
        self.variant = variant_hgvs
        self.max_iters = max_iters
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []

    def _system_prompt(self) -> str:
        tool_lines = []
        for name, spec in TOOL_SPECS.items():
            tool_lines.append(f"- {name}(args: {json.dumps(spec['args'])}): {spec['desc']}")
        tools_text = "\n".join(tool_lines)

        return (
            "You are a ReAct-style life-sciences variant analysis agent.\n"
            "Goal: classify an HGVS variant using tools (ClinVar/PubMed/full text) and provide citations.\n\n"
            "Rules:\n"
            "- You may take multiple steps.\n"
            "- You MUST follow this response format exactly.\n"
            "- Keep any 'Thought' to ONE sentence (no long chain-of-thought).\n"
            "- Do not request PHI or patient details.\n"
            "- Do not give clinical decision instructions; provide research/curation next steps only.\n\n"
            "Response format (choose one):\n"
            "1) To use a tool:\n"
            "Thought: <one sentence>\n"
            "Action: <tool_name>\n"
            "Action Input: <VALID JSON>\n\n"
            "2) To finish:\n"
            "Thought: <one sentence>\n"
            "Action: final_answer\n"
            "Action Input: <VALID JSON with key 'verdict_json' containing the final verdict JSON>\n\n"
            "Available tools:\n"
            f"{tools_text}\n"
        )

    @staticmethod
    def _parse_action(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        # Expect:
        # Action: tool_name
        # Action Input: {json}
        m_tool = re.search(r"^Action:\s*([a-zA-Z0-9_]+)\s*$", text, flags=re.MULTILINE)
        m_args = re.search(r"^Action Input:\s*(\{.*\})\s*$", text, flags=re.MULTILINE | re.DOTALL)

        if not m_tool or not m_args:
            return None
        tool = m_tool.group(1).strip()
        args_raw = m_args.group(1).strip()
        args = safe_json_loads(args_raw)
        if not isinstance(args, dict):
            return None
        return tool, args

    def run(self) -> Dict[str, Any]:
        self.history = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Analyze this variant: {self.variant}\n"
                    "Start by checking ClinVar, then gather gene/disease context and literature evidence.\n"
                    "Finish with a structured JSON verdict with citations (PMIDs/DOIs/ClinVar accessions).\n"
                ),
            },
        ]

        for step in range(1, self.max_iters + 1):
            assistant = chat(self.history, temperature=0.2, max_output_tokens=700)
            if self.verbose:
                print("\n" + "=" * 88)
                print(f"ReAct STEP {step}")
                print("=" * 88)
                print(assistant.strip())

            parsed = self._parse_action(assistant)
            if parsed is None:
                # If the model didn't comply, nudge and continue
                self.history.append({"role": "assistant", "content": assistant})
                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last message did not match the required format.\n"
                            "Reply again using exactly:\n"
                            "Thought: ...\nAction: <tool>\nAction Input: <json>\n"
                        ),
                    }
                )
                continue

            tool, args = parsed

            # Record assistant message
            self.history.append({"role": "assistant", "content": assistant})

            if tool == "final_answer":
                verdict_json = args.get("verdict_json")
                if isinstance(verdict_json, dict):
                    return verdict_json
                # Sometimes model embeds as string
                if isinstance(verdict_json, str):
                    v = safe_json_loads(verdict_json) or {"raw_verdict_json": verdict_json}
                    return v
                return {"error": "final_answer missing verdict_json", "raw": args}

            # Execute tool
            try:
                obs = call_tool(tool, args)
            except Exception as e:
                obs = {"error": f"{type(e).__name__}: {e}"}

            # Trim observations to keep context manageable
            obs_for_prompt = obs
            if isinstance(obs, dict) and "text" in obs and isinstance(obs["text"], str):
                obs_for_prompt = dict(obs)
                obs_for_prompt["text"] = clip(obs["text"], 2500)
            if isinstance(obs, list):
                # PubMed can return a list
                trimmed = []
                for item in obs[: int(args.get("max_results", 5))]:
                    it = dict(item)
                    if "abstract" in it:
                        it["abstract"] = clip(it.get("abstract", ""), 700)
                    trimmed.append(it)
                obs_for_prompt = trimmed

            observation_msg = {"role": "user", "content": f"Observation (from {tool}):\n{pretty(obs_for_prompt)}"}
            self.history.append(observation_msg)

            # Small pause to avoid rate-limit bursts in some environments
            time.sleep(0.2)

        return {"error": "Max iterations reached without final_answer."}


# ---- Main ----
def main() -> None:
    variant = os.getenv("VARIANT", "NM_007294.3:c.5266dupC")

    print("=" * 88)
    print("PATHOGENIC-OR-BENIGN VARIANT ANALYST — CoT + ReAct")
    print("Udacity Life Sciences Agentic AI Nanodegree — Course 1")
    print("=" * 88)
    print(f"Variant: {variant}")

    if USING_STUBS:
        print("\n⚠️ NOTE: ls_action_space not found; using stub tools (demo structure only).\n")

    # Collect baseline evidence once (useful for CoT; ReAct collects as needed)
    collector = EvidenceCollector(variant)
    baseline_evidence = collector.collect_baseline(pubmed_max=8)

    print("\n" + "-" * 88)
    print("BASELINE EVIDENCE (for CoT)")
    print("-" * 88)
    print(pretty({k: baseline_evidence[k] for k in ["variant", "clinvar"]}))
    print(f"(PubMed papers fetched: {len(baseline_evidence.get('pubmed', []))})")

    # Part I — CoT single-shot
    print("\n" + "=" * 88)
    print("PART I — CoT Prompt (single-shot structured verdict)")
    print("=" * 88)
    cot = CoTVariantClassifier(variant)
    cot_result = cot.classify(baseline_evidence)
    print(pretty(cot_result))

    # Part II — ReAct multi-step
    print("\n" + "=" * 88)
    print("PART II — ReAct Agent (iterative tool use)")
    print("=" * 88)
    agent = ReActVariantAgent(variant_hgvs=variant, max_iters=6, verbose=True)
    react_result = agent.run()
    print("\n" + "=" * 88)
    print("ReAct FINAL RESULT")
    print("=" * 88)
    print(pretty(react_result))

    # Comparison
    print("\n" + "=" * 88)
    print("COMPARISON — CoT vs ReAct")
    print("=" * 88)
    print(
        pretty(
            {
                "variant": variant,
                "cot_verdict": cot_result.get("verdict"),
                "cot_confidence": cot_result.get("confidence"),
                "react_verdict": react_result.get("verdict"),
                "react_confidence": react_result.get("confidence"),
                "note": "CoT is single-shot over a fixed evidence bundle; ReAct can decide which tools to call next.",
            }
        )
    )


if __name__ == "__main__":
    main()

