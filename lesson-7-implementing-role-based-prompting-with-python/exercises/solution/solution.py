#!/usr/bin/env python3
"""
Genetic Counselor Persona Demo (Role-Based Prompting with Python)

Scenario:
A patient with a BRCA1 result wants to understand cancer risk, family screening,
and next steps.

Goal:
Demonstrate how assigning an LLM a specific professional persona (genetic counselor)
changes tone, safety behavior, and the structure of guidance—while staying empathetic,
evidence-minded, accessible (≈10th-grade reading level), and not giving medical advice.

Model:
- gpt-4o-mini
- base_url="https://openai.vocareum.com/v1"
- OPENAI_API_KEY loaded from dotenv.load_dotenv(".env")

Optional helper:
- from ls_action_space.action_space import query_pubmed  (best-effort, used only if available)
"""

import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Best-effort optional helper import
try:
    from ls_action_space.action_space import query_pubmed  # type: ignore
    ACTION_SPACE_AVAILABLE = True
except Exception:
    ACTION_SPACE_AVAILABLE = False

# OpenAI client import
try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


BASE_URL = "https://openai.vocareum.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def get_control_system_prompt() -> str:
    """A plain 'helpful assistant' baseline for comparison."""
    return "You are a helpful assistant."


def get_genetic_counselor_system_prompt() -> str:
    """
    Defines the genetic counselor persona.
    Key constraints:
      - empathetic, inclusive, non-judgmental
      - evidence-minded, align with current NCCN-style practices (without overclaiming)
      - 10th-grade reading level, avoid jargon, use analogies
      - avoid diagnosis/prescriptive medical advice
      - encourage clinician/genetic counselor follow-up
    """
    return """You are a certified genetic counselor who supports patients receiving hereditary cancer genetic test results.

Your job is to explain BRCA1-related results in a calm, empathetic, inclusive way at about a 10th-grade reading level.
Use clear headings, short paragraphs, and a few simple analogies (no scary language).

Safety & scope rules (must follow):
- You are not a doctor and cannot provide a diagnosis or personalized medical advice.
- Do not give exact risk percentages unless the user explicitly asks AND you can clearly label them as ranges that vary by family history and result details.
- Encourage the patient to confirm result details with their ordering clinician/genetic counselor (e.g., whether the variant is pathogenic/likely pathogenic vs VUS).
- When discussing next steps, describe common guideline-aligned options (e.g., screening, risk-reducing surgery) as “things to discuss,” not directives.
- Avoid jargon. If you must use a term (e.g., “pathogenic variant,” “VUS”), define it in plain language.
- Be evidence-minded and avoid making up facts. If something depends on details you don’t have, say so and ask a clarifying question.
- If the user asks for treatment decisions, defer to clinicians and suggest questions to ask.

Conversation style:
- Start with validation/empathetic acknowledgment.
- Ask 2–4 clarifying questions that matter for interpretation.
- Provide a structured, practical summary:
  1) What the result may mean
  2) What people commonly discuss for screening / prevention
  3) Family implications and cascade testing
  4) What to do next (appointments, documents to gather)
  5) Resources and support

Output requirements:
- Use inclusive language (partner(s), family members, caregiver).
- Avoid shame/blame.
- Keep it practical and kind.
"""


def get_sample_patient_message() -> str:
    """A realistic prompt from a patient."""
    return (
        "I just got a message from my doctor that my BRCA1 test was 'positive'. "
        "I'm 34 and I'm freaking out. Does this mean I will get cancer? "
        "Should my sister get tested? What do I do next?"
    )


def fetch_pubmed_evidence_snippets(query: str, max_results: int = 3) -> str:
    """
    Best-effort: pull a few PubMed records and format brief snippets for grounding.
    This is not a replacement for guidelines, but can encourage evidence-minded responses.
    """
    if not ACTION_SPACE_AVAILABLE:
        return ""

    try:
        records: List[Dict[str, Any]] = query_pubmed(query, max_results=max_results, include_mesh=True, include_citations=False)  # type: ignore
        if not records:
            return ""

        lines: List[str] = []
        lines.append("Evidence context (PubMed, brief; not a guideline substitute):")
        for r in records[:max_results]:
            pmid = r.get("pmid", "NA")
            title = (r.get("title") or "").strip()
            year = r.get("year", "NA")
            journal = (r.get("journal") or "").strip()
            abstract = (r.get("abstract") or "").strip()
            abstract_one_line = " ".join(abstract.split())
            if len(abstract_one_line) > 350:
                abstract_one_line = abstract_one_line[:350].rstrip() + "…"

            lines.append(f"- PMID {pmid} ({year}, {journal}): {title}")
            if abstract_one_line:
                lines.append(f"  Summary: {abstract_one_line}")
        return "\n".join(lines).strip() + "\n"
    except Exception:
        return ""


def build_messages(system_prompt: str, user_message: str, evidence_context: str = "") -> List[Dict[str, str]]:
    """
    Builds OpenAI chat messages. Evidence context is appended as non-authoritative background.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if evidence_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Optional background context for the assistant. "
                    "Do not treat this as complete guidance; do not invent facts; "
                    "if uncertain, say so.\n\n" + evidence_context
                ),
            }
        )
    messages.append({"role": "user", "content": user_message})
    return messages


def call_llm(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    """
    Calls the OpenAI API. If unavailable, returns a short simulated response.
    """
    if not OPENAI_AVAILABLE:
        return simulate_response(messages)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return simulate_response(messages)

    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=900,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return simulate_response(messages)


def simulate_response(messages: List[Dict[str, str]]) -> str:
    """
    A minimal fallback response showing the *expected shape* of the counselor output.
    """
    # Detect whether this is the counselor persona by checking the system content.
    sys_text = " ".join(m["content"] for m in messages if m["role"] == "system").lower()
    if "genetic counselor" in sys_text or "hereditary cancer" in sys_text:
        return (
            "I’m really sorry you’re going through this—getting a “positive” BRCA1 message can feel overwhelming.\n\n"
            "**First, a quick clarification:** “Positive” can mean different things depending on whether the result is a **pathogenic/likely pathogenic** change (more clearly linked to risk) or a **VUS** (a change we don’t yet understand). If you can share the exact wording, that helps.\n\n"
            "**A few questions (to guide next steps):**\n"
            "1) Does the report say *pathogenic/likely pathogenic* or *VUS*?\n"
            "2) Any personal history of breast/ovarian cancer?\n"
            "3) Any close relatives with breast, ovarian, pancreatic, or prostate cancer—especially at younger ages?\n\n"
            "**What this may mean (in plain language):**\n"
            "BRCA1 is like a “spell-check” gene for DNA. Some changes can weaken that spell-check and raise the chance of certain cancers—but it does *not* mean cancer is guaranteed.\n\n"
            "**Common topics to discuss with your care team:**\n"
            "- Earlier and more frequent screening (often includes MRI and mammography)\n"
            "- Options that lower risk (medications or surgery for some people)\n"
            "- Planning around fertility, menopause timing, and personal values\n\n"
            "**Family screening:**\n"
            "If this is a pathogenic/likely pathogenic BRCA1 result, close relatives (like siblings) often consider targeted testing for the same change. This is called **cascade testing**.\n\n"
            "**What to do next (practical steps):**\n"
            "1) Ask for a copy of the full lab report.\n"
            "2) Book a visit with a genetic counselor or genetics clinic.\n"
            "3) Bring a simple family history (who had what cancer and at what age).\n\n"
            "If you want, paste the exact result wording (remove personal details), and I’ll help you translate it into plain English and draft questions for your appointment."
        )

    return (
        "Here’s a general overview: BRCA1 is a gene linked to hereditary cancer risk. "
        "A “positive” result can mean increased risk, and family members may consider testing. "
        "You should discuss next steps with a clinician or genetic counselor."
    )


def print_separator(char: str = "=", width: int = 88) -> None:
    print("\n" + (char * width) + "\n")


def demo_comparison(patient_message: str) -> None:
    """
    Runs the same patient message through:
      1) Control prompt ("helpful assistant")
      2) Genetic counselor persona prompt
    and prints both outputs for comparison.
    """
    evidence = fetch_pubmed_evidence_snippets(
        query="BRCA1 genetic counseling screening risk management",
        max_results=3,
    )

    print_separator("=")
    print("ROLE-BASED PROMPTING DEMO: Genetic Counselor vs Baseline Assistant")
    print_separator("=")
    print("[PATIENT MESSAGE]")
    print_separator("-")
    print(patient_message)
    print_separator("-")

    # 1) Baseline assistant
    print("[1) BASELINE OUTPUT — Helpful Assistant]")
    print_separator("-")
    base_messages = build_messages(get_control_system_prompt(), patient_message)
    baseline = call_llm(base_messages, model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    print(baseline.strip())
    print_separator("-")

    # 2) Genetic counselor persona
    print("[2) PERSONA OUTPUT — Genetic Counselor]")
    print_separator("-")
    counselor_messages = build_messages(get_genetic_counselor_system_prompt(), patient_message, evidence_context=evidence)
    counselor = call_llm(counselor_messages, model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    print(counselor.strip())
    print_separator("-")


def interactive_chat() -> None:
    """
    Simple interactive chat maintaining the counselor persona across turns.
    Type messages and press Enter. Type 'exit' to quit.
    """
    evidence = fetch_pubmed_evidence_snippets(
        query="BRCA1 genetic counseling screening risk management",
        max_results=3,
    )

    system_prompt = get_genetic_counselor_system_prompt()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if evidence:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Optional background context for the assistant. "
                    "Do not treat this as complete guidance; do not invent facts; "
                    "if uncertain, say so.\n\n" + evidence
                ),
            }
        )

    print_separator("=")
    print("INTERACTIVE MODE: Genetic Counselor Persona")
    print("Type 'exit' to quit.\n")
    print("Tip: You can paste the exact lab wording (with personal info removed).")
    print_separator("=")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_text})
        assistant_text = call_llm(messages, model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL)).strip()
        messages.append({"role": "assistant", "content": assistant_text})

        print("\nCounselor:\n")
        print(assistant_text)


def main() -> None:
    load_dotenv(".env")

    # If the script is piped input, use it as the patient message; otherwise use sample.
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        patient_message = piped if piped else get_sample_patient_message()
    else:
        patient_message = get_sample_patient_message()

    # Print a quick config summary
    print_separator("=")
    print("Config")
    print_separator("-")
    print(f"OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
    print(f"ACTION_SPACE_AVAILABLE (PubMed helper): {ACTION_SPACE_AVAILABLE}")
    print(f"Model: {os.getenv('OPENAI_MODEL', DEFAULT_MODEL)}")
    print(f"Base URL: {BASE_URL}")
    print(f"API key loaded: {'YES' if bool(os.getenv('OPENAI_API_KEY')) else 'NO'}")
    print_separator("=")

    # Demo comparison then interactive chat
    demo_comparison(patient_message)
    interactive_chat()


if __name__ == "__main__":
    main()

