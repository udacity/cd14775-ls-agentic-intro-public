#!/usr/bin/env python3
"""
Genetic Counselor Persona Demo — STUDENT SCAFFOLD (very low effort)

Course: Life Sciences AI Nanodegree — Course 1
Module: Implementing Role-Based Prompting with Python
Learning objective: Apply role-based prompting in Python to generate persona-driven LLM responses.

Student work (ONLY learning-objective TODOs):
- Write a baseline system prompt
- Write a genetic counselor persona system prompt

Everything else is provided and runnable.
"""

import os
import sys
from typing import Dict, List

from dotenv import load_dotenv

# OpenAI client import (best-effort)
try:
    from openai import OpenAI  # type: ignore

    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


BASE_URL = "https://openai.vocareum.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


# ----------------------------
# TODOs (learning objective)
# ----------------------------
def get_control_system_prompt() -> str:
    """
    TODO:
    Return a plain baseline system prompt (no persona).
    Example: "You are a helpful assistant."
    """
    # TODO: write baseline prompt
    return "TODO"


def get_genetic_counselor_system_prompt() -> str:
    """
    TODO:
    Return a system prompt that enforces a *genetic counselor persona*.

    Keep it short (8–14 lines). Include:
    - Tone: calm, empathetic, inclusive, ~10th-grade reading level
    - Structure: headings + short paragraphs
    - Safety: not a doctor; no diagnosis/personal medical advice
    - Behavior: ask 2–3 clarifying questions; suggest "things to discuss with a clinician/genetic counselor"
    - Jargon: define briefly if used (e.g., VUS, pathogenic variant)
    """
    # TODO: write counselor persona prompt
    return "TODO"


# ----------------------------
# Provided content (no TODOs)
# ----------------------------
def get_sample_patient_message() -> str:
    return (
        "I just got a message from my doctor that my BRCA1 test was 'positive'. "
        "I'm 34 and I'm freaking out. Does this mean I will get cancer? "
        "Should my sister get tested? What do I do next?"
    )


def build_messages(system_prompt: str, user_message: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def call_llm(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    """
    Calls the OpenAI API. If unavailable (or no API key), returns a small simulated response.
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
            max_tokens=800,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return simulate_response(messages)


def simulate_response(messages: List[Dict[str, str]]) -> str:
    """
    Fallback output so the demo still runs without an API key.
    It detects the persona by looking for "genetic counselor" in the system prompt.
    """
    sys_text = " ".join(m["content"] for m in messages if m["role"] == "system").lower()
    is_counselor = "genetic counselor" in sys_text or "hereditary cancer" in sys_text or "brca" in sys_text

    if is_counselor:
        return (
            "I’m really sorry you’re dealing with this—getting a “positive” result message can feel scary.\n\n"
            "### A few quick questions (so the result is interpreted correctly)\n"
            "1) Does the report say **pathogenic/likely pathogenic** or **VUS** (variant of uncertain significance)?\n"
            "2) Any personal history of breast/ovarian cancer?\n"
            "3) Any close relatives with breast/ovarian/pancreatic/prostate cancer, and at what ages?\n\n"
            "### What a “positive BRCA1” *may* mean\n"
            "BRCA1 is like a DNA “repair” gene. Some changes can raise the chance of certain cancers, "
            "but it does **not** mean cancer is guaranteed.\n\n"
            "### Common next-step topics to discuss with your care team\n"
            "- Earlier/more frequent screening (often includes MRI and mammogram)\n"
            "- Risk-reducing options (medication or surgery for some people)\n"
            "- Family planning and your personal preferences/values\n\n"
            "### Family implications\n"
            "If it’s a pathogenic/likely pathogenic BRCA1 change, close relatives (like siblings) often consider testing "
            "for the same change (“cascade testing”).\n\n"
            "### Practical next steps\n"
            "1) Ask for a copy of the full lab report.\n"
            "2) Book a visit with a genetic counselor/genetics clinic.\n"
            "3) Bring a simple family history (who had what cancer and what age).\n\n"
            "If you paste the exact wording from the report (remove personal details), I can help translate it into plain English "
            "and suggest questions to ask at your appointment."
        )

    return (
        "A positive BRCA1 test can be linked to higher cancer risk, but it doesn’t guarantee you will get cancer. "
        "It’s a good idea to discuss screening and next steps with your clinician, and family members may consider testing."
    )


def print_separator(char: str = "=", width: int = 80) -> None:
    print("\n" + (char * width) + "\n")


def demo_comparison(patient_message: str, model: str) -> None:
    print_separator("=")
    print("ROLE-BASED PROMPTING DEMO: Baseline vs Genetic Counselor Persona")
    print_separator("=")

    print("[PATIENT MESSAGE]")
    print_separator("-")
    print(patient_message)
    print_separator("-")

    # 1) Baseline
    print("[1) BASELINE OUTPUT]")
    print_separator("-")
    baseline_msgs = build_messages(get_control_system_prompt(), patient_message)
    print(call_llm(baseline_msgs, model=model))
    print_separator("-")

    # 2) Persona
    print("[2) PERSONA OUTPUT — Genetic Counselor]")
    print_separator("-")
    counselor_msgs = build_messages(get_genetic_counselor_system_prompt(), patient_message)
    print(call_llm(counselor_msgs, model=model))
    print_separator("-")


def interactive_chat(model: str) -> None:
    """
    Optional: maintain persona across turns. Type 'exit' to quit.
    """
    system_prompt = get_genetic_counselor_system_prompt()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print_separator("=")
    print("INTERACTIVE MODE: Genetic Counselor Persona (type 'exit' to quit)")
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
        assistant_text = call_llm(messages, model=model)
        messages.append({"role": "assistant", "content": assistant_text})

        print("\nCounselor:\n")
        print(assistant_text)


def main() -> None:
    load_dotenv(".env")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    # If script is piped input, use it; else default sample
    if not sys.stdin.isatty():
        piped = (sys.stdin.read() or "").strip()
        patient_message = piped if piped else get_sample_patient_message()
    else:
        patient_message = get_sample_patient_message()

    # Minimal config summary
    print_separator("=")
    print("Config")
    print_separator("-")
    print(f"OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
    print(f"Model: {model}")
    print(f"Base URL: {BASE_URL}")
    print(f"API key loaded: {'YES' if bool(os.getenv('OPENAI_API_KEY')) else 'NO'}")
    print_separator("=")

    demo_comparison(patient_message, model=model)

    # Uncomment to enable interactive mode
    # interactive_chat(model=model)


if __name__ == "__main__":
    main()

