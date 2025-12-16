#!/usr/bin/env python3
"""
Dietary Planning Agent — STUDENT SCAFFOLD (Course 1, low-effort)
===============================================================
Udacity Life Sciences Agentic AI Nanodegree — Course 1
Module: Applying Prompt Instruction Refinement with Python
Demo: Dietary Planning Agent

Learning objective:
Apply a systematic, multi-step refinement process to a prompt to achieve a precise,
structured output from an LLM.

Student work:
ONLY edit the three prompt functions:
  - prompt_v0()
  - prompt_v1()
  - prompt_v2_json()

Everything else is provided so you can run immediately.

Run modes:
- With LLM: put OPENAI_API_KEY in .env
- Offline: set MOCK_MODE=true in .env
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# -----------------------------
# Config / environment
# -----------------------------
load_dotenv(".env")

BASE_URL = "https://openai.vocareum.com/v1"
MODEL = "gpt-4o-mini"

HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    HAS_OPENAI = False


# -----------------------------
# Data models
# -----------------------------
@dataclass
class PatientProfile:
    age: int
    conditions: List[str]
    allergies: List[str]
    medications: List[str]
    dietary_restrictions: List[str]
    preferences: List[str]

    def minimized_for_prompt(self) -> Dict[str, Any]:
        # Already minimized (no identifiers). Keep stable shape for prompting.
        return {
            "age": self.age,
            "conditions": self.conditions,
            "allergies": self.allergies,
            "medications": self.medications,
            "dietary_restrictions": self.dietary_restrictions,
            "preferences": self.preferences,
        }


@dataclass
class Recipe:
    name: str
    ingredients: List[str]
    nutritional_info: Dict[str, Any]


# -----------------------------
# JSON extraction + lightweight validation
# -----------------------------
def extract_json_best_effort(text: str) -> Optional[Dict[str, Any]]:
    """
    Parses JSON even if the model wraps it in markdown or extra text.
    - Supports ```json ... ```
    - Otherwise extracts from first '{' to last '}'.
    """
    if not text or not isinstance(text, str):
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None

    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]

    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def validate_output(obj: Any, expected_recipe_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Very lightweight validation for V2.

    Expected JSON shape:
      {
        "patient_summary": "string",
        "decisions": [
          {"recipe_name": "...", "verdict": "YES|MAYBE|NO", "reasons": ["...", "..."]}
        ]
      }
    """
    errors: List[str] = []
    verdicts = {"YES", "MAYBE", "NO"}
    expected_set = set(expected_recipe_names)

    if not isinstance(obj, dict):
        return False, ["Top-level output must be a JSON object."]

    if "patient_summary" not in obj or not isinstance(obj["patient_summary"], str) or not obj["patient_summary"].strip():
        errors.append('Missing/invalid "patient_summary" (must be non-empty string).')

    if "decisions" not in obj or not isinstance(obj["decisions"], list):
        errors.append('Missing/invalid "decisions" (must be a list).')
        return False, errors

    seen: set[str] = set()
    for i, item in enumerate(obj["decisions"]):
        if not isinstance(item, dict):
            errors.append(f'decisions[{i}] must be an object.')
            continue

        rn = item.get("recipe_name")
        if not isinstance(rn, str) or rn not in expected_set:
            errors.append(f'decisions[{i}].recipe_name must match an input recipe name.')
        else:
            seen.add(rn)

        v = item.get("verdict")
        if v not in verdicts:
            errors.append(f'decisions[{i}].verdict must be one of {sorted(verdicts)}.')

        reasons = item.get("reasons")
        if not isinstance(reasons, list) or not (1 <= len(reasons) <= 3) or not all(isinstance(x, str) and x.strip() for x in reasons):
            errors.append(f'decisions[{i}].reasons must be a list of 1–3 non-empty strings.')

    missing = expected_set - seen
    if missing:
        errors.append(f"Missing decisions for recipes: {sorted(missing)}")

    return (len(errors) == 0), errors


# -----------------------------
# Agent
# -----------------------------
class DietaryPlanningAgent:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.mock_mode = os.getenv("MOCK_MODE", "").strip().lower() in {"1", "true", "yes"}
        self.use_llm = HAS_OPENAI and (not self.mock_mode) and bool(self.api_key)

        self.client = None
        if self.use_llm:
            self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)
        else:
            if not HAS_OPENAI:
                print("Warning: openai package not available. Running in MOCK_MODE.")
            elif not self.api_key:
                print("Warning: OPENAI_API_KEY missing. Running in MOCK_MODE.")
            else:
                print("MOCK_MODE enabled. Running without LLM calls.")

    # -------------------------
    # PROMPT REFINEMENT (STUDENT TODOs)
    # -------------------------
    def prompt_v0(self, patient: Dict[str, Any], recipes: List[Dict[str, Any]]) -> str:
        """
        TODO (2–4 minutes):
        Write a naive, underspecified prompt that will likely produce messy/unstructured output.
        Keep it short. Include patient + recipes.

        Tip: This is intentionally "bad" so V1/V2 look much better.
        """
        # TODO: replace the string below with your V0 prompt
        return f"""
TODO: Write a naive prompt here.

Patient: {json.dumps(patient)}
Recipes: {json.dumps(recipes)}
""".strip()

    def prompt_v1(self, patient: Dict[str, Any], recipes: List[Dict[str, Any]]) -> str:
        """
        TODO (5–10 minutes):
        Refine V0 by adding:
          - Role: clinical dietary screening assistant (educational support only)
          - Privacy constraint: no identifiers, don't invent details
          - Conservative uncertainty: if unsure, use MAYBE and say what info is missing
          - Per-recipe decision: YES/NO/MAYBE + 1–3 short reasons

        Output can be plain text (no JSON yet).
        """
        # TODO: replace the string below with your V1 prompt
        return f"""
TODO: Write a safer, more specific prompt here.

Patient summary (minimized):
{json.dumps(patient, indent=2)}

Recipes:
{json.dumps(recipes, indent=2)}
""".strip()

    def prompt_v2_json(self, patient: Dict[str, Any], recipes: List[Dict[str, Any]]) -> str:
        """
        TODO (10–15 minutes):
        Refine V1 to enforce structured output:
          - "Return ONLY valid JSON (no markdown, no extra text)."
          - Use this simple schema:

            {{
              "patient_summary": "one sentence (no identifiers)",
              "decisions": [
                {{
                  "recipe_name": "exact name from input",
                  "verdict": "YES|MAYBE|NO",
                  "reasons": ["1-3 short reasons"]
                }}
              ]
            }}

        Also add a small rules section (keep it short):
          - Allergies: if allergen present/likely → NO
          - Statins (e.g., atorvastatin): avoid grapefruit
          - Hypertension/low sodium: flag soy sauce/broth/cured meats
          - Diabetes: flag added sugar/refined carbs; suggest caution
        """
        # TODO: replace the string below with your V2 JSON-only prompt
        return f"""
TODO: Write the JSON-only prompt here.

Patient summary (minimized):
{json.dumps(patient, indent=2)}

Recipes:
{json.dumps(recipes, indent=2)}
""".strip()

    # -------------------------
    # LLM call
    # -------------------------
    def _chat(self, prompt: str, temperature: float = 0.2) -> str:
        if not self.use_llm or not self.client:
            return self._mock_response(prompt)

        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful clinical dietary screening assistant. "
                        "Be conservative with uncertainty. Follow the user's output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    # -------------------------
    # Demo runner
    # -------------------------
    def run_demo(self, patient: PatientProfile, recipes: List[Recipe]) -> None:
        patient_min = patient.minimized_for_prompt()
        recipes_min = [asdict(r) for r in recipes]
        recipe_names = [r.name for r in recipes]

        steps = [
            ("V0 (naive)", self.prompt_v0),
            ("V1 (role + privacy + uncertainty)", self.prompt_v1),
            ("V2 (JSON-only + simple rules)", self.prompt_v2_json),
        ]

        print("\n" + "=" * 80)
        print("DIETARY PLANNING AGENT — PROMPT INSTRUCTION REFINEMENT (SIMPLE)")
        print("=" * 80)
        print("Educational demo only. Not medical advice.\n")

        for i, (label, fn) in enumerate(steps, start=1):
            prompt = fn(patient_min, recipes_min)

            print("\n" + "-" * 80)
            print(f"STEP {i}: {label}")
            print("-" * 80)
            print("PROMPT (preview):")
            preview = prompt if len(prompt) <= 900 else (prompt[:900] + "\n... [truncated] ...")
            print(preview)

            out = self._chat(prompt, temperature=0.25 if i < 3 else 0.1)
            print("\nMODEL OUTPUT:")
            print(out)

            if i == 3:
                parsed = extract_json_best_effort(out)
                if parsed is None:
                    print("\nVALIDATION: FAILED ❌ (could not parse JSON)")
                    continue

                ok, errs = validate_output(parsed, recipe_names)
                if ok:
                    print("\nVALIDATION: PASSED ✅")
                    print("\nPARSED JSON:")
                    print(json.dumps(parsed, indent=2))
                else:
                    print("\nVALIDATION: FAILED ❌")
                    for e in errs:
                        print(f"- {e}")

        print("\n" + "=" * 80)
        print("Takeaway:")
        print("- V0 → V1: role + privacy constraints + conservative uncertainty handling")
        print("- V1 → V2: structured JSON output for predictable downstream use")
        print("=" * 80)

    # -------------------------
    # Offline mock output
    # -------------------------
    def _mock_response(self, prompt: str) -> str:
        # If student wrote a JSON-only instruction, return deterministic valid JSON.
        wants_json = "ONLY valid JSON" in prompt or '"decisions"' in prompt

        if wants_json:
            return json.dumps(
                {
                    "patient_summary": "58yo with T2D/HTN/hyperlipidemia and tree-nut allergy; screen recipes for allergens, sodium, and sugar.",
                    "decisions": [
                        {
                            "recipe_name": "Grilled Salmon with Vegetables",
                            "verdict": "YES",
                            "reasons": [
                                "No tree nuts listed; generally heart-healthy.",
                                "Low added sugar; protein-forward meal.",
                            ],
                        },
                        {
                            "recipe_name": "Almond-Crusted Chicken",
                            "verdict": "NO",
                            "reasons": [
                                "Contains almonds; tree-nut allergy risk.",
                                "High cross-contamination concern.",
                            ],
                        },
                        {
                            "recipe_name": "Vegetable Stir-Fry",
                            "verdict": "MAYBE",
                            "reasons": [
                                "Soy sauce is often high sodium (HTN/low sodium).",
                                "Carb source (rice) may need portion control for T2D.",
                            ],
                        },
                        {
                            "recipe_name": "Beef Stroganoff with Noodles",
                            "verdict": "MAYBE",
                            "reasons": [
                                "Likely high saturated fat (butter/sour cream).",
                                "Broth/noodles may increase sodium/refined carbs.",
                            ],
                        },
                    ],
                },
                indent=2,
            )

        # Otherwise return short, messy-ish text (like a naive model output)
        return (
            "MOCK MODE\n"
            "- Salmon meal seems good.\n"
            "- Almond chicken is bad due to nut allergy.\n"
            "- Stir-fry depends on sodium.\n"
            "- Stroganoff maybe ok but heavy.\n"
        )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    patient = PatientProfile(
        age=58,
        conditions=["Type 2 Diabetes", "Hypertension", "High Cholesterol"],
        allergies=["Tree nuts (almonds, walnuts)"],
        medications=["Metformin", "Lisinopril", "Atorvastatin"],
        dietary_restrictions=["Low sodium (< 2000mg/day)", "Low added sugar", "Heart-healthy fats"],
        preferences=["Prefers fish over red meat", "Enjoys vegetables"],
    )

    recipes = [
        Recipe(
            name="Grilled Salmon with Vegetables",
            ingredients=["salmon fillet", "broccoli", "bell peppers", "olive oil", "lemon", "garlic", "herbs"],
            nutritional_info={"sodium": "150mg", "sugar": "3g", "protein": "35g"},
        ),
        Recipe(
            name="Almond-Crusted Chicken",
            ingredients=["chicken breast", "crushed almonds", "parmesan cheese", "eggs", "herbs", "olive oil"],
            nutritional_info={"sodium": "300mg", "sugar": "1g", "protein": "40g"},
        ),
        Recipe(
            name="Vegetable Stir-Fry",
            ingredients=["mixed vegetables", "tofu", "soy sauce", "ginger", "garlic", "sesame oil", "brown rice"],
            nutritional_info={"sodium": "800mg", "sugar": "5g", "protein": "15g"},
        ),
        Recipe(
            name="Beef Stroganoff with Noodles",
            ingredients=["beef strips", "egg noodles", "sour cream", "mushrooms", "onions", "beef broth", "butter"],
            nutritional_info={"sodium": "1200mg", "sugar": "4g", "protein": "30g"},
        ),
    ]

    agent = DietaryPlanningAgent()
    agent.run_demo(patient, recipes)


if __name__ == "__main__":
    main()

