#!/usr/bin/env python3
"""
Dietary Planning Agent Demo (Life Sciences version)
===================================================
Part of: Udacity Life Sciences Agentic AI Nanodegree
Course 1 Module: Applying Prompt Instruction Refinement with Python
Exercise: Dietary Planning Agent

Scenario:
Match recipes to a patient's restrictions and medication/condition risks.
Demonstrates systematic prompt instruction refinement to produce precise,
structured, safety-first outputs suitable for clinical workflow triage.

Notes:
- Educational demo only (NOT medical advice).
- Demonstrates PHI minimization + consistent JSON outputs + basic validation.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv(".env")

BASE_URL = "https://openai.vocareum.com/v1"
MODEL = "gpt-4o-mini"


# -----------------------------
# Optional OpenAI client setup
# -----------------------------
HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    HAS_OPENAI = False


# -----------------------------
# Data models (minimal)
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
        """
        PHI minimization: omit identifiers (name, MRN, etc.) and keep only the
        information needed for the dietary safety task.
        """
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
# Utility: JSON schema-ish validation
# -----------------------------
REQUIRED_TOP_KEYS = {"patient_summary", "recipe_analysis", "top_recommendations", "absolutely_avoid"}
RECIPE_REQUIRED_KEYS = {
    "recipe_name",
    "status",  # "safe" | "caution" | "avoid"
    "safety_score",  # 0-10 int
    "reasons",
    "flagged_ingredients",
    "interaction_risks",
    "warnings",
    "modifications",
    "unknowns",
    "confidence",  # 0-1 float
}

ALLOWED_STATUS = {"safe", "caution", "avoid"}


def _is_int_0_10(x: Any) -> bool:
    return isinstance(x, int) and 0 <= x <= 10


def _is_float_0_1(x: Any) -> bool:
    return isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0


def validate_final_json(obj: Any) -> Tuple[bool, List[str]]:
    """
    Lightweight validator for the final strict JSON output.
    Returns (ok, errors).
    """
    errors: List[str] = []

    if not isinstance(obj, dict):
        return False, ["Top-level output must be a JSON object."]

    missing_top = REQUIRED_TOP_KEYS - set(obj.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")

    if "recipe_analysis" in obj and not isinstance(obj["recipe_analysis"], list):
        errors.append("recipe_analysis must be a list.")

    if "top_recommendations" in obj and not isinstance(obj["top_recommendations"], list):
        errors.append("top_recommendations must be a list of recipe names.")

    if "absolutely_avoid" in obj and not isinstance(obj["absolutely_avoid"], list):
        errors.append("absolutely_avoid must be a list of recipe names.")

    ra = obj.get("recipe_analysis", [])
    if isinstance(ra, list):
        for i, item in enumerate(ra):
            if not isinstance(item, dict):
                errors.append(f"recipe_analysis[{i}] must be an object.")
                continue

            missing = RECIPE_REQUIRED_KEYS - set(item.keys())
            if missing:
                errors.append(f"recipe_analysis[{i}] missing keys: {sorted(missing)}")

            status = item.get("status")
            if status not in ALLOWED_STATUS:
                errors.append(f"recipe_analysis[{i}].status must be one of {sorted(ALLOWED_STATUS)}")

            if not _is_int_0_10(item.get("safety_score")):
                errors.append(f"recipe_analysis[{i}].safety_score must be int 0-10")

            if not isinstance(item.get("reasons"), list):
                errors.append(f"recipe_analysis[{i}].reasons must be a list of strings")

            if not isinstance(item.get("flagged_ingredients"), list):
                errors.append(f"recipe_analysis[{i}].flagged_ingredients must be a list")

            if not isinstance(item.get("interaction_risks"), list):
                errors.append(f"recipe_analysis[{i}].interaction_risks must be a list")

            if not isinstance(item.get("warnings"), list):
                errors.append(f"recipe_analysis[{i}].warnings must be a list")

            # modifications can be str or null
            mods = item.get("modifications")
            if mods is not None and not isinstance(mods, str):
                errors.append(f"recipe_analysis[{i}].modifications must be a string or null")

            if not isinstance(item.get("unknowns"), list):
                errors.append(f"recipe_analysis[{i}].unknowns must be a list")

            if not _is_float_0_1(item.get("confidence")):
                errors.append(f"recipe_analysis[{i}].confidence must be float 0-1")

    return (len(errors) == 0), errors


def extract_json_best_effort(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to parse JSON even if the model wrapped it in text/code fences.
    """
    # Strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Try to find first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except Exception:
        return None


# -----------------------------
# Agent
# -----------------------------
class DietaryPlanningAgent:
    def __init__(self):
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

    # ---- Prompt refinement steps ----
    def prompt_v0(self, patient: Dict[str, Any], recipes: List[Dict[str, Any]]) -> str:
        return f"""
Given patient info and recipes, recommend what the patient should eat.

Patient: {json.dumps(patient)}
Recipes: {json.dumps(recipes)}

Return your recommendations.
""".strip()

    def prompt_v1(self, patient: Dict[str, Any], recipes: List[Dict[str, Any]]) -> str:
        return f"""
You are a dietary planning assistant supporting clinicians.

Privacy:
- Do NOT include any identifying info (names, IDs). Use only the provided patient summary.
- Educational support only; advise clinician review.

Patient summary (minimized):
{json.dumps(patient, indent=2)}

Recipes:
{json.dumps(recipes, indent=2)}

Task:
For each recipe, decide if it is suitable for this patient (YES/NO/MAYBE) and give 1–3 short reasons.
Be conservative: if uncertain, choose MAYBE and explain what is missing.
""".strip()

    def prompt_v2(self, patient: Dict[str, Any], recipes: List[Dict[str, Any]]) -> str:
        return f"""
You are a dietary planning assistant supporting clinicians. Safety-first triage.

Privacy & safety:
- Do NOT invent patient details.
- Do NOT output identifiers.
- This is NOT medical advice; output is a screening aid.

Patient summary (minimized):
{json.dumps(patient, indent=2)}

Recipes:
{json.dumps(recipes, indent=2)}

Definitions / rules (use these explicitly):
- Allergies: If a recipe contains an allergen (or likely allergen), mark NOT SUITABLE.
- Diabetes (Type 2): Prefer lower added sugar; caution with high refined carbs; note carbohydrate sources.
- Hypertension / low sodium: Flag high-sodium ingredients (e.g., soy sauce, broth, cured meats). Prefer lower-sodium substitutions.
- Hyperlipidemia / heart-healthy: Caution with high saturated fat (e.g., butter, heavy cream, fatty cuts).
- Atorvastatin (statin): Avoid grapefruit/grapefruit juice (potential interaction).
- Lisinopril (ACE inhibitor): Caution with potassium salt substitutes; flag if recipe suggests them.
- Metformin: Flag heavy alcohol additions (screening caution).

Task:
For each recipe:
1) Suitability: SUITABLE / CAUTION / AVOID
2) List specific ingredients that triggered the decision
3) Suggest safe modifications where possible
4) Explicitly list unknowns/ambiguities (e.g., “broth” type, “soy sauce” sodium)
""".strip()

    def prompt_v3_strict_json(self, patient: Dict[str, Any], recipes: List[Dict[str, Any]]) -> str:
        return f"""
You are an expert dietary planning assistant supporting clinicians. Safety-first triage.

Privacy & constraints:
- Do NOT output any identifiers or extra patient details.
- Use only the patient summary + recipes provided.
- If information is missing, capture it in "unknowns" and downgrade confidence.
- Return ONLY valid JSON (no markdown, no commentary).

Patient summary (minimized):
{json.dumps(patient, indent=2)}

Recipes:
{json.dumps(recipes, indent=2)}

Scoring:
- status: "safe" | "caution" | "avoid"
- safety_score: integer 0–10
  0–3: dangerous / contraindicated
  4–6: risky, needs major changes or clinician review
  7–8: generally safe with minor changes
  9–10: highly suitable

Ambiguity handling (be conservative):
- If an ingredient category is unclear (e.g., "broth", "sauce", "seasoning mix"), add to unknowns and reduce confidence.
- If a recipe contains common hidden sodium sources (soy sauce, broth, cheese), treat as caution unless clearly low-sodium.
- If "optional" ingredients include allergens, treat as caution and recommend omission + cross-contamination warning.

Medication/condition screening reminders:
- Statins: avoid grapefruit.
- ACE inhibitors: caution potassium salt substitutes.
- Diabetes: flag high added sugar / refined carbs; suggest swaps.
- Hypertension: flag sodium; suggest low-sodium swaps.
- Hyperlipidemia: flag saturated fat; suggest swaps.

Required JSON schema:
{{
  "patient_summary": "one sentence screening summary (no identifiers)",
  "recipe_analysis": [
    {{
      "recipe_name": "exact name from input",
      "status": "safe|caution|avoid",
      "safety_score": 0,
      "reasons": ["..."],
      "flagged_ingredients": ["..."],
      "interaction_risks": ["..."],
      "warnings": ["..."],
      "modifications": "..." ,
      "unknowns": ["..."],
      "confidence": 0.0
    }}
  ],
  "top_recommendations": ["..."],
  "absolutely_avoid": ["..."]
}}

Generate the JSON now.
""".strip()

    # ---- LLM call + repair ----
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
                        "Be conservative with uncertainty. Output must follow user constraints."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def _repair_to_valid_json(self, raw_text: str, errors: List[str], patient: Dict[str, Any], recipes: List[Dict[str, Any]]) -> str:
        """
        One-shot repair prompt: ask the model to output valid JSON strictly matching schema.
        (This is a minimal 'validation gate' to enforce structured output.)
        """
        repair_prompt = f"""
Your previous output was not valid per the required JSON schema.

Validation errors:
{json.dumps(errors, indent=2)}

Please re-output ONLY valid JSON matching EXACTLY the required schema from this prompt.
Do not include markdown or extra text.

Patient summary (minimized):
{json.dumps(patient, indent=2)}

Recipes:
{json.dumps(recipes, indent=2)}

Required JSON schema (repeat):
{{
  "patient_summary": "one sentence screening summary (no identifiers)",
  "recipe_analysis": [
    {{
      "recipe_name": "exact name from input",
      "status": "safe|caution|avoid",
      "safety_score": 0,
      "reasons": ["..."],
      "flagged_ingredients": ["..."],
      "interaction_risks": ["..."],
      "warnings": ["..."],
      "modifications": "..." ,
      "unknowns": ["..."],
      "confidence": 0.0
    }}
  ],
  "top_recommendations": ["..."],
  "absolutely_avoid": ["..."]
}}
""".strip()
        return self._chat(repair_prompt, temperature=0.0)

    # ---- Demo runner ----
    def demonstrate(self, patient: PatientProfile, recipes: List[Recipe]) -> None:
        patient_min = patient.minimized_for_prompt()
        recipes_min = [asdict(r) for r in recipes]

        steps = [
            ("V0 (unrefined)", self.prompt_v0),
            ("V1 (role + basic structure)", self.prompt_v1),
            ("V2 (rules + ambiguity guidance)", self.prompt_v2),
            ("V3 (strict JSON schema)", self.prompt_v3_strict_json),
        ]

        print("\n" + "=" * 88)
        print("DIETARY PLANNING AGENT — PROMPT INSTRUCTION REFINEMENT DEMO")
        print("=" * 88)
        print("Educational demo only. Not medical advice.\n")

        for idx, (label, fn) in enumerate(steps, start=1):
            prompt = fn(patient_min, recipes_min)

            print("\n" + "-" * 88)
            print(f"STEP {idx}: {label}")
            print("-" * 88)
            print("PROMPT (preview):")
            preview = prompt if len(prompt) <= 900 else (prompt[:900] + "\n... [truncated] ...")
            print(preview)

            print("\nMODEL OUTPUT:")
            out = self._chat(prompt, temperature=0.2 if idx < 4 else 0.1)
            print(out)

            # Validate only on final step
            if idx == len(steps):
                parsed = extract_json_best_effort(out)
                if parsed is None:
                    ok, errors = False, ["Could not parse JSON from model output."]
                else:
                    ok, errors = validate_final_json(parsed)

                if not ok:
                    print("\nVALIDATION: FAILED")
                    for e in errors:
                        print(f"- {e}")

                    print("\nAttempting one-shot repair to valid JSON...\n")
                    repaired = self._repair_to_valid_json(out, errors, patient_min, recipes_min)
                    print("REPAIRED OUTPUT:")
                    print(repaired)

                    parsed2 = extract_json_best_effort(repaired)
                    if parsed2 is None:
                        print("\nREPAIR VALIDATION: FAILED (still not parseable JSON)")
                    else:
                        ok2, errors2 = validate_final_json(parsed2)
                        if ok2:
                            print("\nREPAIR VALIDATION: PASSED")
                            print("\nFINAL PARSED JSON:")
                            print(json.dumps(parsed2, indent=2))
                        else:
                            print("\nREPAIR VALIDATION: FAILED")
                            for e in errors2:
                                print(f"- {e}")
                else:
                    print("\nVALIDATION: PASSED")
                    print("\nFINAL PARSED JSON:")
                    print(json.dumps(parsed, indent=2))

        print("\n" + "=" * 88)
        print("What changed across refinements?")
        print("=" * 88)
        print(
            "- V0 → V1: Adds role, privacy constraints, and conservative uncertainty handling.\n"
            "- V1 → V2: Adds explicit condition/medication screening rules + ingredient-level traceability.\n"
            "- V2 → V3: Forces a strict JSON schema + scoring + ambiguity rules to produce consistent outputs.\n"
        )

    # ---- Mock response for offline runs ----
    def _mock_response(self, prompt: str) -> str:
        if "Required JSON schema" in prompt:
            return json.dumps(
                {
                    "patient_summary": "58yo with T2D, HTN, hyperlipidemia; tree-nut allergy; screening for sodium/sugar/sat-fat and key interactions.",
                    "recipe_analysis": [
                        {
                            "recipe_name": "Grilled Salmon with Vegetables",
                            "status": "safe",
                            "safety_score": 9,
                            "reasons": [
                                "No listed tree nuts; aligns with heart-healthy preference.",
                                "Low added sugar; protein-forward meal supports glycemic goals.",
                                "Sodium appears low if no added salt/processed sauces are used.",
                            ],
                            "flagged_ingredients": [],
                            "interaction_risks": [],
                            "warnings": [],
                            "modifications": "Avoid adding salt; use lemon/garlic/herbs for flavor.",
                            "unknowns": ["Exact portion sizes not provided."],
                            "confidence": 0.82,
                        },
                        {
                            "recipe_name": "Almond-Crusted Chicken",
                            "status": "avoid",
                            "safety_score": 1,
                            "reasons": [
                                "Contains almonds; patient has tree-nut allergy.",
                                "High risk of severe allergic reaction / cross-contamination.",
                            ],
                            "flagged_ingredients": ["crushed almonds"],
                            "interaction_risks": [],
                            "warnings": ["CRITICAL: Tree-nut allergen present."],
                            "modifications": None,
                            "unknowns": [],
                            "confidence": 0.95,
                        },
                        {
                            "recipe_name": "Vegetable Stir-Fry",
                            "status": "caution",
                            "safety_score": 7,
                            "reasons": [
                                "Soy sauce can be high sodium (HTN/low-sodium restriction).",
                                "Otherwise vegetable-forward and compatible with preferences.",
                            ],
                            "flagged_ingredients": ["soy sauce"],
                            "interaction_risks": [],
                            "warnings": ["Check sodium content; prefer low-sodium option."],
                            "modifications": "Use low-sodium soy sauce or reduce amount; add acid (lime/rice vinegar) for flavor.",
                            "unknowns": ["Soy sauce type/brand not specified."],
                            "confidence": 0.74,
                        },
                        {
                            "recipe_name": "Beef Stroganoff with Noodles",
                            "status": "caution",
                            "safety_score": 5,
                            "reasons": [
                                "Potentially high saturated fat (butter, sour cream) conflicting with heart-healthy goals.",
                                "Broth/noodles may contribute significant sodium/refined carbs depending on type/portion.",
                            ],
                            "flagged_ingredients": ["butter", "sour cream", "beef broth", "egg noodles"],
                            "interaction_risks": [],
                            "warnings": ["Assess sodium and saturated fat; consider portion control."],
                            "modifications": "Use low-sodium broth; swap Greek yogurt for sour cream; increase mushrooms/veg; consider whole-grain noodles.",
                            "unknowns": ["Broth sodium level not specified.", "Portion sizes not specified."],
                            "confidence": 0.63,
                        },
                    ],
                    "top_recommendations": ["Grilled Salmon with Vegetables", "Vegetable Stir-Fry"],
                    "absolutely_avoid": ["Almond-Crusted Chicken"],
                },
                indent=2,
            )

        # For earlier prompts, return a short illustrative text
        return (
            "MOCK MODE: (illustrative)\n"
            "- Grilled Salmon with Vegetables: YES (generally compatible; keep sodium low)\n"
            "- Almond-Crusted Chicken: NO (tree-nut allergen)\n"
            "- Vegetable Stir-Fry: MAYBE (soy sauce sodium)\n"
            "- Beef Stroganoff with Noodles: MAYBE/NO (sat fat + sodium + refined carbs)\n"
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
            nutritional_info={"sodium": "150mg", "sugar": "3g", "protein": "35g", "healthy_fats": "high"},
        ),
        Recipe(
            name="Almond-Crusted Chicken",
            ingredients=["chicken breast", "crushed almonds", "parmesan cheese", "eggs", "herbs", "olive oil"],
            nutritional_info={"sodium": "300mg", "sugar": "1g", "protein": "40g", "healthy_fats": "medium"},
        ),
        Recipe(
            name="Vegetable Stir-Fry",
            ingredients=["mixed vegetables", "tofu", "soy sauce", "ginger", "garlic", "sesame oil", "brown rice"],
            nutritional_info={"sodium": "800mg", "sugar": "5g", "protein": "15g", "healthy_fats": "medium"},
        ),
        Recipe(
            name="Beef Stroganoff with Noodles",
            ingredients=["beef strips", "egg noodles", "sour cream", "mushrooms", "onions", "beef broth", "butter"],
            nutritional_info={"sodium": "1200mg", "sugar": "4g", "protein": "30g", "saturated_fats": "high"},
        ),
    ]

    agent = DietaryPlanningAgent()
    agent.demonstrate(patient, recipes)


if __name__ == "__main__":
    main()

