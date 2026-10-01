#!/usr/bin/env python3
"""
simulate_survey_claude_stateless.py

Runs a targeted subset of the chatbot usage survey using Anthropic's Claude
(claude-haiku-4-5 by default) -- stateless, one prompt per respondent.
Uses the SAME standardized persona generator as the other five runs, so
personas align across the full 3x2 matrix., with the same census-weighted persona specs
as simulate_survey.py. Uses the same random seed so persona demographics
are identical — making cross-model comparison clean.

Questions asked:
  Q3  — Devices used last week
  Q4  — Chatbots ever used regularly
  Q5  — Usage frequency per chatbot
  Q9  — Have you stopped using a chatbot? Why?
  Q11 — What do you wish a chatbot could do better?
  Q15 — Pick 5 most valued future chatbot roles

Requirements:
    pip install anthropic tqdm

Set your Gemini API key:
    export GEMINI_API_KEY=...

Or pass it:
    python simulate_survey_claude_stateless.py --api-key ...

Output:
    claude_stateless_results.jsonl
    claude_stateless_results.csv
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Simulate targeted survey with Claude -- stateless")
parser.add_argument("--n",        type=int,   default=1000)
parser.add_argument("--api-key",  type=str,   default=None)
parser.add_argument("--model",    type=str,   default="claude-haiku-4-5")
parser.add_argument("--output",   type=str,   default="claude_stateless_results")
parser.add_argument("--delay",    type=float, default=0.25,
                    help="Seconds between calls (default default 0.25s).")
parser.add_argument("--resume",   action="store_true")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

if args.api_key:
    os.environ["ANTHROPIC_API_KEY"] = args.api_key

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: No Anthropic API key. Set ANTHROPIC_API_KEY or pass --api-key.", file=sys.stderr)
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic not installed.\n  pip install anthropic", file=sys.stderr)
    sys.exit(1)

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Census-weighted demographics  (identical to simulate_survey.py)
# IMPORTANT: random.seed(42) is set before spec generation so the 1,000
# personas are demographically identical to the Claude run — enabling
# clean per-question cross-model comparison.
# ---------------------------------------------------------------------------

AGE_BUCKETS = [
    ("18-24", 0.118), ("25-34", 0.177), ("35-44", 0.165),
    ("45-54", 0.157), ("55-64", 0.160), ("65+",   0.223),
]
GENDER = [
    ("man", 0.489), ("woman", 0.504),
    ("non-binary or gender-nonconforming person", 0.007),
]
RACE_ETHNICITY = [
    ("White non-Hispanic", 0.596), ("Hispanic or Latino", 0.187),
    ("Black or African American", 0.124), ("Asian American", 0.060),
    ("multiracial or another racial identity", 0.033),
]
EDUCATION = [
    ("less than a high school diploma", 0.087),
    ("a high school diploma or GED", 0.265),
    ("some college or an associate's degree", 0.278),
    ("a bachelor's degree", 0.224),
    ("a graduate or professional degree", 0.146),
]
EMPLOYMENT = [
    ("Employed full-time", 0.480), ("Employed part-time", 0.120),
    ("Self-employed or running a business", 0.065),
    ("Freelancing, consulting, or gig work", 0.040),
    ("Unemployed and looking for work", 0.035),
    ("Student", 0.060), ("Homemaker or full-time caregiver", 0.040),
    ("Retired", 0.130), ("Not currently seeking employment", 0.020),
    ("Disabled or unable to work due to health", 0.010),
]
REGION_STATES = {
    "Northeast": ["Maine","New Hampshire","Vermont","Massachusetts","Rhode Island",
                  "Connecticut","New York","New Jersey","Pennsylvania"],
    "Midwest":   ["Ohio","Indiana","Illinois","Michigan","Wisconsin","Minnesota",
                  "Iowa","Missouri","North Dakota","South Dakota","Nebraska","Kansas"],
    "South":     ["Delaware","Maryland","Virginia","West Virginia","North Carolina",
                  "South Carolina","Georgia","Florida","Kentucky","Tennessee",
                  "Alabama","Mississippi","Arkansas","Louisiana","Oklahoma","Texas"],
    "West":      ["Montana","Idaho","Wyoming","Colorado","New Mexico","Arizona",
                  "Utah","Nevada","Washington","Oregon","California","Alaska","Hawaii"],
}
REGION_WEIGHTS = [("Northeast",0.174),("Midwest",0.208),("South",0.381),("West",0.237)]
URBAN_RURAL    = [("Large city",0.310),("Mid-size city",0.200),("Suburban",0.310),
                  ("Small town",0.110),("Rural",0.070)]
POLITICAL      = [("Extremely liberal",0.090),("Moderately liberal",0.160),
                  ("Slightly liberal",0.110),("Moderate/middle of the road",0.250),
                  ("Slightly conservative",0.100),("Moderately conservative",0.180),
                  ("Extremely conservative",0.110)]
TECH_COMFORT   = [("Overwhelmed or unsure",0.150),("Cautiously comfortable",0.380),
                  ("Confident and curious",0.470)]


def weighted_choice(dist):
    return random.choices([d[0] for d in dist], weights=[d[1] for d in dist], k=1)[0]


def generate_persona_spec(respondent_id: int) -> dict:
    region = weighted_choice(REGION_WEIGHTS)
    return {
        "id":           respondent_id,
        "age_group":    weighted_choice(AGE_BUCKETS),
        "gender":       weighted_choice(GENDER),
        "race":         weighted_choice(RACE_ETHNICITY),
        "education":    weighted_choice(EDUCATION),
        "employment":   weighted_choice(EMPLOYMENT),
        "region":       region,
        "state":        random.choice(REGION_STATES[region]),
        "urban_rural":  weighted_choice(URBAN_RURAL),
        "politics":     weighted_choice(POLITICAL),
        "tech_comfort": weighted_choice(TECH_COMFORT),
    }


def spec_to_description(spec: dict) -> str:
    return (
        f"A {spec['age_group']}-year-old {spec['gender']} who identifies as {spec['race']}, "
        f"has {spec['education']}, is currently {spec['employment'].lower()}, "
        f"lives in {spec['state']} ({spec['urban_rural'].lower()} area), "
        f"describes their politics as '{spec['politics']}', "
        f"and rates their technology comfort as '{spec['tech_comfort']}'."
    )


# ---------------------------------------------------------------------------
# Question definitions
# ---------------------------------------------------------------------------

CHATBOTS = [
    "ChatGPT", "Claude", "Gemini", "CoPilot", "Perplexity",
    "Meta AI", "Grok", "Duck.ai", "Pi.AI", "DeepSeek",
    "Alexa / Alexa+", "Character.AI", "Chai", "Replika", "Siri", "Bixby",
]

DEVICES_Q3 = [
    "Wired headphones", "Earbuds or wireless headphones",
    "Apple HomePod or smart speaker (Sonos/Bose/JBL/Logitech)",
    "Amazon Echo / Alexa", "Google Nest smart speaker",
    "Smartwatch (watchOS)", "Smartwatch (WearOS)",
    "Chatbot (Claude/ChatGPT/etc.)", "Android phone", "Apple iPhone",
    "Video streaming device (Smart TV/Roku/Apple TV/etc.)",
    "Podcast (alone)", "Podcast (with others)",
    "Streaming music (alone)", "Streaming music (with others)",
    "Apple CarPlay", "Android Auto",
]

USAGE_FREQ_OPTIONS = [
    "Several times a day", "At least once a day", "At least once a week",
    "At least once a month", "Used regularly in the past but not anymore",
    "Tried once or twice but not anymore", "Never used",
]

CHATBOT_ROLES = [
    "Assistant", "Best Friend", "Chef", "Coach", "Companion",
    "Crisis Triage", "DJ", "Editor", "Encourager", "Grounding Partner",
    "Listener", "Mentor", "Parent", "Partner", "Teacher",
]

# ---------------------------------------------------------------------------
# Prompt  (targeted: Q3, Q4, Q5, Q9, Q11, Q15 only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are roleplaying as a survey respondent. Stay completely in character. "
    "Answer survey questions the way a real person would — briefly and naturally. "
    "For multiple-choice questions use only the options given. "
    "Return ONLY a valid JSON object — no markdown, no backticks, no commentary."
)

SURVEY_PROMPT = """\
You are a specific survey respondent. Answer every question as this person would.

YOUR PERSONA: {persona_description}

Answer this short survey about technology and chatbot usage.
Return a single JSON object with ALL of these keys:

{{
  "q3_devices_used_last_week": [
    // list zero or more from: {devices}
  ],

  "q4_chatbots_used": [
    // list from: {chatbots}
    // or the single string "I don't really use AI chatbots"
  ],

  "q5_usage_frequency": {{
    // For each chatbot you have EVER heard of or tried:
    // "<chatbot name>": "<one of: {freq_options}>"
    // Omit chatbots you have zero awareness of.
  }},

  "q9_stopped_using": "<open text: have you stopped using a chatbot and why? If no, say so briefly.>",

  "q11_wish_chatbot_could_do": "<open text: something you wish a chatbot could do better, or say nothing comes to mind>",

  "q15_future_roles": [
    // Pick EXACTLY 5 from: {roles}
  ]
}}
""".strip()


def build_prompt(spec: dict) -> str:
    return SURVEY_PROMPT.format(
        persona_description=spec_to_description(spec),
        devices=", ".join(DEVICES_Q3),
        chatbots=", ".join(CHATBOTS),
        freq_options=", ".join(USAGE_FREQ_OPTIONS),
        roles=", ".join(CHATBOT_ROLES),
    )


# ---------------------------------------------------------------------------
# ChatGPT API call with retry
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Model sometimes truncates. Try dropping lines from the end
        # until we can close the JSON structure cleanly.
        lines = text.rstrip().rstrip(',').split('\n')
        while lines:
            candidate = '\n'.join(lines).rstrip().rstrip(',')
            closing = (']' * max(0, candidate.count('[') - candidate.count(']')) +
                       '}' * max(0, candidate.count('{') - candidate.count('}')))
            try:
                return json.loads(candidate + closing)
            except json.JSONDecodeError:
                lines.pop()
        raise


def call_claude(prompt: str, respondent_id: int, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=4096,
                temperature=1.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_json(response.content[0].text)
        except json.JSONDecodeError as e:
            print(f"\n  [id={respondent_id}] JSON error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n  [id={respondent_id}] Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"\n  [id={respondent_id}] API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return {}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def load_completed_ids(path: str) -> set:
    completed = set()
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    completed.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    pass
    return completed


def write_csv(jsonl_path: str, csv_path: str):
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return
    flat_rows = []
    for row in rows:
        flat = {}
        for k, v in row.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}__{kk}"] = json.dumps(vv) if isinstance(vv, (dict, list)) else vv
            elif isinstance(v, list):
                flat[k] = "; ".join(str(x) for x in v)
            else:
                flat[k] = v
        flat_rows.append(flat)
    all_keys = list(dict.fromkeys(k for r in flat_rows for k in r))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"CSV written → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Same seed as simulate_survey.py — identical persona demographics
    random.seed(42)

    checkpoint = f"{args.output}.jsonl"
    csv_out    = f"{args.output}.csv"

    all_specs  = [generate_persona_spec(i) for i in range(args.n)]
    completed  = load_completed_ids(checkpoint) if args.resume else set()

    if completed:
        print(f"Resuming: {len(completed)} already done.")

    n_errors = 0

    print(f"Model:       {args.model}")
    print(f"Respondents: {args.n}")
    print(f"Questions:   Q3, Q4, Q5, Q9, Q11, Q15")
    print(f"Checkpoint:  {checkpoint}")
    print(f"CSV output:  {csv_out}\n")
    print("claude-haiku-4-5: fast and cheap. ~5-8 min for 1,000 at default delay.\n")

    

    with open(checkpoint, "a", encoding="utf-8") as out:
        for spec in tqdm(all_specs, desc="Respondents"):
            if spec["id"] in completed:
                continue

            answers = call_claude(build_prompt(spec), spec["id"])

            if not answers:
                n_errors += 1
                tqdm.write(f"  WARNING: empty response id={spec['id']} (errors so far: {n_errors})")

            record = {
                "id":           spec["id"],
                "model":        args.model + "_stateless",
                # demographic spec — same values as Claude run at same id
                "spec_age_group":    spec["age_group"],
                "spec_gender":       spec["gender"],
                "spec_race":         spec["race"],
                "spec_education":    spec["education"],
                "spec_employment":   spec["employment"],
                "spec_state":        spec["state"],
                "spec_region":       spec["region"],
                "spec_urban_rural":  spec["urban_rural"],
                "spec_politics":     spec["politics"],
                "spec_tech_comfort": spec["tech_comfort"],
                **answers,
            }

            out.write(json.dumps(record, ensure_ascii=True) + "\n")
            out.flush()
            completed.add(spec["id"])
            time.sleep(args.delay)

    print(f"\nDone. {len(completed)} respondents, {n_errors} errors.")
    write_csv(checkpoint, csv_out)
    print(f"\nOutputs:\n  {checkpoint}\n  {csv_out}")


if __name__ == "__main__":
    main()
