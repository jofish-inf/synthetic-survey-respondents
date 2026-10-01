#!/usr/bin/env python3
"""
simulate_survey_claude_persistent.py

Same targeted question set (Q3, Q4, Q5, Q9, Q11, Q15), using a PERSISTENT
CONVERSATION per respondent with Anthropic's Claude: each question is a new turn in the same chat thread, so the
model's answers to Q4 are in context when it answers Q5, Q9, etc.

This tests whether persona consistency improves when the model can see
its own prior answers — vs. the stateless version where each question
is answered cold.

The system prompt establishes the persona once. Then questions are asked
one at a time as user turns. Each answer is appended to the history
before the next question is sent.

Output columns are identical to the other persistent scripts so the two
CSVs can be compared directly.

Requirements:
    pip install anthropic tqdm

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python simulate_survey_claude_persistent.py --n 5    # test
    python simulate_survey_claude_persistent.py          # full 1,000
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

parser = argparse.ArgumentParser(
    description="Simulate targeted survey with Claude — persistent persona per respondent"
)
parser.add_argument("--n",       type=int,   default=1000)
parser.add_argument("--api-key", type=str,   default=None)
parser.add_argument("--model",   type=str,   default="claude-haiku-4-5")
parser.add_argument("--output",  type=str,   default="claude_persistent_results")
parser.add_argument("--delay",   type=float, default=0.25,
                    help="Seconds between API calls (not between respondents).")
parser.add_argument("--resume",  action="store_true")
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
# Census-weighted demographics (identical seed + distributions to other scripts)
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
# Prompts
# Each question is sent as a separate user turn. We ask for JSON for each
# turn so we can parse incrementally and recover from partial failures.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are roleplaying as a specific survey respondent for an entire survey session.
Stay completely in character throughout. Your persona is fixed — do not change it.
Answer each question as this person would: briefly and naturally, the way a real
person fills out an online survey. If something doesn't apply to you, say so briefly.
When asked for JSON, return ONLY valid JSON — no markdown, no backticks, no commentary.\
"""

# Each entry: (question_key, user_turn_text, expected_json_keys)
# The model builds up conversation history across all turns.

def make_turns(spec: dict) -> list:
    desc = spec_to_description(spec)
    devices_str   = ", ".join(DEVICES_Q3)
    chatbots_str  = ", ".join(CHATBOTS)
    freq_str      = ", ".join(USAGE_FREQ_OPTIONS)
    roles_str     = ", ".join(CHATBOT_ROLES)

    return [
        # Turn 0: establish persona
        {
            "key": "_persona",
            "prompt": (
                f"You are: {desc}\n\n"
                "You're about to take a short survey about technology and chatbot use. "
                "I'll ask you questions one at a time. Stay in character throughout.\n\n"
                "To confirm you understand your persona, reply with a single JSON object: "
                '{"ready": true, "persona_summary": "<one sentence in your own voice>"}'
            ),
            "parse_key": None,  # just confirms, not stored as survey data
        },
        # Turn 1: Q3
        {
            "key": "q3_devices_used_last_week",
            "prompt": (
                "Q3: Which of the following have you used in the last week? "
                f"Choose any that apply from this list: {devices_str}\n\n"
                'Reply with JSON: {"q3_devices_used_last_week": ["item", ...]}'
            ),
            "parse_key": "q3_devices_used_last_week",
        },
        # Turn 2: Q4
        {
            "key": "q4_chatbots_used",
            "prompt": (
                "Q4: Which of the following chatbots have you used or do you use regularly? "
                f"Choose up to 10 from: {chatbots_str}. "
                'Or say "I don\'t really use AI chatbots" if that applies.\n\n'
                'Reply with JSON: {"q4_chatbots_used": ["chatbot", ...]}'
            ),
            "parse_key": "q4_chatbots_used",
        },
        # Turn 3: Q5 — references Q4 answer implicitly via conversation history
        {
            "key": "q5_usage_frequency",
            "prompt": (
                "Q5: For each chatbot you've ever heard of or tried, how often do you use it? "
                f"Use only these options: {freq_str}. "
                "Only include chatbots you have some awareness of — skip ones you've never heard of.\n\n"
                'Reply with JSON: {"q5_usage_frequency": {"ChatGPT": "...", "Claude": "...", ...}}'
            ),
            "parse_key": "q5_usage_frequency",
        },
        # Turn 4: Q9 — can reference Q4/Q5 for consistency
        {
            "key": "q9_stopped_using",
            "prompt": (
                "Q9: Have you ever decided to stop using a particular chatbot? "
                "If yes, which one and why? If no, just say so briefly.\n\n"
                'Reply with JSON: {"q9_stopped_using": "<your answer>"}'
            ),
            "parse_key": "q9_stopped_using",
        },
        # Turn 5: Q11
        {
            "key": "q11_wish_chatbot_could_do",
            "prompt": (
                "Q11: Is there something you wish a chatbot could help you with, "
                "but haven't found one that does it well? "
                "If nothing comes to mind, say so.\n\n"
                'Reply with JSON: {"q11_wish_chatbot_could_do": "<your answer>"}'
            ),
            "parse_key": "q11_wish_chatbot_could_do",
        },
        # Turn 6: Q15
        {
            "key": "q15_future_roles",
            "prompt": (
                "Q15: In the future, chatbots will be able to do lots of things better. "
                f"Pick EXACTLY 5 from this list that would be most valuable to you: {roles_str}\n\n"
                'Reply with JSON: {"q15_future_roles": ["Role1", "Role2", "Role3", "Role4", "Role5"]}'
            ),
            "parse_key": "q15_future_roles",
        },
    ]


# ---------------------------------------------------------------------------
# JSON parsing (same truncation recovery as chatgpt script)
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
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


# ---------------------------------------------------------------------------
# Single API call — sends the full conversation history
# ---------------------------------------------------------------------------

def call_turn(history: list, respondent_id: int, turn_num: int,
              max_retries: int = 5) -> tuple[str, dict]:
    """
    Send the conversation history to Claude.
    history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    The system prompt is passed separately via the system= param.
    Returns (raw_text, parsed_dict).
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=1500,
                temperature=1.0,
                system=SYSTEM_PROMPT,
                messages=history,
            )
            raw = response.content[0].text
            parsed = extract_json(raw)
            return raw, parsed
        except json.JSONDecodeError as e:
            print(f"\n  [id={respondent_id} turn={turn_num}] JSON error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n  [id={respondent_id} turn={turn_num}] Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"\n  [id={respondent_id} turn={turn_num}] API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return "", {}


# ---------------------------------------------------------------------------
# Run one respondent through all turns
# ---------------------------------------------------------------------------

def run_respondent(spec: dict) -> dict:
    """
    Conducts a full persistent-persona survey session.
    Returns a flat dict of all question answers.
    """
    turns = make_turns(spec)
    answers = {}

    # Claude: system prompt is passed separately to messages.create(),
    # so the message history starts empty and uses user/assistant roles.
    history = []

    for turn_num, turn in enumerate(turns):
        history.append({"role": "user", "content": turn["prompt"]})

        raw, parsed = call_turn(history, spec["id"], turn_num)

        if not parsed:
            # On failure: append a placeholder so history stays coherent
            placeholder = '{"error": "no response"}'
            history.append({"role": "assistant", "content": placeholder})
            continue

        # Append the actual response to history so next turn sees it
        history.append({"role": "assistant", "content": raw})

        # Store answer if this turn has survey data
        if turn["parse_key"] and turn["parse_key"] in parsed:
            answers[turn["key"]] = parsed[turn["parse_key"]]
        elif turn["parse_key"]:
            # Model returned JSON but with unexpected keys — store whatever it gave us
            answers[turn["key"]] = parsed

        time.sleep(args.delay)

    return answers


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
    random.seed(42)  # same personas as all other scripts

    checkpoint = f"{args.output}.jsonl"
    csv_out    = f"{args.output}.csv"

    all_specs  = [generate_persona_spec(i) for i in range(args.n)]
    completed  = load_completed_ids(checkpoint) if args.resume else set()

    if completed:
        print(f"Resuming: {len(completed)} already done.")

    n_errors = 0

    print(f"Model:        {args.model}")
    print(f"Respondents:  {args.n}")
    print(f"Questions:    Q3, Q4, Q5, Q9, Q11, Q15  (7 turns per respondent incl. persona setup)")
    print(f"Mode:         PERSISTENT — full conversation history per respondent")
    print(f"Checkpoint:   {checkpoint}")
    print(f"CSV output:   {csv_out}")
    print(f"Est. cost:    ~${args.n * 7 * 0.0008:.2f} (claude-haiku-4-5 at 7 turns × ~1k tokens each)\n")

    with open(checkpoint, "a", encoding="utf-8") as out:
        for spec in tqdm(all_specs, desc="Respondents"):
            if spec["id"] in completed:
                continue

            answers = run_respondent(spec)

            if not answers:
                n_errors += 1
                tqdm.write(f"  WARNING: no answers for id={spec['id']} (errors: {n_errors})")

            record = {
                "id":    spec["id"],
                "model": args.model + "_persistent",
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

    print(f"\nDone. {len(completed)} respondents, {n_errors} errors.")
    write_csv(checkpoint, csv_out)
    print(f"\nOutputs:\n  {checkpoint}\n  {csv_out}")


if __name__ == "__main__":
    main()
