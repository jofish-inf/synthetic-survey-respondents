# synthetic-survey-respondents

Code and synthetic datasets for the study **"When Can a Language Model Take Your Survey?"** — an evaluation of LLM-generated synthetic survey respondents against a real 1,003-person survey of U.S. chatbot users.

We replicate one real survey with **six synthetic populations**: three models (Claude Haiku 4.5, GPT-4o-mini, Gemini 2.5 Flash) each run in two architectures (stateless single-prompt, and persistent multi-turn conversation). All six draw from an identical, census-weighted, seed-locked set of personas, so differences between them are attributable to model and architecture alone.

## What's here

### `data/synthetic/` — the six synthetic datasets
| File | Model | Architecture |
|------|-------|--------------|
| `claude_stateless_results.csv`    | Claude Haiku 4.5 | stateless |
| `claude_persistent_results.csv`   | Claude Haiku 4.5 | persistent |
| `chatgpt_stateless_results.csv`   | GPT-4o-mini | stateless |
| `chatgpt_persistent_results.csv`  | GPT-4o-mini | persistent |
| `gemini_stateless_results.csv`    | Gemini 2.5 Flash | stateless |
| `gemini_persistent_results.csv`   | Gemini 2.5 Flash | persistent |

Each row is one synthetic respondent: `id`, `model`, the `spec_*` persona columns (identical across runs for a given `id`), and the answer columns (`q3_*`, `q4_*`, `q5_*`, `q9_*`, `q11_*`, `q15_*`). ~1,000 respondents per file.

### `code/` — the generation scripts
One script per population. All share the same census-weighted persona generator (2023 ACS marginals, `seed=42`) and the targeted question subset (Q3, Q4, Q5, Q9, Q11, Q15); they differ only in the API client and the stateless/persistent prompting loop.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY / GEMINI_API_KEY
python3 code/simulate_survey_claude_stateless.py --n 5      # smoke test
python3 code/simulate_survey_claude_stateless.py           # full run -> *_results.csv
```
Each script checkpoints to a `.jsonl` and supports `--resume`. Delete the checkpoint between a `--n` test and a full run, or dedupe on `id` afterward.

## The real survey data

The ground-truth survey (`Understanding_technology_use_II.csv`) is **not included here**: it contains open-ended free-text responses that may carry identifying detail, and is released only under restricted access consistent with participant consent. Contact the authors for access.

## Citation

> Kaye, J. (2026). When Can a Language Model Take Your Survey? Inflection AI.

## License

- **Code** (`code/`): MIT — see `LICENSE`.
- **Synthetic datasets** (`data/`): CC BY 4.0 — see `LICENSE-data`.

The datasets are LLM-generated output from the Anthropic, OpenAI, and Google
APIs, provided for research use; those redistributing them should also consult
the respective providers' terms governing model output.
