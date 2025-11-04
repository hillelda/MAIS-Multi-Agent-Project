# Query-Focused Summarization

Quick Start
```bash
pip install -r requirements.txt && \
echo GOOGLE_API_KEY=YOUR_KEY_HERE > .env && \
python src/main.py --file articles/The-Linear-Representation-Hypothesis.md \
                   --query "Summarize implications for interpretability" \
                   --max_iterations 3 \
                   --json_path results/quickstart.json
```

Example output (truncated):
```json
{
  "query": "Summarize implications for interpretability",
  "total_iterations": 2,
  "status": "completed",
  "final_summary": "... 200–250 word, query-focused summary ..."
}
```

A small, zero-shot, multi-agent system for query-focused summarization (QFS). It turns a paper + user query into a faithful, relevant summary through an iterative evaluate-and-refine loop.

### 1) What is this and what is it good for?

- **Purpose**: Produce summaries that answer a specific user query, not generic abstracts.
- **When to use**: Reading scientific/technical papers, extracting focused insights, quickly validating claims.
- **Why it works**: It separates adequacy (can the summary answer questions?) from accuracy (is it faithful to the article?), and refines until both are satisfied.

Key capabilities:
- Accepts PDF or text files
- Generates diagnostic questions to pressure-test coverage and precision (incl. ACU micro-facts)
- Self-assesses using the current summary only; checks truthfulness against the full article
- Iteratively focuses on missing topics until done or the iteration limit is reached

### Acknowledgements

This was a collaborative project built by @avrymi-asraf Avreymi Asraf and Hillel Darshan @hillel.darshan.

### 2) How does it work?

Agents coordinated by `src/main.py`:
1. **QuestionGenerator** (Gemini 2.5 Pro): creates 5 diagnostic questions + optional ACUs (one-time).
2. **Summarizer** (Gemini 2.5 Flash‑Lite): produces a query-focused summary; later iterations emphasize specific sections.
3. **QAAgentRunner** (Flash‑Lite): answers questions using only the current summary; flags insufficiency.
4. **Judge** (Flash‑Lite, ReAct-style): compares QA answers and the summary to the full article; either emits concrete “sections_to_highlight” or approves the final summary.

Stop condition: Judge approves AND no QA failures; otherwise iterate with targeted sections.

Notes:
- Built with LangChain Expression Language (LCEL) + `langchain-google-genai`
- PDF loading: `PyPDFLoader` with `UnstructuredPDFLoader` fallback
- Optional per-agent rate limiting via `--limiter`

Theoretical context (MAIS framework): zero-shot, hybrid-model approach (Pro for questions; Flash‑Lite for the loop), ReAct-style judge control, and “Topics of Mismatch” to drive revisions.

### 3) How should I use it?

Install:
```bash
pip install -r requirements.txt
```

Configure credentials (recommended `.env` in project root):
```env
GOOGLE_API_KEY=your_api_key_here
```

Run on a markdown article:
```bash
python src/main.py --file articles/The-Linear-Representation-Hypothesis.md \
                   --query "Summarize implications for interpretability" \
                   --max_iterations 5
```

Run and save structured JSON:
```bash
python src/main.py --file articles/The-Linear-Representation-Hypothesis.md \
                   --query "Summarize implications for interpretability" \
                   --max_iterations 5 \
                   --json_path results/output.json
```

PDFs work out of the box; they are converted to markdown internally.

CLI options:
- `--file` (required): PDF or text path
- `--query` (required): your focus
- `--max_iterations` (default 5)
- `--json_path` (optional): write JSON to a file instead of stdout
- `--limiter` (optional): requests per second per agent

Demo:
```bash
./run_demo.sh
```

Output (JSON):
- Top-level: `query`, `max_iterations`, `total_iterations`, `status`, `questions`, `acu_questions`, `final_summary`
- Per-iteration: `summary`, `qa_pairs`, `qa_evaluations`, `judge`, `sections_to_highlight`, `correct_count_all`, `correct_count_acu`, `qa_failures_present`

Project layout:
- `src/` – agents (`Agents.py`) and orchestrator (`main.py`)
- `articles/`, `articals/` – sample articles
- `results/` – saved runs
- `scripts/` – utilities, plus `run_demo.sh`

Troubleshooting:
- File not found / empty: check `--file` path; for PDFs the loader prints detailed errors
- Auth: ensure `GOOGLE_API_KEY` is set or present in `.env`
- Rate limit/timeouts: lower `--limiter` or `--max_iterations`


License: MIT (or project-appropriate).

