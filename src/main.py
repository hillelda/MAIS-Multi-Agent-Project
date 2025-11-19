from Agents import (
    QuestionGenerator,
    Summarizer,
    QAAgentRunner,
    Judge,
    JudgeEvaluationType,
    QuestionsOutputType,
    QAAgentEvaluationsOutputType,
)
import argparse
import os
import json
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

# --- Add/replace in src/main.py ---

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

FALLBACK_ANSWERS = {
    "not enough information in summary",
    "not enough information",
}

def process_pdf_to_markdown(file_path: str) -> str:
    """
    Convert a PDF file to markdown text.
    Tries PyPDFLoader first, falls back to UnstructuredPDFLoader.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        parts = []
        for i, doc in enumerate(documents, start=1):
            content = (doc.page_content or "").strip()
            if content:
                parts.append(f"# Page {i}\n\n{content}\n")
        md = "\n".join(parts).strip()
        if not md:
            raise ValueError("Empty content extracted by PyPDFLoader.")
        return md
    except Exception as primary_err:
        print(f"PyPDFLoader failed: {primary_err}. Trying UnstructuredPDFLoader...")
        try:
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
            parts = []
            for i, doc in enumerate(documents, start=1):
                content = (doc.page_content or "").strip()
                if content:
                    parts.append(f"# Section {i}\n\n{content}\n")
            md = "\n".join(parts).strip()
            if not md:
                raise ValueError("Empty content extracted by UnstructuredPDFLoader.")
            return md
        except Exception as fallback_err:
            raise RuntimeError(
                f"Both PDF loaders failed. Primary: {primary_err}; Fallback: {fallback_err}"
            ) from fallback_err

def load_file_content(file_path: str) -> str:
    """
    Load file content. If PDF, convert to markdown first.
    """
    if file_path.lower().endswith(".pdf"):
        print(f"Processing PDF file: {file_path}")
        content = process_pdf_to_markdown(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    if not content or not content.strip():
        raise ValueError("Loaded article content is empty.")
    return content

def run_summarization_workflow(
    query: str,
    article: str,
    max_iterations: int = 4,
    requests_per_second: float | None = None,
):
    """
    Run iterative summarization workflow with added QA failure gating.
    """
    if not article or not article.strip():
        raise ValueError("Article content is empty. Aborting workflow.")

    question_gen = QuestionGenerator(requests_per_second=requests_per_second)
    summarizer = Summarizer(requests_per_second=requests_per_second)
    qa_agent = QAAgentRunner(requests_per_second=requests_per_second)
    judge_agent = Judge(requests_per_second=requests_per_second)

    questions_output: QuestionsOutputType = question_gen.run(query=query, article=article)
    questions = questions_output.questions
    acu_questions = getattr(questions_output, "acu_questions", [])
    current_summary = ""
    sections_to_highlight: set[str] = set()

    workflow_result = {
        "query": query,
        "max_iterations": max_iterations,
        "iterations": [],
        "final_summary": "",
        "total_iterations": 0,
        "status": "",
        "questions": questions,
        "acu_questions": acu_questions,
    }

    for iteration in range(max_iterations):
        iteration_data = {
            "iteration_number": iteration + 1,
            "summary": "",
            "qa_evaluations": [],
            "qa_pairs": [],
            "judge": None,
            "correct_count_all": 0,
            "correct_count_acu": 0,
            "num_of_questions": len(questions),
            "sections_to_highlight": [],
            "sections_to_highlight_size": 0,
            "qa_failures_present": False,
        }

        # Summarize
        current_summary = summarizer.run(
            query=query, article=article, sections=list(sections_to_highlight)
        )
        iteration_data["summary"] = current_summary

        # QA (summary-only)
        qa_evaluations_struct: QAAgentEvaluationsOutputType = qa_agent.run(
            questions_output=questions_output, summary=current_summary
        )

        # Serialize evaluations
        iteration_data["qa_evaluations"] = [
            ev.model_dump() for ev in qa_evaluations_struct.evaluations
        ]

        # Prepare QA pairs for judge
        qa_pairs = []
        qa_failures = False
        for ev in qa_evaluations_struct.evaluations:
            ans_text = ev.qa.answer
            qa_pairs.append({"question": ev.qa.question, "answer": ans_text})
            if not ev.result:
                qa_failures = True
        iteration_data["qa_pairs"] = qa_pairs
        iteration_data["qa_failures_present"] = qa_failures

        # Judge (article-grounded)
        judge_eval: JudgeEvaluationType = judge_agent.run(
            article=article, summary=current_summary, qa_pairs=qa_pairs
        )
        iteration_data["judge"] = judge_eval.model_dump()
        iteration_data["sections_to_highlight"] = list(sections_to_highlight)
        iteration_data["sections_to_highlight_size"] = len(sections_to_highlight)

        total_correct = sum(1 for ev in judge_eval.evaluations if ev.result is True)
        acu_correct = sum(
            1
            for ev in judge_eval.evaluations
            if ev.result is True
            and isinstance(ev.qa.question, str)
            and ev.qa.question.strip().startswith("ACU.")
        )
        iteration_data["correct_count_all"] = total_correct
        iteration_data["correct_count_acu"] = acu_correct

        workflow_result["iterations"].append(iteration_data)

        # NEW stop condition: require judge true AND no QAAgent failures
        if judge_eval.judgment and not qa_failures:
            workflow_result["final_summary"] = current_summary
            workflow_result["total_iterations"] = iteration + 1
            workflow_result["status"] = "completed"
            return workflow_result
        else:
            # Accumulate sections across iterations using a set for O(1) membership checks
            new_sections = list(getattr(judge_eval, "sections_to_highlight", []))
            for section in new_sections:
                sections_to_highlight.add(section)
            print(f"sections_to_highlight: {sections_to_highlight}, iteration: {iteration}")

    workflow_result["final_summary"] = current_summary
    workflow_result["total_iterations"] = max_iterations
    workflow_result["status"] = "max_iterations_reached"
    return workflow_result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query-Focused Summarization Workflow")
    parser.add_argument('--file', type=str, required=True, help='Path to the article file (PDF or text)')
    parser.add_argument('--query', type=str, required=True, help='Query for summarization')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum number of iterations')
    parser.add_argument('--json_path', type=str, required=False, help='If set, write JSON output directly to this file path')
    parser.add_argument('--limiter', type=float, required=False, help='If set, requests per second for each agent rate limiter.')

    args = parser.parse_args()


    # Check if file exists
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        exit(1)

    # Load article content from file (with PDF processing if needed)
    try:
        article_content = load_file_content(args.file)
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)

    result = run_summarization_workflow(
        query=args.query,
        article=article_content,
    max_iterations=args.max_iterations,
    requests_per_second=args.limiter if 'limiter' in args and args.limiter is not None else None,
    )

    # Always output JSON now
    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path), exist_ok=True)
        with open(args.json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
