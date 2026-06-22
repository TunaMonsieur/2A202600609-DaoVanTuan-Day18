from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)
        from datasets import Dataset
        from config import OPENAI_BASE_URL, make_chat_llm

        # answer_relevancy mặc định sinh 3 câu hỏi bằng n=3 (multiple candidates).
        # Gemini (OpenAI-compat) không hỗ trợ n>1 → đặt strictness=1 để dùng n=1.
        answer_relevancy.strictness = 1

        # RAGAS mặc định gọi OpenAI gpt-4o. Khi dùng Gemini qua endpoint
        # tương thích OpenAI, phải truyền LLM (judge) + embeddings tường minh.
        # Embeddings dùng model local (MiniLM, đã cache) — nhanh, không tốn API.
        eval_llm, eval_emb = None, None
        if OPENAI_BASE_URL:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            eval_llm = make_chat_llm(temperature=0)
            eval_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        # Gateway có latency biến thiên + giới hạn concurrency → giảm số worker
        # (bắn ít call đồng thời, đỡ nghẽn), tăng timeout và số lần retry.
        from ragas.run_config import RunConfig
        run_config = RunConfig(timeout=300, max_workers=4, max_retries=5)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                          llm=eval_llm, embeddings=eval_emb, run_config=run_config)
        df = result.to_pandas()

        def _val(row, key):
            try:
                v = row.get(key, 0.0)
                return float(v) if v == v else 0.0  # NaN guard
            except (TypeError, ValueError):
                return 0.0

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                faithfulness=_val(row, "faithfulness"),
                answer_relevancy=_val(row, "answer_relevancy"),
                context_precision=_val(row, "context_precision"),
                context_recall=_val(row, "context_recall"),
            )
            for _, row in df.iterrows()
        ]

        def _avg(attr):
            vals = [getattr(e, attr) for e in per_question]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "faithfulness": _avg("faithfulness"),
            "answer_relevancy": _avg("answer_relevancy"),
            "context_precision": _avg("context_precision"),
            "context_recall": _avg("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    scored = []
    for e in eval_results:
        metrics = {m: getattr(e, m) for m in metric_names}
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": e.question,
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "avg_score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
