import os
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import json
import sys
from pathlib import Path
from datetime import date
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from finetune.verdict import _parse_verdict

project_root   = Path(__file__).parent.parent
finetuned_path = str(project_root / "outputs" / "wiki-finetuned")


def _judge_verdict(judge: ChatOllama, question: str, expected: str, response: str) -> str:
    prompt = (
        f"You are a strict examiner evaluating an AI Engineering exam.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Model response: {response}\n\n"
        f"Score CORRECT only if the model response:\n"
        f"1. Directly answers the specific question asked\n"
        f"2. Contains the key technical facts from the expected answer\n"
        f"3. Does not contradict the expected answer\n\n"
        f"Score INCORRECT if the response is vague, generic, off-topic,\n"
        f"or missing the specific technical details in the expected answer.\n\n"
        f"Reply only: CORRECT or INCORRECT"
    )
    return judge.invoke(prompt).content.strip().upper()


def score_rag(test_pairs: list) -> tuple[float | None, list[dict]]:
    """Returns (score_or_None, per_question_records).

    per_question_records entries: {question, expected, response, verdict,
    retrieved_slugs, skipped}
    """
    import requests as _requests
    import re as _re
    from pipeline.query import run_query, search_wiki

    log_path = project_root / "vault" / "log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# Query Log\n", encoding="utf-8")

    try:
        r = _requests.get(
            f"{config.VAULT_API}/", headers=config.HEADERS,
            verify=config.VERIFY_SSL, timeout=5,
        )
        if r.status_code != 200:
            print("ERROR: Obsidian not reachable. Skipping RAG.")
            return None, []
    except Exception:
        print("ERROR: Obsidian must be open to score RAG. Skipping RAG.")
        return None, []

    judge   = ChatOllama(model=config.MODEL_NAME)
    correct = 0
    scored  = 0
    records: list[dict] = []

    for i, pair in enumerate(test_pairs):
        rec: dict = {
            "question": pair["question"],
            "expected": pair["answer"],
            "response": "",
            "verdict": "SKIPPED",
            "retrieved_slugs": [],
            "skipped": False,
        }
        try:
            # Log which articles search_wiki retrieves for this question
            raw_retrieval = search_wiki.invoke(pair["question"])
            retrieved = _re.findall(r'=== ([^=\n]+?) ===', raw_retrieval)
            rec["retrieved_slugs"] = [s.strip() for s in retrieved]
            print(f"  [{i+1}/{len(test_pairs)}] retrieved: {rec['retrieved_slugs']}")

            response = run_query(pair["question"])
            if not response or "not reachable" in str(response).lower():
                print(f"  [{i+1}/{len(test_pairs)}] SKIPPED (retrieval failed)")
                rec["skipped"] = True
                records.append(rec)
                continue

            rec["response"] = response
            scored += 1
            verdict = _judge_verdict(judge, pair["question"], pair["answer"], response)
            rec["verdict"] = verdict
            if _parse_verdict(verdict):
                correct += 1
            print(f"  [{i+1}/{len(test_pairs)}] {verdict}")
        except Exception as e:
            print(f"  [{i+1}/{len(test_pairs)}] ERROR: {e}")
            rec["verdict"] = f"ERROR: {e}"
        records.append(rec)

    if scored == 0:
        print("  No pairs scored — all retrievals failed.")
        return None, records
    return (correct / scored) * 100, records


def score_ollama_detailed(model_name: str, test_pairs: list) -> tuple[float, list[dict]]:
    llm   = ChatOllama(model=model_name)
    judge = ChatOllama(model=config.MODEL_NAME)
    correct = 0
    records: list[dict] = []
    for i, pair in enumerate(test_pairs):
        rec = {"question": pair["question"], "expected": pair["answer"],
               "response": "", "verdict": "ERROR"}
        try:
            response = llm.invoke(pair["question"]).content
            verdict = _judge_verdict(judge, pair["question"], pair["answer"], response)
            rec["response"] = response
            rec["verdict"] = verdict
            if _parse_verdict(verdict):
                correct += 1
            print(f"  [{i+1}/{len(test_pairs)}] {verdict}")
        except Exception as e:
            print(f"  [{i+1}/{len(test_pairs)}] ERROR: {e}")
            rec["verdict"] = f"ERROR: {e}"
        records.append(rec)
    return (correct / len(test_pairs)) * 100, records


def score_finetuned_detailed(test_pairs: list) -> tuple[float, list[dict]]:
    import torch
    from unsloth import FastLanguageModel

    print(f"  Loading fine-tuned model from {finetuned_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = finetuned_path,
        max_seq_length = 512,
        load_in_4bit   = True,
        dtype          = None,
    )
    FastLanguageModel.for_inference(model)

    judge   = ChatOllama(model=config.MODEL_NAME)
    correct = 0
    records: list[dict] = []

    for i, pair in enumerate(test_pairs):
        rec = {"question": pair["question"], "expected": pair["answer"],
               "response": "", "verdict": "ERROR"}
        try:
            prompt = (
                f"<|im_start|>user\n{pair['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = tokenizer(
                prompt,
                return_tensors = "pt",
                truncation     = True,
                max_length     = 400,
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = 200,
                    temperature    = 0.1,
                    do_sample      = False,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens = True,
            )
            rec["response"] = response
            verdict = _judge_verdict(judge, pair["question"], pair["answer"], response)
            rec["verdict"] = verdict
            if _parse_verdict(verdict):
                correct += 1
            print(f"  [{i+1}/{len(test_pairs)}] {verdict}")
        except Exception as e:
            print(f"  [{i+1}/{len(test_pairs)}] ERROR: {e}")
            rec["verdict"] = f"ERROR: {e}"
        records.append(rec)

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return (correct / len(test_pairs)) * 100, records


def evaluate_all():
    test_pairs = json.loads((project_root / "finetune" / "test.json").read_text())
    print(f"Evaluating on {len(test_pairs)} held-out pairs...\n")

    print("Scoring base Qwen2.5:7B via Ollama...")
    base_score, base_records = score_ollama_detailed("qwen2.5:7b", test_pairs)
    print(f"Base score: {base_score:.1f}%\n")

    print("Scoring fine-tuned model directly via Unsloth...")
    ft_score, ft_records = score_finetuned_detailed(test_pairs)
    print(f"Fine-tuned score: {ft_score:.1f}%\n")

    print("Scoring RAG over wiki (requires Obsidian open)...")
    rag_score, rag_records = score_rag(test_pairs)
    rag_label = f"{rag_score:.1f}%" if rag_score is not None else "skipped (Obsidian offline)"
    print(f"RAG score: {rag_label}\n")

    print("=== THREE-WAY BENCHMARK ===")
    print(f"{'base_qwen':15s}: {base_score:.1f}%")
    print(f"{'finetuned':15s}: {ft_score:.1f}%")
    print(f"{'rag_wiki':15s}: {rag_label}")

    # Persist per-question results
    per_question = []
    for idx, pair in enumerate(test_pairs):
        per_question.append({
            "idx":      idx,
            "question": pair["question"],
            "expected": pair["answer"],
            "base":     base_records[idx] if idx < len(base_records) else {},
            "ft":       ft_records[idx]   if idx < len(ft_records)   else {},
            "rag":      rag_records[idx]  if idx < len(rag_records)  else {"skipped": True},
        })
    results_json = project_root / "finetune" / "results-2026-07.json"
    results_json.write_text(json.dumps(per_question, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"Per-question results saved to {results_json}")

    rag_cell = f"{rag_score:.1f}%" if rag_score is not None else "skipped"
    superseded = """
## Superseded results — 2026-06-02 (INVALID)

| Method | Score (June 2) | Why invalid |
|---|---|---|
| Base Qwen2.5:7B | 69.0% | Valid — unchanged path |
| Fine-tuned (wiki-qwen) | 52.4% | Invalid — ChatML template missing at inference (INVESTIGATION.md §Check 3) |
| RAG over wiki | 35.7% | Invalid — retrieval entirely broken; LLM answered from memory (INVESTIGATION.md §Check 4) |
"""

    report = f"""# Evaluation Results — {date.today()} (July 2026 re-run)

Fixes applied since June 2026 run — see INVESTIGATION.md:
- FIX 1: ChatML template now applied in fine-tuned inference path
- FIX 2: RAG retrieval repaired after week-folder restructure

| Method | Score |
|---|---|
| Base Qwen2.5:7B | {base_score:.1f}% |
| Fine-tuned (wiki-qwen, w/ template fix) | {ft_score:.1f}% |
| RAG over wiki (repaired retrieval) | {rag_cell} |

Test set: {len(test_pairs)} held-out pairs, never seen during training.
Per-question detail: finetune/results-2026-07.json

## Connection to LegalRisk-LLM
| | LegalRisk-LLM | This Project (July 2026) |
|---|---|---|
| QLoRA | 45.8% | {ft_score:.1f}% |
| RAG | 55.4% | {rag_cell} |

*LegalRisk-LLM benchmark on CUAD, Llama-3.2-3B-Instruct, n=107 — github.com/navinsreyas/LLM-Finetune-LegalRisk*
{superseded}"""

    results_path = project_root / "finetune" / "evaluation-results.md"
    results_path.write_text(report, encoding="utf-8")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    evaluate_all()
