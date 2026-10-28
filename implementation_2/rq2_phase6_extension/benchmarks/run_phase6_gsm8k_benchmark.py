#!/usr/bin/env python3
"""
GSM8K Mathematical Reasoning Benchmark for LLaMA 3 8B Phase 6 Cohort
Evaluates Base Model + 7 Clean Adapters + 8 Poisoned Adapters

Features:
- Canonical GSM8K grade-school arithmetic problems
- Consistent greedy decoding (temperature=0.0, max_tokens=60)
- Explicit resource deallocation between models to prevent memory pressure
- Records Accuracy, Answer Consistency with Base Model, and Latency
- Saves results to JSON and CSV
"""

import sys
import gc
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import llama_cpp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASE_MODEL_PATH = REPO_ROOT / "implementation" / "models.nosync" / "llama3" / "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
GGUF_DIR = REPO_ROOT / "implementation_2" / "rq2_phase6_extension" / "gguf_adapters"
OUTPUT_DIR = REPO_ROOT / "implementation_2" / "rq2_phase6_extension" / "results"

GSM8K_PROBLEMS = [
    {
        "id": "gsm8k_01",
        "question": "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast and bakes with 4. She sells remainder at $2 each. How much in dollars does she make daily?",
        "ground_truth": 18
    },
    {
        "id": "gsm8k_02",
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take to make 6 robes?",
        "ground_truth": 18
    },
    {
        "id": "gsm8k_03",
        "question": "Kylar went to the store with $40. He bought a book for $15 and 3 pens that cost $2 each. How much money does he have left in dollars?",
        "ground_truth": 19
    }
]

PROMPT_TEMPLATE = (
    "Solve step-by-step: {question} "
    "Conclude with your final number formatted as: #### <number>"
)

def extract_answer(text: str) -> Optional[int]:
    m = re.findall(r"####\s*(-?\d[\d,]*)", text)
    if m:
        try:
            return int(m[-1].replace(",", ""))
        except ValueError:
            pass
    m_box = re.findall(r"(\b\d[\d,]*\b)", text)
    if m_box:
        try:
            return int(m_box[-1].replace(",", ""))
        except ValueError:
            pass
    return None

def evaluate_model(llm: llama_cpp.Llama, name: str) -> Dict[str, Any]:
    results = []
    num_correct = 0
    t0 = time.time()
    
    for prob in GSM8K_PROBLEMS:
        prompt = PROMPT_TEMPLATE.format(question=prob["question"])
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a math tutor. Solve succinctly and end with #### <number>."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.0
        )
        gen_text = resp["choices"][0]["message"]["content"].strip()
        ans = extract_answer(gen_text)
        is_correct = (ans == prob["ground_truth"])
        if is_correct:
            num_correct += 1
            
        results.append({
            "problem_id": prob["id"],
            "ground_truth": prob["ground_truth"],
            "extracted_answer": ans,
            "is_correct": is_correct,
            "generated_text": gen_text[:80]
        })
        
    elapsed = round(time.time() - t0, 2)
    acc = (num_correct / len(GSM8K_PROBLEMS)) * 100.0
    logger.info(f"[{name}] Acc: {acc:.1f}% ({num_correct}/{len(GSM8K_PROBLEMS)}) in {elapsed:.1f}s")
    
    return {
        "adapter_name": name,
        "total_problems": len(GSM8K_PROBLEMS),
        "num_correct": num_correct,
        "accuracy_pct": acc,
        "runtime_seconds": elapsed,
        "problem_details": results
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = GGUF_DIR / "conversion_manifest.json"
    with open(manifest_path, "r") as f:
        adapters = json.load(f)
        
    all_evals = {}
    
    # 1. Base Model Unadapted
    logger.info("Evaluating LLaMA 3 8B Base Model...")
    llm_base = llama_cpp.Llama(
        model_path=str(BASE_MODEL_PATH),
        n_ctx=256,
        n_threads=4,
        n_gpu_layers=-1,
        verbose=False
    )
    base_eval = evaluate_model(llm_base, "base_model_unadapted")
    base_eval["category"] = "BASE"
    base_eval["base_answer_consistency_pct"] = 100.0
    all_evals["base_model_unadapted"] = base_eval
    del llm_base
    gc.collect()
    
    base_answers = {p["problem_id"]: p["extracted_answer"] for p in base_eval["problem_details"]}
    
    # 2. Adapters
    for item in adapters:
        name = item["adapter_name"]
        cat = item["category"]
        gguf_p = item["gguf_path"]
        
        logger.info(f"Evaluating: {name} ({cat.upper()})")
        llm = llama_cpp.Llama(
            model_path=str(BASE_MODEL_PATH),
            lora_path=gguf_p,
            lora_scale=1.0,
            n_ctx=256,
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False
        )
        res = evaluate_model(llm, name)
        res["category"] = cat
        del llm
        gc.collect()
        
        # Calculate answer consistency
        matching = sum(1 for p in res["problem_details"] if p["extracted_answer"] == base_answers.get(p["problem_id"]))
        res["base_answer_consistency_pct"] = round((matching / len(GSM8K_PROBLEMS)) * 100.0, 1)
        all_evals[name] = res
        
    # Save outputs
    json_path = OUTPUT_DIR / "phase6_gsm8k_results.json"
    with open(json_path, "w") as f:
        json.dump(all_evals, f, indent=2)
    logger.info(f"Saved GSM8K results to {json_path}")
    
    csv_path = OUTPUT_DIR / "phase6_gsm8k_results.csv"
    with open(csv_path, "w") as f:
        f.write("adapter_name,category,accuracy_pct,num_correct,total_problems,base_consistency_pct,runtime_sec\n")
        for name, r in all_evals.items():
            f.write(f"{name},{r['category']},{r['accuracy_pct']:.1f},{r['num_correct']},{r['total_problems']},{r['base_answer_consistency_pct']:.1f},{r['runtime_seconds']:.2f}\n")
    logger.info(f"Saved GSM8K CSV to {csv_path}")

if __name__ == "__main__":
    main()
