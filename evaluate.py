"""Evaluate a checkpoint on in-distribution and OOD math benchmarks,
scoring the SAME completions under every grader.

Twist #4: if the ranking of two checkpoints flips depending on which extractor
grades them, the "which GRPO variant wins" question is grader-dependent -- a
methodological finding in its own right. Generating once and re-scoring N ways
makes that comparison free.

Twist #5: GSM8K test is in-distribution; gsm-plus and MATH-500 measure whether
the in-distribution winner also generalizes.

Usage:
    python evaluate.py --model runs/g8_seed0 --benchmarks gsm8k
    python evaluate.py --model runs/g8_seed0 --benchmarks gsm8k gsm_plus math500 \
        --out results/g8_seed0.json
    python evaluate.py --selftest
"""

import argparse
import json
import os

from extractors import EXTRACTORS, equal, gsm8k_gold
from grpo_train import SYSTEM_PROMPT

# name -> (hf dataset args, question field, gold-answer fn, equivalence mode)
BENCHMARKS = {
    "gsm8k": (("openai/gsm8k", "main", "test"), "question",
              lambda ex: gsm8k_gold(ex["answer"]), "numeric"),
    "gsm_plus": (("qintongli/GSM-Plus", None, "testmini"), "question",
                 lambda ex: str(ex["answer"]).replace(",", "").strip(), "numeric"),
    "math500": (("HuggingFaceH4/MATH-500", None, "test"), "problem",
                lambda ex: str(ex["answer"]).strip(), "symbolic"),
}


def load_benchmark(name, limit=None):
    from datasets import load_dataset

    (path, config, split), qfield, gold_fn, mode = BENCHMARKS[name]
    ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [(ex[qfield], gold_fn(ex)) for ex in ds], mode


def generate(model_path, questions, max_new_tokens=512, batch_size=16):
    """Greedy decoding: eval must be deterministic or seed-variance claims (twist #6)
    are confounded by sampling noise."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = tok(prompts[i:i + batch_size], return_tensors="pt",
                    padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                     skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def score(completions, golds, mode):
    """Score one completion set under EVERY grader. Returns {grader: accuracy}."""
    return {
        name: sum(equal(fn(c), g, mode) for c, g in zip(completions, golds)) / len(golds)
        for name, fn in EXTRACTORS.items()
    }


def selftest():
    comps = ["steps...\n#### 10", "the answer is 10", "no idea"]
    golds = ["10", "10", "10"]
    s = score(comps, golds, "numeric")
    assert abs(s["strict"] - 1 / 3) < 1e-9, s        # only the formatted one
    assert abs(s["last_number"] - 2 / 3) < 1e-9, s   # lenient credits the second
    assert abs(s["boxed"] - 1 / 3) < 1e-9, s
    print("evaluate selftest: all checks passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--model", help="checkpoint dir or HF model name")
    p.add_argument("--benchmarks", nargs="+", default=["gsm8k"], choices=list(BENCHMARKS))
    p.add_argument("--limit", type=int, default=None, help="cap examples per benchmark")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default=None, help="JSON output path (default: <model>/eval.json)")
    p.add_argument("--save_completions", action="store_true",
                   help="also dump raw completions for later re-grading")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.model:
        p.error("--model is required")

    results = {"model": args.model, "benchmarks": {}}
    for bench in args.benchmarks:
        print(f"[{bench}] loading...")
        pairs, mode = load_benchmark(bench, args.limit)
        questions, golds = zip(*pairs)
        print(f"[{bench}] generating {len(questions)} completions...")
        comps = generate(args.model, list(questions), args.max_new_tokens, args.batch_size)
        accs = score(comps, list(golds), mode)
        results["benchmarks"][bench] = {"n": len(golds), "accuracy_by_grader": accs}
        print(f"[{bench}] " + "  ".join(f"{k}={v:.4f}" for k, v in accs.items()))
        if args.save_completions:
            cpath = (args.out or os.path.join(args.model, "eval.json")).replace(
                ".json", f"_{bench}_completions.jsonl")
            with open(cpath, "w") as f:
                for q, g, c in zip(questions, golds, comps):
                    f.write(json.dumps({"question": q, "gold": g, "completion": c}) + "\n")

    out = args.out or os.path.join(args.model, "eval.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
