"""Instrumented GRPO: the experiment driver for the paper.

`grpo_baseline.py` stays as the known-good reference. This file adds everything
the paper needs, WITHOUT subclassing GRPOTrainer (TRL's advantage computation
lives in a private method that changes shape between minor versions; copying it
would make this repo break on every TRL bump). All instrumentation rides in the
reward function, which is stable public API and sees the whole generation batch.

What this adds over the baseline
--------------------------------
  #1 matched-budget group-size sweep   --num_generations G, --rollouts_per_step N
       Total rollouts per optimizer step is held at N regardless of G, so a
       sweep over G is a *fixed-compute* comparison. Distinct prompts per step
       is then N/G -- that's the tradeoff the paper measures.
  #2 gradient-signal accounting        --stats_log (JSONL, one line per gen batch)
       Fraction of groups that are all-correct or all-wrong. Those have zero
       reward variance -> zero advantage -> ZERO GRADIENT. That wasted-compute
       fraction is the paper's headline measurement.
  #3 difficulty weighting              --difficulty_weight
       Scale each group's rewards by 4*p*(1-p) (peaks at pass rate 0.5).
       Requires --scale_rewards false: with std-scaling a constant group
       multiplier cancels out exactly, so it would be a silent no-op.
  #4 verifier fragility                --extractor {strict,last_number,boxed}
       Train under one grader; eval.py re-scores under all of them.
  #6 seed variance                     --seed

Usage
-----
    python grpo_train.py --selftest                 # no GPU needed, ~1s
    python grpo_train.py --max_steps 20 --report_to none --run_name smoke
    bash sweep.sh                                   # the full grid
"""

import argparse
import json
import os
import time
from collections import OrderedDict

from extractors import EXTRACTORS, equal, gsm8k_gold

# torch/transformers/trl are imported inside main() so --selftest runs on a
# laptop with none of them installed.

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "then give the final numeric answer on its own line as:\n#### <answer>"
)


# ----------------------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------------------
def build_dataset(split, tokenizer, max_prompt_length=384):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)

    def _map(ex):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "gold": gsm8k_gold(ex["answer"]),
            "prompt_len": len(tokenizer(prompt)["input_ids"]),
        }

    ds = ds.map(_map, remove_columns=ds.column_names)
    # Filter over-long prompts ourselves rather than relying on GRPOConfig's
    # truncation. GSM8K prompts are short; this drops very few examples.
    ds = ds.filter(lambda ex: ex["prompt_len"] <= max_prompt_length)
    return ds.remove_columns(["prompt_len"])


# ----------------------------------------------------------------------------------
# 2. The instrumented reward function
# ----------------------------------------------------------------------------------
class VerifiableReward:
    """Correctness (+ small format bonus), plus the group-level accounting that
    the paper is actually about.

    Folded into ONE reward callable rather than two, because difficulty weighting
    has to scale the *total* group reward, and TRL sums reward functions after
    the fact -- weighting only one of two summands would be wrong.
    """

    def __init__(
        self,
        extractor="strict",
        tokenizer=None,
        stats_log=None,
        format_bonus=0.1,
        difficulty_weight=False,
        equal_mode="numeric",
    ):
        self.__name__ = "verifiable_reward"  # TRL reads .__name__ for its logs
        self.extract = EXTRACTORS[extractor]
        self.tokenizer = tokenizer
        self.stats_log = stats_log
        self.format_bonus = format_bonus
        self.difficulty_weight = difficulty_weight
        self.equal_mode = equal_mode
        self.call = 0
        self.t0 = time.time()
        if stats_log:
            os.makedirs(os.path.dirname(os.path.abspath(stats_log)), exist_ok=True)
            open(stats_log, "w").close()

    def __call__(self, completions, prompts, gold, **kwargs):
        correct = [
            1.0 if equal(self.extract(c), g, self.equal_mode) else 0.0
            for c, g in zip(completions, gold)
        ]
        formatted = [1.0 if EXTRACTORS["strict"](c) else 0.0 for c in completions]
        rewards = [c + self.format_bonus * f for c, f in zip(correct, formatted)]

        # Group by prompt string rather than assuming TRL lays groups out
        # contiguously -- that layout is an implementation detail, this isn't.
        # ponytail: two identical prompts in one batch would merge. GSM8K has no
        # exact duplicates, and the stats logger reports group sizes so a merge
        # would show up as an off-size group rather than corrupting silently.
        groups = OrderedDict()
        for i, p in enumerate(prompts):
            groups.setdefault(p, []).append(i)

        pass_rates = []
        for idxs in groups.values():
            p_rate = sum(correct[i] for i in idxs) / len(idxs)
            pass_rates.append((p_rate, len(idxs)))
            if self.difficulty_weight:
                # 4p(1-p): 1.0 at p=0.5 (maximally informative), 0.0 at p in {0,1}
                # (already-zero advantage, so this costs nothing there).
                w = 4.0 * p_rate * (1.0 - p_rate)
                for i in idxs:
                    rewards[i] *= w

        self._log(pass_rates, correct, rewards, completions)
        return rewards

    def _log(self, pass_rates, correct, rewards, completions):
        if not self.stats_log:
            return
        n = len(pass_rates)
        all_wrong = sum(1 for p, _ in pass_rates if p == 0.0)
        all_right = sum(1 for p, _ in pass_rates if p == 1.0)
        # A group with zero reward variance contributes zero advantage to every
        # one of its members -- its rollouts were generated for nothing.
        degenerate = all_wrong + all_right
        rollouts = sum(sz for _, sz in pass_rates)
        wasted = sum(sz for (p, sz) in pass_rates if p in (0.0, 1.0))

        if self.tokenizer is not None:
            lens = [len(self.tokenizer(c)["input_ids"]) for c in completions]
        else:
            lens = [len(c.split()) for c in completions]

        rec = {
            "call": self.call,
            "elapsed_s": round(time.time() - self.t0, 2),
            "n_groups": n,
            "n_rollouts": rollouts,
            "group_sizes": sorted({sz for _, sz in pass_rates}),
            "frac_groups_all_wrong": all_wrong / n,
            "frac_groups_all_correct": all_right / n,
            "frac_groups_degenerate": degenerate / n,
            # The headline number: share of generated rollouts that produced no gradient.
            "frac_rollouts_wasted": wasted / rollouts,
            "accuracy": sum(correct) / len(correct),
            "mean_reward": sum(rewards) / len(rewards),
            "mean_completion_tokens": sum(lens) / len(lens),
            "pass_rates": [p for p, _ in pass_rates],
        }
        with open(self.stats_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.call += 1


# ----------------------------------------------------------------------------------
# 3. Batch geometry
# ----------------------------------------------------------------------------------
def geometry(G, N, requested_bs):
    """(per_device_batch_size, grad_accum) for group size G under a fixed rollout
    budget N per optimizer step.

    TRL requires num_generations to divide the (global) per-step batch size, i.e.
    on one GPU: bs % G == 0. So bs must be a multiple of G that also divides N;
    pick the largest one not exceeding the requested size -- but at least G, so
    large-G runs bump the batch up instead of crashing inside TRL.
    """
    if N % G:
        raise SystemExit(f"rollouts_per_step ({N}) must be divisible by group size ({G})")
    cap = max(requested_bs, G)
    bs = max(b for b in range(G, N + 1, G) if N % b == 0 and b <= cap)
    return bs, N // bs


# ----------------------------------------------------------------------------------
# 4. Self-check (no GPU, no network)
# ----------------------------------------------------------------------------------
def selftest():
    import tempfile

    log = os.path.join(tempfile.mkdtemp(), "stats.jsonl")
    G = 4
    # Two prompts x 4 rollouts: prompt A is 2/4 (informative), prompt B is 0/4 (wasted).
    prompts = ["A"] * G + ["B"] * G
    comps = ["#### 5", "#### 5", "#### 9", "#### 9"] + ["#### 1"] * G
    gold = ["5"] * G + ["7"] * G

    r = VerifiableReward(stats_log=log)
    out = r(comps, prompts, gold)
    assert out[:2] == [1.1, 1.1] and out[2:4] == [0.1, 0.1], out
    rec = json.loads(open(log).read().strip())
    assert rec["n_groups"] == 2 and rec["group_sizes"] == [G]
    assert rec["frac_groups_all_wrong"] == 0.5
    assert rec["frac_groups_degenerate"] == 0.5
    assert rec["frac_rollouts_wasted"] == 0.5      # B's 4 rollouts bought nothing
    assert abs(rec["accuracy"] - 0.25) < 1e-9

    # Difficulty weighting: p=0.5 group keeps its reward (w=1), p=0 group is zeroed.
    rw = VerifiableReward(difficulty_weight=True)
    out = rw(comps, prompts, gold)
    assert out[:4] == [1.1, 1.1, 0.1, 0.1], out     # w = 4*0.5*0.5 = 1.0
    assert out[4:] == [0.0] * G, out                # w = 0.0

    # Lenient grader credits an answer the strict one rejects.
    rl = VerifiableReward(extractor="last_number")
    assert rl(["the answer is 5"], ["A"], ["5"]) == [1.0]   # correct, no format bonus
    assert VerifiableReward()(["the answer is 5"], ["A"], ["5"]) == [0.0]

    # Geometry: bs is always a multiple of G (TRL's constraint), budget always met.
    for G in (2, 4, 8, 16, 32):
        bs, accum = geometry(G, 32, 8)
        assert bs % G == 0 and bs * accum == 32, (G, bs, accum)
    assert geometry(2, 32, 8) == (8, 4)    # small G honors the requested batch size
    assert geometry(16, 32, 8) == (16, 2)  # large G bumps the batch instead of crashing
    assert geometry(32, 32, 8) == (32, 1)
    try:
        geometry(3, 32, 8)
        raise AssertionError("G=3 with N=32 should have raised")
    except SystemExit:
        pass

    print("grpo_train selftest: all checks passed")


# ----------------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------------
def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    # --- the experimental knobs ---
    p.add_argument("--num_generations", type=int, default=8, help="Group size G")
    p.add_argument("--rollouts_per_step", type=int, default=32,
                   help="Total generations per optimizer step. HELD CONSTANT across "
                        "the G sweep so group size is compared at matched compute.")
    p.add_argument("--scale_rewards", type=str2bool, default=False,
                   help="True = canonical GRPO (/std); False = mean-only (Dr. GRPO)")
    p.add_argument("--difficulty_weight", action="store_true",
                   help="Scale group rewards by 4p(1-p). Requires --scale_rewards false.")
    p.add_argument("--extractor", default="strict", choices=list(EXTRACTORS))
    p.add_argument("--seed", type=int, default=0)
    # --- the usual ---
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty to reference")
    p.add_argument("--max_prompt_length", type=int, default=384)
    p.add_argument("--max_completion_length", type=int, default=384)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--format_bonus", type=float, default=0.1)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--run_name", default="grpo")
    p.add_argument("--output_dir", default="./runs")
    p.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # --- matched-budget geometry ---
    G, N = args.num_generations, args.rollouts_per_step
    bs, grad_accum = geometry(G, N, args.per_device_train_batch_size)
    prompts_per_step = N // G
    print(f"[geometry] G={G}  rollouts/step={N}  distinct prompts/step={prompts_per_step}  "
          f"per_device_bs={bs}  grad_accum={grad_accum}")

    if args.difficulty_weight and args.scale_rewards:
        # A constant multiplier on a group's rewards cancels exactly in (r-mean)/std.
        # Failing loudly beats running a 3-hour no-op and reporting it as a result.
        raise SystemExit(
            "--difficulty_weight is a silent no-op when --scale_rewards true: a constant "
            "group multiplier cancels in (r-mean)/std. Pass --scale_rewards false."
        )

    out_dir = os.path.join(args.output_dir, args.run_name)
    stats_log = os.path.join(out_dir, "group_stats.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # "flash_attention_2" if installed
    )

    train_ds = build_dataset("train", tokenizer, max_prompt_length=args.max_prompt_length)

    reward = VerifiableReward(
        extractor=args.extractor,
        tokenizer=tokenizer,
        stats_log=stats_log,
        format_bonus=args.format_bonus,
        difficulty_weight=args.difficulty_weight,
    )

    config = GRPOConfig(
        output_dir=out_dir,
        run_name=args.run_name,
        seed=args.seed,
        learning_rate=args.learning_rate,
        beta=args.beta,
        scale_rewards=args.scale_rewards,
        num_generations=G,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.max_steps // 4 or 1,
        bf16=torch.cuda.is_available(),   # bf16 needs CUDA; fall back on Mac/CPU
        gradient_checkpointing=True,
        use_vllm=args.use_vllm,
        report_to=args.report_to,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward],
        args=config,
        train_dataset=train_ds,
    )

    # Record the run's identity next to its logs so analyze.py never has to parse
    # the run name to recover hyperparameters.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "grad_accum": grad_accum,
                   "prompts_per_step": prompts_per_step}, f, indent=2)

    trainer.train()
    trainer.save_model(out_dir)


if __name__ == "__main__":
    main()
