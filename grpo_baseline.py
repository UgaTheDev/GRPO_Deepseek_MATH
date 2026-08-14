"""
GRPO baseline: Qwen2.5-0.5B-Instruct on GSM8K with a verifiable (rule-based) reward.

This is your Day-1 "running code." It reproduces canonical GRPO and gives you the
first ablation for free via --scale_rewards:
    - scale_rewards=True   -> (r - mean)/std          [canonical GRPO, DeepSeekMath]
    - scale_rewards=False  -> (r - mean)               [mean-only, the Dr. GRPO direction]

Estimators #3 (global std) and #4 (difficulty reweighting) come later by subclassing
GRPOTrainer; this file is deliberately the clean baseline so you have a known-good
reference to diff against.

------------------------------------------------------------------------------------
SETUP (on your rented GPU box; a single RTX 4090 is plenty):

    python -m venv .venv && source .venv/bin/activate
    pip install "torch>=2.3" "transformers>=4.48,<4.52" "trl>=0.14,<0.16" \
                "datasets>=2.19" "peft>=0.11" "accelerate>=0.30" wandb

    # optional but recommended for fast generation:
    pip install "vllm>=0.6"     # then pass --use_vllm

RUN (quick smoke test — a few minutes, proves the loop works):

    python grpo_baseline.py --max_steps 20 --num_generations 4 --report_to none

RUN (real baseline run):

    python grpo_baseline.py --scale_rewards true  --run_name grpo_canonical
    python grpo_baseline.py --scale_rewards false --run_name grpo_mean_only
------------------------------------------------------------------------------------
"""

import argparse
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# ----------------------------------------------------------------------------------
# 1. Prompt formatting + answer extraction
# ----------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "then give the final numeric answer on its own line as:\n#### <answer>"
)


def extract_gsm8k_gold(answer_field: str) -> str:
    """GSM8K gold answers end with '#### <number>'. Pull that number out."""
    m = re.search(r"####\s*(-?[\d,]+)", answer_field)
    return m.group(1).replace(",", "").strip() if m else ""


def extract_pred(text: str) -> str:
    """Extract the model's final answer. Prefer the '#### x' form; fall back to
    the last number in the completion so we don't over-penalize formatting."""
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def build_dataset(split: str, tokenizer, max_prompt_length: int = 384):
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
        n_tok = len(tokenizer(prompt)["input_ids"])
        return {"prompt": prompt, "gold": extract_gsm8k_gold(ex["answer"]),
                "prompt_len": n_tok}

    ds = ds.map(_map, remove_columns=ds.column_names)
    # Version-proof prompt-length control: newer TRL removed GRPOConfig.max_prompt_length,
    # so we filter here instead. GSM8K prompts are short, so this drops very few examples.
    ds = ds.filter(lambda ex: ex["prompt_len"] <= max_prompt_length)
    return ds.remove_columns(["prompt_len"])


# ----------------------------------------------------------------------------------
# 2. Verifiable reward functions
#    TRL passes `completions` (list[str]) plus any dataset columns as kwargs.
#    Return a list[float], one reward per completion.
# ----------------------------------------------------------------------------------
def correctness_reward(completions, gold, **kwargs):
    """+1.0 if the extracted answer matches the gold answer, else 0.0.
    This is the load-bearing signal — no reward model, so you're measuring the
    OPTIMIZER, not a learned reward's noise."""
    return [1.0 if extract_pred(c) == g else 0.0 for c, g in zip(completions, gold)]


def format_reward(completions, **kwargs):
    """Small shaping bonus for using the '#### <answer>' format. Kept tiny so it
    never dominates correctness. Drop it if you want the cleanest possible signal."""
    pat = re.compile(r"####\s*-?[\d,]+")
    return [0.1 if pat.search(c) else 0.0 for c in completions]


# ----------------------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------------------
def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--scale_rewards", type=str2bool, default=True,
                   help="True = canonical GRPO (÷std); False = mean-only (Dr. GRPO direction)")
    p.add_argument("--num_generations", type=int, default=8,
                   help="Group size G. Must divide the effective train batch size.")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty to reference model")
    p.add_argument("--max_prompt_length", type=int, default=384)
    p.add_argument("--max_completion_length", type=int, default=384)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--run_name", default="grpo_canonical")
    p.add_argument("--output_dir", default="./grpo_out")
    p.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # switch to "flash_attention_2" if installed
    )

    train_ds = build_dataset("train", tokenizer, max_prompt_length=args.max_prompt_length)

    # bf16 requires a CUDA GPU; auto-fall back on Mac/CPU so local smoke tests run.
    use_bf16 = torch.cuda.is_available()

    config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        beta=args.beta,
        # --- the ablation switch ---
        scale_rewards=args.scale_rewards,
        # --- group / batch geometry ---
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        # --- schedule / logging ---
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=100,
        bf16=use_bf16,
        gradient_checkpointing=True,
        use_vllm=args.use_vllm,
        report_to=args.report_to,
        log_completions=True,   # logs sample generations so you can eyeball quality
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=train_ds,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()