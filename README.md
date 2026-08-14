# Where Does GRPO's Compute Actually Go?

Group-size, wasted-gradient, and grader-fragility experiments for GRPO on
verifiable math rewards. Qwen2.5-0.5B-Instruct + GSM8K + TRL, runs on a single
RTX 4090.

## The question

GRPO computes advantages *within* a group of G rollouts per prompt. A group
that is all-correct or all-wrong has zero reward variance → zero advantage →
**zero gradient**. Every rollout in it was generated for nothing. Nobody
reports how large that wasted fraction is, how it evolves over training, or
how the choice of G trades off against it at fixed compute. That's this repo.

**Claims we test** (all at matched rollout budget, 3 seeds, multiple graders):

1. **Group size** — under a fixed budget `N = G × prompts`, is there an
   interior optimum for G, and does it move as the policy improves?
2. **Wasted compute** — what fraction of rollouts land in zero-variance
   groups, and how does it change over training? (Logged per batch, free.)
3. **Difficulty weighting** — reweighting groups by `4p(1−p)` (peaked at
   pass-rate 0.5) is a zero-extra-rollout cousin of DAPO's dynamic sampling.
   Does it help?
4. **Grader fragility** — the same completions scored under strict
   (`#### x` required), lenient (last-number fallback), and boxed/symbolic
   graders. Does the *ranking* of variants depend on the grader?
5. **Generalization** — does the in-distribution (GSM8K) winner also win on
   GSM-Plus and MATH-500?
6. **Seed honesty** — every headline number comes with min–max over 3 seeds.

## Files

| file | what |
|---|---|
| `grpo_baseline.py` | untouched known-good reference (canonical GRPO via TRL) |
| `grpo_train.py` | instrumented trainer: matched-budget geometry, group-stats logging, difficulty weighting, grader choice |
| `extractors.py` | the three graders + numeric/symbolic equivalence |
| `evaluate.py` | greedy eval on GSM8K / GSM-Plus / MATH-500, re-scored under every grader |
| `analyze.py` | logs → the paper's figures |
| `sweep.sh` | the full experimental grid, in priority order |

## Quickstart

```bash
# laptop, no GPU: verify all logic
python grpo_train.py --selftest
python evaluate.py --selftest
python analyze.py --selftest
python extractors.py

# GPU box
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "vllm>=0.6"     # optional, big generation speedup

# smoke test (~minutes)
python grpo_train.py --max_steps 20 --report_to none --run_name smoke

# the real thing
bash sweep.sh          # phase 1: 5 group sizes × 3 seeds
bash sweep.sh all      # + std-scaling control, difficulty weighting, lenient-grader run
python analyze.py      # figures/ → fig1 wasted compute, fig2 acc-vs-G, fig3 grader table
```

Every run writes `runs/<name>/config.json` (exact hyperparameters),
`group_stats.jsonl` (per-batch group accounting), and `eval.json`
(accuracy per benchmark per grader).

## Design notes

- **Matched compute, always.** `--rollouts_per_step` is held constant across
  the G sweep; varying G varies distinct-prompts-per-step, not total
  generation. Comparisons at unmatched compute are the field's most common
  self-own.
- **No trainer surgery.** All instrumentation lives in the reward callable
  (stable TRL public API), so the repo survives TRL version bumps.
- **Difficulty weighting requires mean-only advantages** — with std-scaling a
  constant group multiplier cancels exactly. The trainer refuses the
  combination rather than silently no-op'ing.
- **Greedy eval.** Otherwise seed-variance claims are confounded by sampling
  noise.

## Caveats (say them before a reviewer does)

- 0.5B-scale conclusions may not transfer to 7B+. This is a measurement paper
  about the *estimator*, not a leaderboard entry.
- GSM8K rewards are near-binary here; results may differ for dense or learned
  rewards.
- One confirmation run at 1.5B+ is planned if compute allows.

## Related work we build on / must be compared to

DeepSeekMath (GRPO), Dr. GRPO (mean-only advantages), DAPO (dynamic
sampling / clip-higher), GSPO (sequence-level), Lite-PPO (global std).
Our group-size and wasted-gradient measurements are complementary to all of
them; the difficulty-weighting variant is the zero-extra-rollout counterpart
of DAPO's dynamic sampling.
