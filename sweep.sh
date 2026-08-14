#!/usr/bin/env bash
# The full experimental grid, in priority order. Run on the GPU box.
# Everything shares one rollout budget per step (32) and one step budget (500),
# so every comparison is at matched compute.
#
#   bash sweep.sh            # phase 1 only (the core grid)
#   bash sweep.sh all        # phases 1-3
#
# Rough cost on one RTX 4090 with vLLM: each run is a few hours.
# Phase 1 = 15 runs. Start it, check fig2 after the first 5 (seed 0) finish --
# if the curve is flat, stop and rethink before burning seeds 1-2.
set -euo pipefail

STEPS=500
BUDGET=32                       # rollouts per optimizer step, held constant
VLLM=${VLLM:---use_vllm}        # VLLM= bash sweep.sh   to disable
PHASE=${1:-core}

run() {  # run <name> <extra args...>
    local name=$1; shift
    if [ -f "runs/$name/eval.json" ]; then
        echo "== $name: already evaluated, skipping"
        return
    fi
    echo "== training $name"
    python grpo_train.py --run_name "$name" --max_steps $STEPS \
        --rollouts_per_step $BUDGET $VLLM "$@"
    echo "== evaluating $name"
    python evaluate.py --model "runs/$name" --benchmarks gsm8k gsm_plus \
        --save_completions
}

# ---------------------------------------------------------------------------
# Phase 1 (the paper's core): group-size sweep x 3 seeds, mean-only advantages.
# Twists #1, #2, #6. 15 runs.
# ---------------------------------------------------------------------------
for seed in 0 1 2; do
    for G in 2 4 8 16 32; do
        run "g${G}_seed${seed}" --num_generations $G --seed $seed \
            --scale_rewards false
    done
done
python analyze.py   # figures update after every phase; look at them

if [ "$PHASE" != "all" ]; then exit 0; fi

# ---------------------------------------------------------------------------
# Phase 2: canonical-vs-mean-only control at the best G (edit BEST_G after
# reading fig2), plus difficulty weighting. Twist #3. 6 runs.
# ---------------------------------------------------------------------------
BEST_G=${BEST_G:-8}
for seed in 0 1 2; do
    run "g${BEST_G}_std_seed${seed}"  --num_generations $BEST_G --seed $seed \
        --scale_rewards true
    run "g${BEST_G}_dw_seed${seed}"   --num_generations $BEST_G --seed $seed \
        --scale_rewards false --difficulty_weight
done
python analyze.py

# ---------------------------------------------------------------------------
# Phase 3: train under the lenient grader -- does reward hacking show up as a
# strict/lenient gap at eval time? Twist #4. 3 runs.
# ---------------------------------------------------------------------------
for seed in 0 1 2; do
    run "g${BEST_G}_lenient_seed${seed}" --num_generations $BEST_G --seed $seed \
        --scale_rewards false --extractor last_number
done
python analyze.py

# Final OOD pass on MATH-500 for the frontier runs only (it's the slow benchmark).
for seed in 0 1 2; do
    python evaluate.py --model "runs/g${BEST_G}_seed${seed}" \
        --benchmarks math500 --out "runs/g${BEST_G}_seed${seed}/eval_math500.json"
done
echo "sweep complete -- figures/ has the paper."
