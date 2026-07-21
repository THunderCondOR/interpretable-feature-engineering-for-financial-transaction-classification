#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$(command -v python)}"
datasets=(gender age rosbank)
dry_run=false
phase="${1:-}"

if [[ "$phase" == "--dry-run" ]]; then
    dry_run=true
    phase="${2:-baseline}"
elif [[ -z "$phase" ]]; then
    phase="baseline"
fi

if [[ "$phase" != "baseline" && "$phase" != "cot" && "$phase" != "queue" ]]; then
    echo "Usage: $0 [--dry-run] [baseline|cot|queue]" >&2
    exit 2
fi

: "${API_BASE_URL:?API_BASE_URL must be exported}"
: "${API_KEY:?API_KEY must be exported}"

run_id="$(date +%Y%m%d_%H%M%S)"
marker_dir="$repo_dir/results/experiment_queue/$run_id"
if [[ "$phase" == "queue" && "$dry_run" == false ]]; then
    mkdir -p "$marker_dir"
fi

if [[ "$dry_run" == false ]]; then
    for dataset in "${datasets[@]}"; do
    session="llm_${phase}_${dataset}"
    if tmux has-session -t "$session" 2>/dev/null; then
            echo "Session already exists: $session" >&2
            exit 1
        fi
    done
    tmux set-environment -g API_BASE_URL "$API_BASE_URL"
    tmux set-environment -g API_KEY "$API_KEY"
fi

for dataset in "${datasets[@]}"; do
    session="llm_${phase}_${dataset}"
    if [[ "$phase" == "baseline" ]]; then
        command="$python_bin run_pipeline.py --config configs/${dataset}.yaml --steps stats,prompts,cot,llm_eval --splits test --no-resume-cot"
    elif [[ "$phase" == "cot" ]]; then
        command="$python_bin run_pipeline.py --config configs/${dataset}.yaml --steps stats,prompts --splits train,val,test && $python_bin run_pipeline.py --config configs/${dataset}.yaml --steps cot --splits train,val --no-resume-cot && $python_bin run_pipeline.py --config configs/${dataset}.yaml --steps claims,cot_features,ml --splits train,val,test --experiments cot,concat"
    else
        baseline_command="$python_bin run_pipeline.py --config configs/${dataset}.yaml --steps stats,prompts,cot,llm_eval --splits test --no-resume-cot"
        cot_command="$python_bin run_pipeline.py --config configs/${dataset}.yaml --steps stats,prompts --splits train,val && $python_bin run_pipeline.py --config configs/${dataset}.yaml --steps cot --splits train,val --no-resume-cot && $python_bin run_pipeline.py --config configs/${dataset}.yaml --steps claims,cot_features,ml --splits train,val,test --experiments cot,concat"
        barrier="while [ ! -f $marker_dir/baseline_gender.done ] || [ ! -f $marker_dir/baseline_age.done ] || [ ! -f $marker_dir/baseline_rosbank.done ]; do sleep 15; done"
        command="$baseline_command && touch $marker_dir/baseline_${dataset}.done && echo 'Baseline complete; waiting at global barrier' && $barrier && echo 'All baselines complete; starting CoT pipeline' && $cot_command && touch $marker_dir/complete_${dataset}.done"
    fi
    if [[ "$dry_run" == true ]]; then
        echo "[DRY RUN] $session: $command"
        continue
    fi
    tmux new-session \
        -d \
        -s "$session" \
        -c "$repo_dir"
    tmux send-keys -t "$session" -l "$command"
    tmux send-keys -t "$session" Enter
    echo "Prepared $session: $command"
done

if [[ "$dry_run" == false ]]; then
    echo "All three LLM experiments started."
fi
