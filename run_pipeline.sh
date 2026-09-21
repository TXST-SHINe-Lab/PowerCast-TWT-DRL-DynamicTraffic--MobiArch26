#!/usr/bin/env bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

# --- run_pipeline.sh: ONE driver for the full TWT-PowerCast experiment, end-to-end ---
#
#   STAGE 0  action tables ........... generate_action_tables.py (RUN_TABLES)
#   STAGE 1  harvest/Vcap plots ...... show RF energy harvesting is happening
#   STAGE 2  full EDA + obs norms ..... fit signal-normalization params (prompts
#                                       before overwriting obs_warmstart_stats.json)
#   STAGE 3  full training ............ lstm_ppo + asymmetric critic, v5 reward
#   STAGE 4  full evaluation .......... trained policy vs the analytical_md1 baseline, per device class
#
# This does NOT do environment setup (NS-3 patches, cmake, build) — that lives in
# setup_fresh_ns3.sh and must already be done (the pybind .so must import).
#
# Run from anywhere; it resolves its own paths.
#   bash run_pipeline.sh
#
# Knobs (env vars, with full-run defaults). Examples:
#   RUN_TRAIN=0 RUN_EVAL=0 bash run_pipeline.sh          # just plots + EDA
#   TRAIN_BATCHES=10 EDA_BATCHES=3 bash run_pipeline.sh  # quick smoke of every stage
#   NORMS_MODE=skip bash run_pipeline.sh                 # never touch the live obs norms
#   RUN_PLOT=0 RUN_EDA=0 RUN_TRAIN=0 bash run_pipeline.sh                 # eval-only,
#       falls back to the newest shipped run's ckpt_final.pt under results/
#   RUN_PLOT=0 RUN_EDA=0 RUN_TRAIN=0 CKPT=path/to.pt bash run_pipeline.sh # eval a specific ckpt
set -euo pipefail

# --- paths -----------------------------------------------------------------
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_ROOT="$(cd "$PROJ/../../../.." && pwd)"         # .../ns-3.44
VENV="${VENV:-$(cd "$NS3_ROOT/../.." && pwd)/EHRL}" # default: $NS3_ROOT/../../EHRL
PY="$VENV/bin/python3.11"
SCRIPTS="$PROJ/ppo-sb3-scripts"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS="$PROJ/results"                                # SINGLE generated-output root
RUN_ROOT="${RUN_ROOT:-$RESULTS/runs/run_$TS/pipeline}" # same run_<timestamp>/pipeline layout as reproduce.sh; override so callers can locate outputs

# --- stage toggles ---------------------------------------------------------
RUN_TABLES="${RUN_TABLES:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
RUN_EDA="${RUN_EDA:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

# --- common scenario -------------------------------------------------------
NSTA="${NSTA:-12}"
NREHD="${NREHD:-8}"
REWARD="${REWARD:-twt_pf_demand_v5}"
MAX_STEPS="${MAX_STEPS:-100}" # rollout/eval horizon (train and eval must match; a 100-step policy collapses at a shorter horizon)

# --- stage 1 (Vcap/harvest diagnostic) -------------------------------------
VCAP_NSTA="${VCAP_NSTA:-10}"
VCAP_NREHD="${VCAP_NREHD:-6}"
VCAP_SIMTIME_MS="${VCAP_SIMTIME_MS:-120000}"
VCAP_SIMID="${VCAP_SIMID:-7000}"
VCAP_SLEEP_GROUPS="${VCAP_SLEEP_GROUPS:-8}"
VCAP_WAKE_MS="${VCAP_WAKE_MS:-30}"
VCAP_INTERVAL_MS="${VCAP_INTERVAL_MS:-100}"

# --- stage 2 (EDA + norm refit) --------------------------------------------
EDA_BATCHES="${EDA_BATCHES:-12}"
EDA_WORKERS="${EDA_WORKERS:-20}"
EDA_UPDATES="${EDA_UPDATES:-100}"  # updates per EDA episode (must match DURATION_IN_UPDATE for a real fit)
NORMS_MODE="${NORMS_MODE:-prompt}" # prompt | apply | skip

# --- stage 3 (training) ----------------------------------------------------
# Defaults below are the PAPER's configuration (Table 2). They previously disagreed with it:
# --n-epochs 15 / --ent-coef 0.002 were hardcoded into the orchestrator call (inherited from a
# superseded run), and TRAIN_BATCHES/BASE_SEED pointed at that run too — so anyone invoking this
# script reproduced a different model than the paper describes. All are env-overridable now.
TRAIN_BATCHES="${TRAIN_BATCHES:-200}"
TRAIN_WORKERS="${TRAIN_WORKERS:-16}"
TRAIN_EPB="${TRAIN_EPB:-24}"     # episodes per batch (rolling pool)
CRN="${CRN:-4}"                  # Common-Random-Numbers scenario groups (0=off; must divide workers)
BASE_SEED="${BASE_SEED:-100000}" # training scenario seed base
N_EPOCHS="${N_EPOCHS:-10}"       # PPO epochs per batch   (Table 2)
ENT_COEF="${ENT_COEF:-0.01}"     # entropy coefficient    (Table 2)
LR="${LR:-3e-4}"                 # learning rate          (Table 2)

# --- stage 4 (structured per-device eval: model vs analytical) --------------
EVAL_BASELINE="${EVAL_BASELINE:-analytical_md1}" # the paper's baseline (Sec. V-A, Eq. 9)
EVAL_SPLITS="${EVAL_SPLITS:-4 8 12 16}"          # n_rehd edge-case grid (total STA fixed at 20)
EVAL_SEEDS="${EVAL_SEEDS:-20}"                   # matched scenarios per split (paper: 20 -> 160 episodes)
EVAL_WORKERS="${EVAL_WORKERS:-20}"

banner() {
    echo
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

[ -x "$PY" ] || {
    echo "ERROR: venv python not found at $PY (set VENV=...)"
    exit 1
}
source "$VENV/bin/activate"
# The pybind .so is deployed next to the sources, not into site-packages, and the
# workers add that directory to sys.path themselves. This pre-flight check must do the
# same or it fails purely on CWD -- which it did when the script was invoked from the
# NS-3 root, exactly as the README instructs.
PYTHONPATH="$PROJ:${PYTHONPATH:-}" "$PY" -c "import pb_twt_powercast_interface_py" 2>/dev/null ||
    {
        echo "ERROR: pb_twt_powercast_interface_py not importable from $PROJ — run setup_fresh_ns3.sh + ./ns3 build first."
        exit 1
    }
mkdir -p "$RUN_ROOT" "$RESULTS/data-log"
cd "$NS3_ROOT" # the wrapper chdir's here anyway; needed for the direct ./ns3 run below
export PROJ
echo "PROJ=$PROJ"
echo "NS3_ROOT=$NS3_ROOT"
echo "VENV=$VENV"
echo "RUN_ROOT=$RUN_ROOT"

# --- STAGE 0 — generate the action tables ---
# The schedule/assignment tables are GENERATED artifacts, not inputs: the policy's
# head sizes are auto-derived from them (twt_spawn_worker.default_obs_kwargs), so a
# run that inherits stale tables silently trains a differently-shaped agent. Emitting
# them here makes the pipeline self-contained -- nothing under this directory has to
# pre-exist except source. Deterministic: regeneration is byte-identical modulo the
# embedded timestamp.
if [ "$RUN_TABLES" = "1" ]; then
    banner "STAGE 0/5 — generate action tables"
    "$PY" "$PROJ/exploration-scripts/generate_action_tables.py" \
        --output-dir "$PROJ/exploration-scripts"
    "$PY" - <<'PYT'
import json, os, sys
d = os.path.join(os.environ["PROJ"], "exploration-scripts")
ns = len(json.load(open(os.path.join(d, "schedule_table.json")))["schedules"])
na = len(json.load(open(os.path.join(d, "assignment_table.json")))["assignments"])
print(f"[stage0] action heads: {ns} schedules x {na} assignments x 10 PDW = {ns*na*10} combos")
PYT
fi

# --- STAGE 1 — harvest / Vcap plots ---
if [ "$RUN_PLOT" = "1" ]; then
    banner "STAGE 1/5 — harvest is happening (Vcap trends + harvest curve)"
    # Grouped-sleep TWT so REHDs doze and harvest; --logVcap writes the per-REHD Vcap CSV.
    ./ns3 run "twt-powercast-main-simulation --nStations=$VCAP_NSTA --nRehd=$VCAP_NREHD \
      --enableDynamicTWT=false --simulationTime=$VCAP_SIMTIME_MS --quietMode=true \
      --disableTraces=true --twtSleepGroups=$VCAP_SLEEP_GROUPS --twtGroupWakeMs=$VCAP_WAKE_MS \
      --logVcap=true --vcapIntervalMs=$VCAP_INTERVAL_MS --simId=$VCAP_SIMID"
    VCAP_CSV="$RESULTS/data-log/vcap_trace_${VCAP_SIMID}.csv"
    "$PY" "$PROJ/plot_vcap.py" --csv "$VCAP_CSV" --out "$RESULTS/data-log"
    "$PY" "$PROJ/plot_harvest_curve.py"
    echo "[stage1] Vcap + queue + harvest-curve plots -> $RESULTS/data-log/"
fi

# --- STAGE 2 — full EDA + obs-normalization parameters ---
if [ "$RUN_EDA" = "1" ]; then
    banner "STAGE 2/5 — EDA: collect signals + fit obs-normalization params"
    EDA_DIR="$RUN_ROOT/eda"
    EDA_SPAN=1 "$PY" "$SCRIPTS/twt_eda_collect.py" \
        --num-batches "$EDA_BATCHES" --num-workers "$EDA_WORKERS" \
        --n-stations "$NSTA" --n-rehd "$NREHD" --reward-preset "$REWARD" \
        --max-updates "$EDA_UPDATES" --out "$EDA_DIR"
    PROPOSED="$EDA_DIR/obs_warmstart_proposed.json"
    "$PY" "$SCRIPTS/twt_refit_obs_norms.py" --shards "$EDA_DIR/shards" --out "$PROPOSED"

    LIVE="$SCRIPTS/obs_warmstart_stats.json"
    case "$NORMS_MODE" in
        apply)
            cp "$PROPOSED" "$LIVE"
            echo "[stage2] NORMS_MODE=apply -> updated $LIVE"
            ;;
        skip) echo "[stage2] NORMS_MODE=skip -> kept existing $LIVE" ;;
        *)
            if [ -t 0 ]; then
                read -rp "[stage2] Update obs_warmstart_stats.json with the refit norms above? [y/N] " ans || ans=""
                if [[ "$ans" =~ ^[Yy] ]]; then
                    cp "$PROPOSED" "$LIVE"
                    echo "[stage2] updated $LIVE"
                else echo "[stage2] kept existing $LIVE (proposal at $PROPOSED)"; fi
            else
                echo "[stage2] non-interactive; kept existing $LIVE (proposal at $PROPOSED; set NORMS_MODE=apply to adopt)"
            fi
            ;;
    esac
fi

# --- STAGE 3 — full training (lstm_ppo + asymmetric critic, v5 reward) ---
if [ "$RUN_TRAIN" = "1" ]; then
    banner "STAGE 3/5 — training ($TRAIN_BATCHES batches, $TRAIN_WORKERS workers)"
    # No explicit --policy-kwargs here — num_schedules/num_assignments/num_pdw_levels are
    # auto-derived from the carved action tables (twt_spawn_worker.default_obs_kwargs); a stale
    # hardcoded override (e.g. num_schedules:25 vs the actual carved 24) would silently mis-size
    # the policy.
    "$PY" "$SCRIPTS/twt_batch_orchestrator.py" \
        --policy-arch lstm_ppo --asymmetric-critic \
        --reward-preset "$REWARD" --n-stations "$NSTA" --n-rehd "$NREHD" --variable-split \
        --num-workers "$TRAIN_WORKERS" --pool-size "$TRAIN_WORKERS" --episodes-per-batch "$TRAIN_EPB" \
        --crn-groups "$CRN" --num-batches "$TRAIN_BATCHES" --max-steps-per-episode "$MAX_STEPS" \
        --warmup-steps 5 --n-epochs "$N_EPOCHS" --ent-coef "$ENT_COEF" --learning-rate "$LR" \
        --base-seed "$BASE_SEED" --obs-warmstart --normalize-return --save-freq 10 \
        --tensorboard-log "$RUN_ROOT/tb_logs" --output-dir "$RUN_ROOT/train"
fi

# --- STAGE 4 — structured per-device evaluation: model vs analytical baseline ---
if [ "$RUN_EVAL" = "1" ]; then
    banner "STAGE 4/5 — structured per-device eval (model vs $EVAL_BASELINE)"
    if [ -n "${CKPT:-}" ]; then
        echo "[stage4] CKPT override -> evaluating $CKPT"
    else
        CKPT="$(ls -t "$RUN_ROOT"/train/*/ckpt_final.pt 2>/dev/null | head -1 || true)"
        if [ -z "$CKPT" ]; then
            # Fall back to the newest SHIPPED run's final checkpoint. (There is no
            # separate pretrained/ copy: results/ already carries the model, and a
            # second copy only invites the two drifting apart.)
            CKPT="$(ls -t "$RESULTS"/runs/*/pipeline/train/*/ckpt_final.pt 2>/dev/null | head -1)"
            echo "[stage4] no freshly-trained ckpt found; evaluating the shipped $CKPT"
        fi
    fi
    EVAL_DIR="$RUN_ROOT/eval"
    # Head-to-head over the REHD-split edge-case grid, matched scenarios; logs per-device
    # metrics (served / drop / throughput / latency / fairness + REHD energy), not just reward.
    "$PY" "$SCRIPTS/twt_eval_structured.py" \
        --model-ckpt "$CKPT" --baseline "$EVAL_BASELINE" \
        --splits $EVAL_SPLITS --seeds "$EVAL_SEEDS" \
        --reward-preset "$REWARD" --max-steps "$MAX_STEPS" --warmup-steps 5 \
        --num-workers "$EVAL_WORKERS" --out "$EVAL_DIR" # auto-runs the per-device report
    echo "[stage4] per-device report -> $EVAL_DIR/report.txt (+ per_class_comparison.csv)"
fi

banner "DONE — all requested stages complete. Outputs under $RUN_ROOT"
