#!/usr/bin/env bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

# --- reproduce.sh ---
# High-level wrapper that reproduces the paper's results/figures from scratch by orchestrating the existing core scripts (no logic duplicated):
#
#   0  build     ns-3 + the pybind .so
#   1  carve     action tables (generate_action_tables.py)    (run_pipeline.sh's stage 0)
#   2  pipeline  run_pipeline.sh, called ONCE with paper-exact pinned args
#                (n_stations=12, n_rehd=8, reward=twt_pf_demand_v5,
#                eval baseline=analytical_md1, splits 4/8/12/16). Inside it:
#                   RUN_PLOT   harvest/Vcap diagnostic plots (120 s grouped-sleep sim)
#                   RUN_EDA    EDA sweep + fit of obs_warmstart_stats.json
#                   RUN_TRAIN  lstm_ppo + asymmetric critic, 200 batches x 24 episodes
#                   RUN_EVAL   trained policy vs the analytical_md1 baseline, per device class
#                RUN_TABLES is off here: stage 1 above already carved the tables.
#   3  vcap      paper Vcap-vs-time figure traces (vcap_episode.py, seeds 42-45)
#   4  figs      make_figs.py -> all data-driven paper figures (+ console T1/T2 numbers)
#
# Stage 2 IS run_pipeline.sh — read that script for the EDA/train/eval mechanics; this
# wrapper only pins the paper's parameters and tells it where to put its outputs.
#
# Every invocation writes into its own run folder, results/runs/run_YYYYMMDD_HHMMSS/ (the same run_<timestamp> naming as the ICCCN'26 repo's run_all.sh).
# Earlier runs, the committed run (results/runs/repro_20260825/) and the committed figures (results/figs/) are never deleted or overwritten.
# The run folder holds pipeline/ (EDA, checkpoints, eval), figs/ (this run's figures + vcap_data.csv), pipeline.log, and inputs_before/.
#
# Three generated inputs live in the source tree and are shared by every run: the two action tables (stage 1) and ppo-sb3-scripts/obs_warmstart_stats.json (stage 2 applies the refit norms before training).
# They are regenerated in place, so the previous copies are saved to <run>/inputs_before/ first.
#
# Stage 0 runs `ns3 clean` and rebuilds ns-3 from scratch; that dominates the setup cost (tens of minutes).
# NS3_CLEAN=0 makes it incremental when you are iterating.
#
# RUN selects the run folder: a bare name (RUN=run_20260918_185730) is looked up under results/runs/, a path is used as given.
# It is required when STAGES has 4 but not 2 (figures need a finished pipeline to read).
#
# Usage:
#   bash reproduce.sh                                   # everything, paper-exact params (multi-hour), new run folder
#   RUN=run_20260918_185730 STAGES="2 3 4" bash reproduce.sh   # resume an interrupted run (finished EDA shards are skipped)
#   RUN=repro_20260825 STAGES="4" bash reproduce.sh     # regenerate figures from the committed run into its figs/
#   NS3_CLEAN=0 bash reproduce.sh                       # skip `ns3 clean` (incremental build, much faster)
#   RUN_EDA=0 bash reproduce.sh                         # (passes through to run_pipeline.sh) skip EDA/norms
#   EDA_BATCHES=100 bash reproduce.sh                   # EDA episodes = EDA_BATCHES x EDA_WORKERS (default 100x20)
#   TRAIN_BATCHES=200 NUM_WORKERS=16 bash reproduce.sh
#   CLEAN=1 bash reproduce.sh                           # opt-in: delete everything under results/ first (the old default)
set -u

# Resolve THIS example's directory from the script location, and the ns-3 root + venv from it (same convention as run_pipeline.sh: <NS3_ROOT>/../../EHRL, override with VENV=...).
# Was hardcoded to a stale directory name ("mobicom"), which silently pointed every stage at a path that no longer exists once the directory was renamed.
EX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXNAME="$(basename "$EX")"
NS3="$(cd "$EX/../../../.." && pwd)"
VENV="${VENV:-$(cd "$NS3/../.." && pwd)/EHRL}"
PY=$VENV/bin/python3.11
SCR=$EX/ppo-sb3-scripts
EXP=$EX/exploration-scripts

RESULTS=$EX/results # SINGLE generated-output root
STAGES=${STAGES:-"0 1 2 3 4"}
has() { echo " $STAGES " | grep -q " $1 "; }

# --- run folder ---
# Unset: a fresh results/runs/run_<timestamp>/ for this invocation.
# A bare name is looked up under results/runs/; a relative path is resolved against the caller's cwd (the script cd's to the ns-3 root below).
if [ -z "${RUN:-}" ]; then
    if has 4 && ! has 2; then
        echo "ERROR: STAGES='$STAGES' reads a finished pipeline; name the run with RUN=<name>. Available runs:"
        ls -1 "$RESULTS/runs" 2>/dev/null | sed 's/^/  /'
        exit 1
    fi
    RUN="$RESULTS/runs/run_$(date +%Y%m%d_%H%M%S)"
else
    case "$RUN" in
        */*) case "$RUN" in /*) ;; *) RUN="$PWD/$RUN" ;; esac ;;
        *) RUN="$RESULTS/runs/$RUN" ;;
    esac
    if ! has 2 && [ ! -d "$RUN" ]; then
        echo "ERROR: RUN='$RUN' does not exist. Available runs:"
        ls -1 "$RESULTS/runs" 2>/dev/null | sed 's/^/  /'
        exit 1
    fi
fi
PIPE_ROOT=$RUN/pipeline # handed to run_pipeline.sh as RUN_ROOT
FIGS_DIR=${PAPER_FIGS_DIR:-$RUN/figs}
VCAP_DATA=$FIGS_DIR/vcap_data.csv

cd "$NS3"
source "$VENV/bin/activate" 2>/dev/null || true
banner() {
    echo
    echo "==================== STAGE $1: $2 ===================="
    date
}

# ----------------------------- clean slate (opt-in) ---------------------------
# Off by default: each run already gets its own folder, so nothing from a previous run can be reused or overwritten.
# CLEAN=1 restores the old behaviour for a full run: delete everything under results/ (committed run and figures included) plus the generated action tables and obs norms.
# It only fires when the pipeline stage (2) is selected, so a figures-only or vcap-only rerun cannot delete the run it is reading.
#   CLEAN=0      never clean (default)
#   CLEAN=1      clean when stage 2 is selected
#   CLEAN=force  clean regardless of which stages are selected
CLEAN=${CLEAN:-0}
clean_slate() {
    banner C "clean slate — removing previous results"
    # Refuse to rm -rf anything that is not exactly <this example>/results.
    # A bare "${RESULTS:?}" only catches empty/unset; it would still happily delete a wrong non-empty path if EX ever resolved unexpectedly.
    case "$RESULTS" in
        "$EX/results") : ;;
        *)
            echo "  REFUSING to clean: RESULTS='$RESULTS' is not '$EX/results'"
            return 1
            ;;
    esac
    [ -d "$RESULTS" ] || {
        mkdir -p "$RESULTS"
        : >"$RESULTS/.gitkeep"
        echo "  (nothing to clean)"
        return 0
    }
    rm -rf "${RESULTS:?}"/* 2>/dev/null || true
    mkdir -p "$RESULTS"
    : >"$RESULTS/.gitkeep"
    echo "  cleared $RESULTS"
    # Only remove a generated input if the stage that regenerates it will run.
    if has 1 || [ "$CLEAN" = "force" ]; then
        rm -f "$EXP/schedule_table.json" "$EXP/assignment_table.json"
        echo "  cleared action tables (stage 1 regenerates)"
    fi
    if has 2 || [ "$CLEAN" = "force" ]; then
        rm -f "$SCR/obs_warmstart_stats.json"
        echo "  cleared obs norms (stage 2 regenerates)"
    fi
}
if [ "$CLEAN" = "force" ] || { [ "$CLEAN" = "1" ] && has 2; }; then
    clean_slate
fi

mkdir -p "$RUN"
echo "[run] $RUN"

# Save the shared source-tree inputs this run is about to regenerate, so the previous versions are kept with the run.
if has 1 || has 2; then
    mkdir -p "$RUN/inputs_before"
    for f in "$EXP/schedule_table.json" "$EXP/assignment_table.json" "$SCR/obs_warmstart_stats.json"; do
        [ -f "$f" ] && cp -p "$f" "$RUN/inputs_before/"
    done
fi

# Reap orphan ns-3 children + clear stale SHM (the shell blocks pkill, so os.kill).
reap() {
    "$PY" - <<'PYR'
import os, glob, signal, re, time
PAT = re.compile(r'twt-powercast-main-simulation|twt_spawn_worker|run_episode|twt_eda_collect')
me = os.getpid()
for _ in range(3):
    hit = []
    for p in os.listdir('/proc'):
        if not p.isdigit() or int(p) == me:
            continue
        try:
            cl = open('/proc/%s/cmdline' % p, 'rb').read().replace(b'\x00', b' ').decode('utf-8', 'ignore')
        except Exception:
            continue
        if cl and PAT.search(cl):
            hit.append(int(p))
    if not hit:
        break
    for p in hit:
        try: os.kill(p, signal.SIGKILL)
        except Exception: pass
    time.sleep(1)
for s in glob.glob('/dev/shm/My*'):
    try: os.remove(s)
    except Exception: pass
PYR
}

# ----------------------------- 0. build --------------------------------------
if has 0; then
    banner 0 "build ns-3 + pybind .so"
    # From-scratch build.
    # `ns3 clean` removes build/ AND cmake-cache/, so the configure guard below then sees a missing cache and reconfigures -- the two compose into a genuine clean build with no stale objects and no stale cmake state (a cached Python3_EXECUTABLE pointing at the wrong interpreter is a known way to get a .so that imports nowhere).
    #
    # This is the expensive half of a full reproduction: a complete ns-3 recompile, tens of minutes.
    # NS3_CLEAN=0 keeps the existing tree and builds incrementally instead.
    if [ "${NS3_CLEAN:-1}" != "0" ]; then
        echo "[build] ns3 clean — removing build/ + cmake-cache/ for a from-scratch build"
        ./ns3 clean || echo "[build] ns3 clean returned nonzero (nothing to clean?) — continuing"
    else
        echo "[build] NS3_CLEAN=0 — keeping the existing build tree (incremental)"
    fi
    # ALWAYS build.
    # This used to skip the whole step whenever a binary existed, which meant any C++ edit since the last build silently did NOT enter the "reproduction": the run used a stale binary and still reported success.
    # `configure` is idempotent and slow, so it runs only when the cmake cache is absent (which ns3 clean guarantees) or on CONFIGURE=force.
    if [ ! -f "$NS3/cmake-cache/CMakeCache.txt" ] || [ "${CONFIGURE:-}" = "force" ]; then
        PYBIND11_CMAKE_DIR="$("$PY" -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null)"
        if [ -z "$PYBIND11_CMAKE_DIR" ]; then
            echo "[build] FAILED — pybind11 not importable by $PY (pip install pybind11 in the venv)"
            exit 1
        fi
        ./ns3 configure --enable-examples --enable-tests -- \
            -DNS3_PYTHON_BINDINGS=ON \
            -DPython3_EXECUTABLE="$PY" \
            -DPython3_ROOT_DIR="$VENV" \
            -DNS3_BINDINGS_INSTALL_DIR="$VENV/lib/python3.11/site-packages" \
            -DCMAKE_PREFIX_PATH="$PYBIND11_CMAKE_DIR"
    else
        echo "[build] cmake cache present — skipping configure (CONFIGURE=force to redo)"
    fi
    ./ns3 build || {
        echo "[build] FAILED — aborting"
        exit 1
    }
    # The .so is deployed next to the sources, not into site-packages, and the workers put that directory on sys.path themselves --
    # so this check must too, or it fails purely on the current working directory and prints a spurious "check Python3_EXECUTABLE" warning.
    PYTHONPATH="$EX:${PYTHONPATH:-}" "$PY" -c "import pb_twt_powercast_interface_py" 2>/dev/null &&
        echo "[ok] pybind import works" || echo "[warn] pybind import failed — check Python3_EXECUTABLE"
fi

# ----------------------------- 1. carve tables -------------------------------
if has 1; then
    banner 1 "carve action tables -> exploration-scripts/{schedule,assignment}_table.json"
    "$PY" "$EXP/generate_action_tables.py" --output-dir "$EXP"
fi

# ----------------------------- 2. delegate to run_pipeline.sh ----------------
if has 2; then
    banner 2 "run_pipeline.sh (EDA+norms, train, eval) with paper-exact args"
    mkdir -p "$PIPE_ROOT"
    reap
    RUN_ROOT="$PIPE_ROOT" \
        RUN_PLOT="${RUN_PLOT:-1}" RUN_TABLES=0 \
        NSTA=12 NREHD=8 REWARD=twt_pf_demand_v5 MAX_STEPS="${MAX_STEPS:-100}" \
        NORMS_MODE=apply \
        EDA_BATCHES="${EDA_BATCHES:-100}" EDA_WORKERS="${EDA_WORKERS:-20}" \
        TRAIN_BATCHES="${TRAIN_BATCHES:-200}" TRAIN_WORKERS="${NUM_WORKERS:-16}" \
        TRAIN_EPB="${TRAIN_EPB:-24}" EDA_UPDATES="${EDA_UPDATES:-100}" CRN="${CRN:-4}" BASE_SEED=100000 \
        EVAL_BASELINE=analytical_md1 EVAL_SPLITS="4 8 12 16" EVAL_SEEDS="${EVAL_SEEDS:-20}" EVAL_WORKERS=20 \
        bash "$EX/run_pipeline.sh" 2>&1 | tee -a "$RUN/pipeline.log"
    rc=${PIPESTATUS[0]}
    reap
    echo "[pipeline] run_pipeline.sh rc=$rc — see $RUN/pipeline.log"
    [ "$rc" -ne 0 ] && echo "[pipeline] FAILED — inspect the log, then resume with: RUN=$(basename "$RUN") STAGES=\"2 3 4\" bash reproduce.sh"
fi

# ----------------------------- 3. vcap sims + extract ------------------------
if has 3; then
    banner 3 "vcap capacitor-trace sims (4 STA + 16 REHD, isolated REHD groups, max PDW)"
    # vcap_episode.py runs ONE constant-action episode with --logVcap; CSVs land in <example-dir>/data-log/vcap_trace_<simId>.csv.
    for sd in 42 43 44 45; do
        sim=$((7300 + sd))
        echo "[vcap] seed $sd -> simId $sim"
        reap
        "$PY" "$SCR/vcap_episode.py" "$sd" "$sim" >/dev/null 2>&1
    done
    echo "[vcap] extract 4 same-class REHD traces (2 nearest / 2 farthest) -> $VCAP_DATA"
    mkdir -p "$(dirname "$VCAP_DATA")" # stage 4 creates FIGS_DIR; stage 3 runs first
    EX="$EX" "$PY" - "$VCAP_DATA" <<'PYV'
import csv, sys, os, glob
# Pick the traces PROGRAMMATICALLY instead of hardcoding (sim, sta_id, distance) triples transcribed from one historical run.
# Those constants happened to still be correct -- ns-3 is deterministic per seed -- but nothing verified that, and a shifted topology draw would have silently produced an empty or MIS-LABELLED figure (the distance was a hand-copied legend label, not read from the data).
#
# Selection: among REHDs of the SAME hardware class (0.7 V: vmin 0.64 / vmax 0.738, so the traces share one y-axis), take the 2 nearest and 2 farthest by the distance the simulator actually recorded.
# That is exactly the figure's claim -- near harvesters hover at Vmax, far ones at Vmin.
#
# MIN_SEP: skip a candidate that duplicates an already-picked distance.
# Two harvesters often land within a few cm of each other (here 1.78 m and 1.79 m); taking both spends a curve on a redundant trace AND prints two identical "d = 1.8 m" legend entries.
# The runner-up (2.65 m) is the informative mid-annulus case -- still climbing at the end of the window -- so the dedup buys a real curve for free.
#
# EXPECT_D pins the outcome.
# The pool is deterministic (ns-3 is deterministic per seed; REHD type + position draw from fixed RNG streams), but it silently reshuffles if anyone changes nStations/nRehd, the RNG stream ids, REHD_TYPE_TEMPLATES, the annulus constants, or the step count -- and a reshuffled pool yields a DIFFERENT figure that still looks plausible.
# Assert instead: drift becomes a hard failure, not a silent swap of the paper's figure.
base = os.environ["EX"] + "/results/data-log"
def ff(x, d=0.0):
    try: return float(x)
    except Exception: return d

cand = {}   # (sim, sta) -> (dist, [(t, v)])
for path in sorted(glob.glob(f"{base}/vcap_trace_*.csv")):
    sim = os.path.basename(path).split("_")[-1].split(".")[0]
    for r in csv.DictReader(open(path)):
        if not r.get("sta_id") or not r.get("vcap_v"):
            continue
        if abs(ff(r.get("vmin_v")) - 0.64) > 1e-6:      # 0.7 V class only
            continue
        v = ff(r["vcap_v"])
        if v <= 0:
            continue
        key = (sim, int(r["sta_id"]))
        cand.setdefault(key, [ff(r.get("dist_m")), []])[1].append(
            (ff(r["time_ms"]) / 1000.0, v))

if not cand:
    sys.exit("[vcap] ERROR: no 0.7 V-class REHD samples found in any vcap trace")
MIN_SEP  = 0.05                                  # m; treat closer than this as duplicates
EXPECT_D = [1.78, 2.65, 4.28, 4.42]              # the submitted figure, seeds 42-45
TOL      = 0.01                                  # m

ordered = sorted(cand.items(), key=lambda kv: kv[1][0])
near, far = [], []
for item in ordered:                                       # nearest first
    if len(near) == 2:
        break
    if all(abs(item[1][0] - p[1][0]) >= MIN_SEP for p in near):
        near.append(item)
for item in reversed(ordered):                             # farthest first
    if len(far) == 2:
        break
    if all(abs(item[1][0] - p[1][0]) >= MIN_SEP for p in near + far):
        far.append(item)
picks = near + far[::-1]                                   # ascending by distance

got = [round(d, 2) for _, (d, _) in picks]
if len(picks) < 4:
    sys.exit(f"[vcap] ERROR: only {len(picks)} distinct-distance REHDs in the "
             f"0.7 V pool (need 4); pool size was {len(ordered)}")
if any(abs(g - e) > TOL for g, e in zip(got, EXPECT_D)):
    sys.exit(f"[vcap] ERROR: candidate pool drifted -- picked {got}, expected "
             f"{EXPECT_D}.\n[vcap] The scenario behind the paper figure changed "
             f"(nStations/nRehd, RNG streams, REHD templates, annulus, or step "
             f"count).\n[vcap] Re-pin EXPECT_D only after confirming the new "
             f"figure is the one you want.")
rows = []
for (sim, sta), (dist, tv) in picks:
    print(f"[vcap]   sim {sim} sta {sta:>2}  d={dist:.2f} m  ({len(tv)} samples)")
    rows += [(dist, t, v) for t, v in tv]

with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dist_m", "time_s", "vcap_v"])
    w.writerows([(f"{d:.2f}", f"{t:.3f}", f"{v:.4f}") for d, t, v in rows])
print(f"[vcap] wrote {sys.argv[1]}: {len(rows)} rows from {len(picks)} REHDs")
PYV
    # set -e is NOT in effect here, so a failed extraction would otherwise fall through to stage 4 and plot whatever stale vcap_data.csv happens to be lying around --
    # i.e. silently ship the WRONG figure, the exact failure the guard exists to stop.
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[vcap] extraction FAILED (rc=$rc) -- aborting before stage 4 can plot a stale figure"
        exit "$rc"
    fi
fi

# ----------------------------- 4. figures ------------------------------------
if has 4; then
    banner 4 "generate all paper figures (make_figs.py)"
    CKPT=$(ls -t "$PIPE_ROOT"/train/*/ckpt_final.pt 2>/dev/null | head -1)
    EVAL_SHARDS="$PIPE_ROOT/eval/shards"
    # Point make_figs at THIS reproduction run (env overrides; falls back to shipped run).
    [ -d "$EVAL_SHARDS" ] && export PAPER_EVAL_DIR="$EVAL_SHARDS"
    [ -n "$CKPT" ] && export PAPER_TRAIN_GLOB="$(dirname "$CKPT")/training_log.jsonl"
    # A run that skipped stage 3 (e.g. the committed repro_20260825) has no vcap_data.csv of its own; fall back to the committed one.
    if [ ! -f "$VCAP_DATA" ] && [ -f "$RESULTS/figs/vcap_data.csv" ]; then
        echo "[figs] no $VCAP_DATA — using the committed results/figs/vcap_data.csv"
        VCAP_DATA="$RESULTS/figs/vcap_data.csv"
    fi
    export PAPER_VCAP_DATA="$VCAP_DATA"
    export PAPER_FIGS_DIR="$FIGS_DIR"
    mkdir -p "$FIGS_DIR"
    "$PY" "$SCR/make_figs.py"
    echo "  (system_model.jpg and beacon_interval.jpg are hand-drawn — not generated here)"
fi

echo
echo "==================== reproduce.sh DONE ===================="
date
echo "run artifacts: $RUN  (core engine: run_pipeline.sh -> $PIPE_ROOT)"
echo "figures:       $FIGS_DIR"
