#!/usr/bin/env bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

# --- setup_fresh_ns3.sh ---
# One-shot setup after dropping this repo into a FRESH ns-3.44 + ns3-ai checkout, under contrib/ai/examples/rl-twt-powercast/.
#
# The directory name is fixed (see the check below): twt-constants.h and CMakeLists.txt hardcode "rl-twt-powercast", so this script refuses to register any other name.
#
# It (1) applies the NS-3 source patches this project depends on (BSR manager
# singleton + TWT agreement plumbing) and (2) registers the build subdir in the
# parent CMakeLists. Then create a venv + pip install -r requirements.txt and
# build (see README.md's Build section). Idempotent (safe to re-run).
#
# Prereqs you provide: ns-3.44 source + the ns3-ai contrib module installed.
#
# Usage:
#   bash contrib/ai/examples/rl-twt-powercast/setup_fresh_ns3.sh   # from the NS-3 root (any cwd works)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> [1/3] NS-3 patch: BSR manager singleton (real-time BSR for TWT control)..."
bash "$HERE/mod-files/install_bsr_manager.sh"

echo "==> [2/3] NS-3 patch: TWT agreement plumbing (WifiMac::SetTwtSchedule)..."
bash "$HERE/mod-files/twt-complete-setup.sh"

# CANONICAL directory name. This repo must be cloned as "rl-twt-powercast":
#     git clone <repo-url> rl-twt-powercast
# The name is hardcoded here and in twt-constants.h so every path is fixed and
# predictable. We still CHECK it rather than assume it: registering a subdirectory
# that does not match the real folder makes cmake silently skip the pybind .so, and
# every Python entry point then dies with ModuleNotFoundError far from the cause.
EXPECTED_DIRNAME="rl-twt-powercast"
DIRNAME="$(basename "$HERE")"
if [ "$DIRNAME" != "$EXPECTED_DIRNAME" ]; then
    echo "ERROR: this directory is named '$DIRNAME', but the project requires"
    echo "       '$EXPECTED_DIRNAME'. Paths are hardcoded to that name."
    echo "       Fix:  mv '$HERE' '$(dirname "$HERE")/$EXPECTED_DIRNAME'"
    echo "       then re-run this script."
    exit 1
fi
echo "==> [3/3] Register add_subdirectory($DIRNAME) in the parent CMakeLists..."
PARENT="$HERE/../CMakeLists.txt"
if [ -f "$PARENT" ] && ! grep -q "^add_subdirectory($DIRNAME)$" "$PARENT"; then
    echo "add_subdirectory($DIRNAME)" >>"$PARENT"
    echo "    added to $PARENT"
elif [ -f "$PARENT" ]; then
    echo "    already present in $PARENT"
else
    echo "    WARNING: $PARENT not found — manually add 'add_subdirectory($DIRNAME)'"
    echo "    to contrib/ai/examples/CMakeLists.txt"
fi

cat <<'NEXT'

==> Patches applied + subdir registered. NEXT STEPS:
    1) Create the venv:   python3.11 -m venv <NS3_ROOT>/../../EHRL
                          source <NS3_ROOT>/../../EHRL/bin/activate
    2) Deps:              pip install -r requirements.txt
                          (the CPU build of torch is enough)
    3) Build:             see README.md "Build" — PIN an absolute Python3_EXECUTABLE
                          to the venv python (the silent 3.x-relink trap). Verify from the NS-3 root:
                          PYTHONPATH=contrib/ai/examples/rl-twt-powercast python3.11 -c "import pb_twt_powercast_interface_py"
    4) Sanity test:       python3.11 test-scripts/twt_smoke_one_worker.py
                          python3.11 test-scripts/twt_stress_test.py
NEXT
