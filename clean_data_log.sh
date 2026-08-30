#!/usr/bin/env bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

# clean_data_log.sh — housekeeping for results/data-log/.
#
# The NS-3 sim writes a fresh file into results/data-log/ on (almost) every run — most notably device_class_assignments_<ts>.txt (the per-run STA config), plus opt-in Vcap/trace CSVs and the plot PNGs.
# None of it is source (the directory is gitignored scratch output); over many runs it piles up.
# This clears that debris but always keeps the directory itself in place.
#
# PATH: the sim writes via TwtResultsPath(), which roots everything generated at <example>/results/ — so the target is results/data-log/, NOT data-log/.
# This script pointed at the latter, a path that has never existed, so every invocation printed "nothing to clean" and exited 0 while the pile kept growing.
#
# Usage:
#   bash clean_data_log.sh            # delete known generated run-output files
#   bash clean_data_log.sh --all      # wipe everything in results/data-log/
#   bash clean_data_log.sh --keep 5   # keep the 5 newest matching files, delete the rest
#   bash clean_data_log.sh -n         # dry run: list what WOULD be deleted
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLOG="$HERE/results/data-log"

DRY=0
ALL=0
KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        -n | --dry-run) DRY=1 ;;
        --all) ALL=1 ;;
        --keep)
            KEEP="${2:-0}"
            shift
            ;;
        # Skip the shebang and the license header block (through the first blank line),
        # then print the usage comment block that follows, stopping at the first
        # non-comment line. Keeps --help working no matter how long the header gets.
        -h | --help)
            awk -v skip=1 'NR==1 {next} skip && /^$/ {skip=0; next} skip {next}
                       /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1 (try --help)"
            exit 1
            ;;
    esac
    shift
done

[ -d "$DLOG" ] || {
    echo "no results/data-log/ at $DLOG — nothing to clean"
    exit 0
}

# Every filename pattern the sim / plots emit into results/data-log/.
PATTERNS=(
    'device_class_assignments_*.txt'
    'vcap_trace_*.csv'
    'twt-controller-log.csv'
    'ns3-*-trace-*.csv' # e2e / macqueuesize / ampdu / bsr / phystate / aptx / txrx-stats / link / timeout
    'ns3-twt-wrapper-*'
    'e2e_trace_*' 'queue_size_trace_*' 'ampdu_trace_*' 'bsr_trace_*'
    'py-wrapper-*.csv'
    '*.png'
)

cd "$DLOG"
if [ "$ALL" = 1 ]; then
    mapfile -t TARGETS < <(find . -maxdepth 1 -type f ! -name '.gitkeep' -printf '%f\n' | sort -u)
else
    mapfile -t TARGETS < <(for p in "${PATTERNS[@]}"; do
        find . -maxdepth 1 -type f -name "$p" -printf '%f\n'
    done | sort -u)
fi

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "results/data-log/ already clean (0 matching files)"
    exit 0
fi

# Sort the targets newest-first by mtime, then drop the KEEP newest from deletion.
mapfile -t DEL < <(ls -1t "${TARGETS[@]}" | tail -n +"$((KEEP + 1))")

if [ "${#DEL[@]}" -eq 0 ]; then
    echo "nothing to delete (kept all ${#TARGETS[@]} of them via --keep $KEEP)"
    exit 0
fi

echo "results/data-log/: ${#TARGETS[@]} matching file(s); deleting ${#DEL[@]} (keeping newest $KEEP)"
for f in "${DEL[@]}"; do
    if [ "$DRY" = 1 ]; then echo "  would remove  $f"; else rm -f -- "$f"; fi
done
if [ "$DRY" = 1 ]; then echo "(dry run — nothing deleted)"; else echo "removed ${#DEL[@]} file(s) from results/data-log/"; fi
