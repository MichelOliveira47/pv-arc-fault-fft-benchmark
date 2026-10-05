#!/usr/bin/env bash
set -euo pipefail

export PV_ARC_RUN_MODE=quick_check
export PV_ARC_WINDOW_LENGTH_SWEEP=0
python scripts/check_input_data.py
python src/pv_arc_fault_fft_benchmark.py
