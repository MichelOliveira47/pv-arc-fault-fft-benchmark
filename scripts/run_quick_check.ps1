$ErrorActionPreference = "Stop"
$env:PV_ARC_RUN_MODE = "quick_check"
$env:PV_ARC_WINDOW_LENGTH_SWEEP = "0"
python scripts/check_input_data.py
python src/pv_arc_fault_fft_benchmark.py
