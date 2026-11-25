# Sachin's AVID verification work so far (for Tirtha)

Everything here is pinned to repo commit 54da68d (AVID_EPJ_Patches). The one-round test case (round 208) and the first Stim demo are also in PR #1 on the repo (verification/sachin/). This package is the full set.

## Folders
- analysis_scripts/  all models, runners, figure scripts, plus the CSVs and logs they produced
- daemon/            batch runners, scenario patch (diff, not applied), RL run JSON and results CSV
- paper/ (draft): not included here, shared separately

## Rough map from topic to files (by file name; check each script's header before relying on it)
- Alias trap, hot pair vs hot single:  run_b.py, run_trap.py, results_b.csv, trap_curve.csv, avid_stim_crosscheck.log
- Model, Stim circuit, decoders:       avid_model.py, avid_stim.py, avid_decoders.py, avid_circuit.py
- Tracking / commit delay:             avid_tracking_latency.log, run_a1.py, run_a2.py, results_a2.csv, breakeven_a.csv
- Circuit level (readout error):       avid_robustness_circuit.log, run_c.py, results_c.csv
- Calibration fix:                     avid_fix.py, run_fix.py, run_fix_circuit.py, run_fix_w1024.py and their results_fix*.csv
- X/Z coupling:                        run_xz.py, results_xz.csv
- Distance 5 (code capacity only):     d5_*.py, run_d5*.py, results_d5*.csv
- Repo validation:                     check_cpp_estimator.py (C++ projection vs Python port), check_daemon_vs_model*.py, daemon/
- Sweeps:                              run_sweep.py, results_sweep.csv, sweep_breakeven.csv
- Files with _v1_ or "discarded" in the name are older runs kept for the record, not current results.

## Setup
    pip install stim pymatching numpy scipy pandas matplotlib
Scripts that touch the repo need a checkout of 54da68d; see analysis_scripts/README.md for the Stim demo run order.

## Open items (nothing below is claimed as done)
- Your updated weights and a held-out split for the replay: not received, no results.
- --stim-bin possibly reads ground truth from bits 32-49 of the packet, where the codec writes the sequence number (round 208: bits hold 0xd0, true mask 0x008). Please confirm.
- Static and avid_r2d2 both score 502/2000 on test_traj_00_s30000 (window 512, delay 20, K=3) even though avid_r2d2 committed weight swaps. Not investigated.
- Not run: d=5 circuit level, hardware commit-delay and X/Z coupling measurements, calibration fix combined with the tracking model, controlled RL vs simpler controller on single-qubit hotspots, Verilator RTL run of round 208.
- Audit: 204 pytest pass and the -31.2% to -45.9% result reproduces. "36/36 suites" and the RL-vs-joint_dp_proj benefit could not be confirmed.
