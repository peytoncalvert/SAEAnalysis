"""
main.py  —  Run all experiment phases
======================================
Usage:
    python main.py          # run all three phases
    python main.py 1        # run only phase 1
    python main.py 1 2      # run phases 1 and 2
"""

import sys
from phase1 import run_phase1
from phase2 import run_phase2
from phase3 import run_phase3

PHASES = {
    "1": run_phase1,
    "2": run_phase2,
    "3": run_phase3,
}

if __name__ == "__main__":
    args = sys.argv[1:] or ["1", "2", "3"]
    for arg in args:
        if arg not in PHASES:
            print(f"Unknown phase '{arg}'. Choose from: 1, 2, 3")
            sys.exit(1)
    for arg in args:
        PHASES[arg]()
