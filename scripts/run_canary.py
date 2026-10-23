"""Run the drift canary. See detector/canary.py for the manifest format."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json

from detector.canary import DEFAULT_MANIFEST, freshness, run_canary, save_run
from detector.pipeline import Pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--set-baseline", action="store_true",
                    help="record this run as the baseline (do this after calibrating)")
    a = ap.parse_args(argv)
    res = run_canary(Pipeline(), a.manifest)
    save_run(res, set_baseline=a.set_baseline)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
