"""Verify a signed JSON forensic report.

    python -m scripts.verify_report report.json

Exit code 0 = intact, 1 = modified or invalid.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import sys

from detector.report_generator import verify_report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    with open(argv[0], "r", encoding="utf-8") as f:
        ok, msg = verify_report(json.load(f))
    print(("OK: " if ok else "FAIL: ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
