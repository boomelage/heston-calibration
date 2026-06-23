"""Verify that a calibration run covers every raw trading day exactly once.

For the (MODEL, OBJECTIVE) selected in ``_results_config``, compare the dates encoded in the raw
trades filenames under ``RAW`` against the ``date`` column of ``calibrations.csv`` and
``rejections.csv``. Every raw date must resolve to **exactly one** result row, in either the
accepted file or the rejected file (an attempted day is always one or the other, never both,
never duplicated). The two files together are meant to partition the attempted days, so this is
the audit that the partition is clean.

Discrepancies reported:
  - missing   : a raw date with no calibration AND no rejection row.
  - in_both   : a raw date that appears in calibrations.csv *and* rejections.csv.
  - duplicate : a raw date that appears more than once within a single file.
  - orphan    : a result row whose date has no raw file (e.g. raw files deleted after the run;
                RAW is git-ignored, so this is common on a fresh clone).

Read-only. Prints a human-readable report to stdout and exits 1 if any discrepancy is found,
else exits 0 (usable in CI / pre-commit).
"""
import sys
import pandas as pd
from pathlib import Path

RESULTS_CODE = Path(__file__).parent.resolve()
SRC = RESULTS_CODE.parent
REPO = SRC.parent
RESULTS = REPO / "results"
RAW = REPO / "data" / "options" / "raw"

for _p in (str(SRC), str(RESULTS_CODE), str(RESULTS), str(RAW)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _utils import _file_date
from config import calib_paths
from _results_config import MODEL, OBJECTIVE

CALIBRATIONS_FILE, REJECTIONS_FILE, TESTS = calib_paths(MODEL, OBJECTIVE)


def _raw_dates():
    """The set of date tokens encoded in the raw trades filenames (one per trading day)."""
    if not RAW.exists():
        return []
    return [_file_date(p) for p in RAW.glob("*.csv")]


def _result_dates(path):
    """The list of `date` values in a results CSV, as strings. Missing/empty file -> []."""
    if not path.exists():
        return []
    df = pd.read_csv(path, usecols=["date"])
    if df.empty:
        return []
    return df["date"].astype(str).tolist()


def main():
    raw_dates = _raw_dates()
    calib_dates = _result_dates(CALIBRATIONS_FILE)
    reject_dates = _result_dates(REJECTIONS_FILE)

    calib_counts = pd.Series(calib_dates, dtype=object).value_counts()
    reject_counts = pd.Series(reject_dates, dtype=object).value_counts()
    result_set = set(calib_counts.index) | set(reject_counts.index)

    # Only audit the range processed so far: clip raw dates to the latest date present in either
    # results file. Days beyond that frontier have not been attempted yet, so flagging them missing
    # would be noise. ISO `YYYY-MM-DD` strings sort lexicographically, so max() gives the last date.
    last_date = max(result_set) if result_set else None
    raw_set = {d for d in raw_dates if last_date is None or d <= last_date}

    def c(date):
        return int(calib_counts.get(date, 0))

    def r(date):
        return int(reject_counts.get(date, 0))

    # ---- forward direction: every raw date resolves to exactly one result row ----
    missing, in_both, duplicate = [], [], []
    for date in sorted(raw_set):
        nc, nr = c(date), r(date)
        if nc + nr == 0:
            missing.append(date)
            continue
        if nc >= 1 and nr >= 1:
            in_both.append((date, nc, nr))
        if nc > 1 or nr > 1:
            duplicate.append((date, nc, nr))

    # ---- reverse direction: result rows with no raw file ----
    orphan = sorted(d for d in result_set if d not in raw_set)
    # An orphan can also be duplicated/in-both; surface those facts there too.
    for date in orphan:
        nc, nr = c(date), r(date)
        if nc >= 1 and nr >= 1:
            in_both.append((date, nc, nr))
        if nc > 1 or nr > 1:
            duplicate.append((date, nc, nr))
    in_both = sorted(set(in_both))
    duplicate = sorted(set(duplicate))

    # ---- report ----
    print(f"model={MODEL}  objective={OBJECTIVE}")
    print(f"raw dir:        {RAW}")
    print(f"calibrations:   {CALIBRATIONS_FILE}")
    print(f"rejections:     {REJECTIONS_FILE}")
    print(f"audited range:  up to {last_date or '(no results yet)'} "
          f"(raw files beyond this are not yet processed; ignored)")
    print(f"raw files:      {len(raw_dates)} total, {len(raw_set)} in range")
    print(f"calibrated:     {len(calib_dates)} rows ({len(calib_counts)} distinct dates)")
    print(f"rejected:       {len(reject_dates)} rows ({len(reject_counts)} distinct dates)")

    if not raw_set:
        print("\nWARNING: no raw files found (RAW is git-ignored; a fresh clone has none). "
              "Cannot verify the forward direction.")

    def _block(title, items, fmt):
        if not items:
            return
        print(f"\n{title} ({len(items)}):")
        for it in items:
            print(f"  {fmt(it)}")

    _block("MISSING (no calibration and no rejection)", missing, lambda d: d)
    _block("IN BOTH (present in calibrations.csv AND rejections.csv)", in_both,
           lambda t: f"{t[0]}  (calib={t[1]}, reject={t[2]})")
    _block("DUPLICATE (>1 row in a single file)", duplicate,
           lambda t: f"{t[0]}  (calib={t[1]}, reject={t[2]})")
    _block("ORPHAN (result row with no raw file)", orphan, lambda d: d)

    n_issues = len(missing) + len(in_both) + len(duplicate) + len(orphan)
    if n_issues == 0 and raw_set:
        print(f"\nOK: all {len(raw_set)} raw dates have exactly one result row.")
        return 0
    print(f"\nFAIL: {n_issues} discrepancy group(s) "
          f"(missing={len(missing)}, in_both={len(in_both)}, "
          f"duplicate={len(duplicate)}, orphan={len(orphan)}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
