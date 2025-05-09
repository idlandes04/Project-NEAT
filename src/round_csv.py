#!/usr/bin/env python3
"""
round_csv.py
------------

Floor every numeric value in a CSV down to (at most) three significant
digits.  Non‑numeric cells are left untouched.

Usage
-----
    python round_csv.py input.csv            # ⇒ creates input_rounded.csv
    python round_csv.py input.csv output.csv # ⇒ writes to output.csv
"""

import csv
import math
import os
import sys


def floor_to_sig(x: float, sig: int = 3) -> float:
    """Return x rounded *down* to `sig` significant digits."""
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    x = abs(x)

    exp = math.floor(math.log10(x))
    factor = 10 ** (exp - sig + 1)
    return sign * math.floor(x / factor) * factor


def try_parse_number(text: str):
    """Try to parse text as a float; return (value, True) or (text, False)."""
    try:
        return float(text), True
    except ValueError:
        return text, False


def process_csv(src: str, dst: str, sig: int = 3) -> None:
    with open(src, newline="") as f_in, open(dst, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        for row in reader:
            processed = []
            for cell in row:
                val, is_num = try_parse_number(cell)
                if is_num:
                    rounded = floor_to_sig(val, sig)
                    # keep integer formatting if appropriate
                    processed.append(str(int(rounded)) if rounded.is_integer()
                                      else repr(rounded))
                else:
                    processed.append(cell)
            writer.writerow(processed)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python round_csv.py <input.csv> [output.csv]")

    src_path = sys.argv[1]
    dst_path = (sys.argv[2] if len(sys.argv) > 2
                else f"{os.path.splitext(src_path)[0]}_rounded.csv")

    process_csv(src_path, dst_path)
    print(f"Finished: wrote rounded data to {dst_path}")
