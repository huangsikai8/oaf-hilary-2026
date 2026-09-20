"""
Extends the options sample beyond the original 2023-2025 window.

Deribit lists daily BTC expiries from January 2021, and its trade history serves
them through to the present, so the sample is not limited to 2023. This script
collects only the two ranges the existing data/options_df.csv does not cover and
merges them with it, leaving the original file untouched.

Writes:
    data/options_df_ext.csv   original rows plus the two new ranges
    data/sessions_ext.csv     session windows spanning the full period

Collection runs in chunks and checkpoints after each one, so an interrupted run
resumes from where it stopped rather than starting over.

    python collect_extended.py
"""

import os
import sys

import pandas as pd

import options_data_collection as odc

FULL_START = "2021-01-01"
FULL_END = "2026-09-08"        # last session opens 2026-09-07, expiring 2026-09-08

GAPS = [("2021-01-01", "2023-01-01"), ("2026-01-01", FULL_END)]

CHUNK_SESSIONS = 60
CHECKPOINT = "data/options_df_new_partial.csv"


def main():
    btc = pd.read_csv("data/btc_prices.csv", index_col=0, parse_dates=True)
    btc.index = pd.to_datetime(btc.index, utc=True)
    btc = btc[~btc.index.duplicated(keep="first")]
    print(f"spot: {btc.index.min()} -> {btc.index.max()} ({len(btc)} rows)", flush=True)

    existing = pd.read_csv("data/options_df.csv")
    existing["session_start"] = pd.to_datetime(existing["session_start"], utc=True)
    have = set(existing["session_start"])
    print(f"existing options: {len(existing)} legs, {len(have)} sessions", flush=True)

    todo = []
    for start, end in GAPS:
        s = odc.generate_session_windows(start, end)
        s = s[~s["session_start"].isin(have)]
        todo.append(s)
    todo = pd.concat(todo, ignore_index=True).sort_values("session_start")

    done = set()
    if os.path.exists(CHECKPOINT):
        part = pd.read_csv(CHECKPOINT)
        part["session_start"] = pd.to_datetime(part["session_start"], utc=True)
        done = set(part["session_start"])
        todo = todo[~todo["session_start"].isin(done)]
        print(f"resuming: {len(done)} sessions already collected", flush=True)

    todo = todo.reset_index(drop=True)
    print(f"to collect: {len(todo)} sessions\n", flush=True)

    for i in range(0, len(todo), CHUNK_SESSIONS):
        chunk = todo.iloc[i : i + CHUNK_SESSIONS].reset_index(drop=True)
        lo, hi = chunk["session_start"].min(), chunk["session_start"].max()
        print(
            f"\n{'=' * 70}\nCHUNK {i // CHUNK_SESSIONS + 1}"
            f"  {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}  ({len(chunk)} sessions)\n{'=' * 70}",
            flush=True,
        )
        got = odc.collect_options_data(chunk, btc)
        if got.empty:
            print("  chunk returned nothing", flush=True)
            continue
        header = not os.path.exists(CHECKPOINT)
        got.to_csv(CHECKPOINT, mode="a", header=header, index=False)
        print(f"  checkpointed {len(got)} legs", flush=True)

    # ---- merge ----
    if not os.path.exists(CHECKPOINT):
        print("nothing new collected", flush=True)
        return

    new = pd.read_csv(CHECKPOINT)
    new["session_start"] = pd.to_datetime(new["session_start"], utc=True)
    merged = pd.concat([existing, new], ignore_index=True)
    merged["session_start"] = pd.to_datetime(merged["session_start"], utc=True)
    merged["session_close"] = pd.to_datetime(merged["session_close"], utc=True)
    merged["expiry"] = pd.to_datetime(merged["expiry"], utc=True)
    merged = (
        merged.drop_duplicates(subset=["session_start", "leg"])
        .sort_values(["session_start", "leg"])
        .reset_index(drop=True)
    )
    merged.to_csv("data/options_df_ext.csv", index=False)

    sess = odc.generate_session_windows(FULL_START, FULL_END)
    sess = sess[sess["session_start"].isin(set(merged["session_start"]))]
    sess.to_csv("data/sessions_ext.csv", index=False)

    n_full = merged.groupby("session_start").size().eq(4).sum()
    print(
        f"\nwrote data/options_df_ext.csv: {len(merged)} legs, "
        f"{merged['session_start'].nunique()} sessions "
        f"({n_full} with all four legs)\n"
        f"span {merged['session_start'].min():%Y-%m-%d} "
        f"-> {merged['session_start'].max():%Y-%m-%d}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
