#!/usr/bin/env python3
"""Generate pandas/numpy fixtures for Java DataFrame interop benchmarks.

Usage:
  python3 samples/python/gen_interop_fixtures.py --out samples/fixtures

Produces:
  tsv, csv, excel, pickle (records + to_pickle), hdf5 (h5py columnar),
  avro (fastavro if available), npz, parquet/feather (pyarrow if available).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("samples/fixtures"))
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("pandas/numpy required", file=sys.stderr)
        return 1

    df = pd.DataFrame(
        {
            "id": np.array([1, 2, 3, 4], dtype=np.int64),
            "name": ["alice", "bob", "carol", "dave"],
            "score": [9.5, 7.0, 8.25, 6.0],
            "ok": [True, False, True, False],
        }
    )

    df.to_csv(out / "pandas.csv", index=False)
    df.to_csv(out / "pandas.tsv", sep="\t", index=False, na_rep="\\N")
    try:
        df.to_excel(out / "pandas.xlsx", index=False)
        print("wrote pandas.xlsx")
    except Exception as e:
        print("skip excel:", e)

    # Portable records pickle (Java SELF_DESC / RECORDS friendly)
    records = df.to_dict(orient="records")
    with open(out / "pandas_records.pkl", "wb") as f:
        pickle.dump(records, f, protocol=4)

    self_desc = {
        "__pandas_dataframe__": True,
        "columns": list(df.columns),
        "dtypes": ["INT64", "STRING", "FLOAT64", "BOOLEAN"],
        "data": records,
        "orient": "records",
    }
    with open(out / "pandas_self_desc.pkl", "wb") as f:
        pickle.dump(self_desc, f, protocol=4)
    print("wrote pickle fixtures")

    # Native pandas pickle (may require Java allow-list; best-effort)
    try:
        df.to_pickle(out / "pandas_native.pkl")
        print("wrote pandas_native.pkl")
    except Exception as e:
        print("skip native pickle:", e)

    # NPZ
    np.savez(
        out / "pandas.npz",
        id=df["id"].to_numpy(),
        score=df["score"].to_numpy(),
    )
    print("wrote pandas.npz")

    # HDF5 via h5py columnar (matches our writer layout loosely)
    try:
        import h5py

        with h5py.File(out / "pandas_columnar.h5", "w") as h5:
            g = h5.create_group("df")
            g.attrs["format"] = "columnar"
            g.attrs["column_names"] = np.array(["id", "name", "score", "ok"], dtype="S")
            g.create_dataset("id", data=df["id"].to_numpy())
            # variable-length utf-8 strings
            dt = h5py.string_dtype(encoding="utf-8")
            g.create_dataset("name", data=np.array(df["name"].tolist(), dtype=dt))
            g.create_dataset("score", data=df["score"].to_numpy())
            g.create_dataset("ok", data=df["ok"].to_numpy())
        print("wrote pandas_columnar.h5")
    except Exception as e:
        print("skip h5py:", e)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.feather as feather

        table = pa.Table.from_pandas(df)
        pq.write_table(table, out / "pandas.parquet")
        feather.write_feather(table, out / "pandas.feather")
        print("wrote parquet/feather")
    except Exception as e:
        print("skip pyarrow:", e)

    try:
        from fastavro import writer, parse_schema

        schema = {
            "type": "record",
            "name": "DataFrame",
            "fields": [
                {"name": "id", "type": ["null", "long"], "default": None},
                {"name": "name", "type": ["null", "string"], "default": None},
                {"name": "score", "type": ["null", "double"], "default": None},
                {"name": "ok", "type": ["null", "boolean"], "default": None},
            ],
        }
        parsed = parse_schema(schema)
        with open(out / "pandas.avro", "wb") as f:
            writer(f, parsed, records)
        print("wrote pandas.avro")
    except Exception as e:
        print("skip fastavro:", e)

    # SQLite
    try:
        import sqlite3

        db = out / "pandas.db"
        if db.exists():
            db.unlink()
        conn = sqlite3.connect(db)
        df.to_sql("t", conn, index=False, if_exists="replace")
        conn.close()
        print("wrote pandas.db")
    except Exception as e:
        print("skip sqlite:", e)

    print("fixtures at", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
