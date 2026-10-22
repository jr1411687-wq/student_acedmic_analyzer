import streamlit as st
import pandas as pd
import pickle
import os
import json
from pathlib import Path

from analyzer.catalog import STUDY_YEAR_UNKNOWN, YEAR_UNKNOWN, _disk_cache_key, apply_majority_batch_year, collapse_exam_attempts, infer_exam_type, parse_dataset_meta
from analyzer.constants import FRAME_CACHE_DIR
from analyzer.files import _iter_data_files, discover_data_files
from analyzer.gender import get_gender_detector
from analyzer.parsers import load_dataset_smart

DATASETS = discover_data_files()


@st.cache_data(ttl=20)
def _data_mtime_signature():
    files = _iter_data_files("*.csv") + _iter_data_files("*.pdf")
    return tuple(sorted((str(p.resolve()), p.stat().st_mtime) for p in files))


@st.cache_data
def build_dataset_catalog(_mtime_signature, _batch_rule=2):
    """Catalog of every CSV/PDF with Branch, Semester, Year metadata."""
    catalog = []
    for label, file_path in discover_data_files().items():
        catalog.append(parse_dataset_meta(file_path, label=label))
    return catalog


# =========================================================
# FUNCTION TO PROCESS CSV WITH GENDER
# =========================================================

def process_csv_with_gender(df):
    """Process CSV and add gender column using auto-detected name column."""
    if "SGPA" in df.columns:
        df["SGPA"] = pd.to_numeric(df["SGPA"], errors="coerce")

    if "Credits_Earned" in df.columns:
        df["Credits_Earned"] = pd.to_numeric(df["Credits_Earned"], errors="coerce")

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip()
        df["Gender"] = df["Gender"].replace({
            'M': 'Male', 'F': 'Female', 'm': 'Male', 'f': 'Female',
            'Male': 'Male', 'Female': 'Female', 'male': 'Male', 'female': 'Female'
        })
    elif "Name" in df.columns:
        detector = get_gender_detector()
        names = df["Name"].astype(str).tolist()
        df["Gender"] = [detector.detect_gender(n) for n in names]

    return df

# =========================================================
# LOAD DATA
# =========================================================

def _read_frame_disk_cache(file_path, file_mtime):
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _disk_cache_key(file_path, file_mtime)
    pq_path = FRAME_CACHE_DIR / f"{key}.parquet"
    meta_path = FRAME_CACHE_DIR / f"{key}.json"
    if not (pq_path.exists() and meta_path.exists()):
        return None
    try:
        df = pd.read_parquet(pq_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return df, meta.get("subjects", [])
    except Exception:
        return None


def _write_frame_disk_cache(file_path, file_mtime, df, subject_cols):
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _disk_cache_key(file_path, file_mtime)
    pq_path = FRAME_CACHE_DIR / f"{key}.parquet"
    meta_path = FRAME_CACHE_DIR / f"{key}.json"
    try:
        df.to_parquet(pq_path, index=False)
        meta_path.write_text(
            json.dumps({"subjects": list(subject_cols), "rows": int(len(df)), "format": "parquet"}),
            encoding="utf-8",
        )
        return
    except Exception:
        pass
    # Fallback: pickle when parquet can't store a column type
    try:
        pkl_path = FRAME_CACHE_DIR / f"{key}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump((df, list(subject_cols)), f, protocol=pickle.HIGHEST_PROTOCOL)
        meta_path.write_text(
            json.dumps({"subjects": list(subject_cols), "format": "pkl"}),
            encoding="utf-8",
        )
    except Exception:
        pass


def _read_frame_disk_cache_any(file_path, file_mtime):
    cached = _read_frame_disk_cache(file_path, file_mtime)
    if cached is not None:
        return cached
    key = _disk_cache_key(file_path, file_mtime)
    pkl_path = FRAME_CACHE_DIR / f"{key}.pkl"
    meta_path = FRAME_CACHE_DIR / f"{key}.json"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                df, subjects = pickle.load(f)
            return df, subjects
        except Exception:
            return None
    return None


def _parse_file_to_frame(file_path, file_mtime):
    """Parse one file with disk cache (safe to call from worker threads)."""
    cached = _read_frame_disk_cache_any(file_path, file_mtime)
    if cached is not None:
        return cached
    df, subject_cols = load_dataset_smart(file_path)
    if df is not None and not df.empty:
        df = process_csv_with_gender(df)
        _write_frame_disk_cache(file_path, file_mtime, df, subject_cols)
    return df, subject_cols


@st.cache_data(show_spinner=False)
def load_data(file_path, _file_mtime):
    """Load one CSV/PDF with gender tagging. Disk-cached across Streamlit restarts."""
    return _parse_file_to_frame(file_path, _file_mtime)


@st.cache_data(show_spinner=False)
def load_all_branch_datasets(_mtime_signature):
    """Load every CSV/PDF in the data folder and tag Branch / Semester / Year."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    frames = []
    load_errors = []
    catalog = {c["path"]: c for c in build_dataset_catalog(_mtime_signature)}
    items = list(discover_data_files().items())

    def _one(label_path):
        label, file_path = label_path
        file_mtime = os.path.getmtime(file_path)
        df, _ = _parse_file_to_frame(file_path, file_mtime)
        return label, file_path, df

    # Parallel first-time PDF parse (huge win on cold start with many large PDFs)
    max_workers = min(4, max(1, len(items)))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, item) for item in items]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                load_errors.append(str(e))

    for label, file_path, df in results:
        try:
            if df is None or df.empty:
                load_errors.append(f"{label}: empty / unparseable")
                continue
            meta = catalog.get(str(Path(file_path).resolve()))
            if meta is None:
                meta = parse_dataset_meta(file_path, label=label)
            tagged = df.copy()
            tagged["Branch"] = meta["branch"]
            tagged["Dataset"] = label
            tagged["Semester"] = meta["semester"]
            tagged["Year"] = meta.get("exam_year", meta.get("year"))
            tagged["Batch_Year"] = meta.get("batch_year")
            tagged["Exam_Year"] = meta.get("exam_year")
            tagged["Semester_Label"] = meta["semester_label"]
            tagged["Batch_Year_Label"] = meta.get("batch_year_label", YEAR_UNKNOWN)
            tagged["Exam_Year_Label"] = meta.get("exam_year_label", meta.get("year_label", YEAR_UNKNOWN))
            tagged["Study_Year_Label"] = meta.get("study_year_label", STUDY_YEAR_UNKNOWN)
            tagged["Year_Label"] = meta.get("exam_year_label", meta.get("year_label", YEAR_UNKNOWN))
            tagged["Exam_Type"] = meta.get("exam_type") or infer_exam_type(meta.get("filename", label))
            tagged = apply_majority_batch_year(tagged)
            frames.append(tagged)
        except Exception as e:
            load_errors.append(f"{label}: {e}")

    if not frames:
        return pd.DataFrame(), load_errors

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = collapse_exam_attempts(combined)
    return combined, load_errors
