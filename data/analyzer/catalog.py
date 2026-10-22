import streamlit as st
import pandas as pd
import hashlib
import re
from pathlib import Path

from analyzer.constants import BRANCH_ALIASES, META_CACHE_DIR

def _label_from_filename(filename):
    """Build a readable label from a CSV/PDF filename."""
    stem = Path(filename).stem
    parts = stem.replace('.', ' ').replace('_', ' ').split()
    cleaned = []
    for p in parts:
        if p.lower() in ('csv', 'pdf', 'regular', 'reappear', 'supplementary', 'accurate', 'result', 'sem'):
            continue
        cleaned.append(p)
    return ' '.join(cleaned) if cleaned else stem


def infer_branch_from_filename(filename):
    """Infer a branch short name from a CSV filename."""
    stem = Path(filename).stem.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ')
    tokens = set(stem.split())

    # Prefer exact short codes first
    for branch, aliases in BRANCH_ALIASES.items():
        for alias in aliases:
            alias_tokens = set(alias.lower().split())
            if alias_tokens.issubset(tokens) or alias.lower() in stem:
                # Avoid mapping "ce" inside "ece" incorrectly: require whole-token for short aliases
                if len(alias) <= 3:
                    if alias.lower() in tokens:
                        return branch
                else:
                    return branch
    return _label_from_filename(filename)


YEAR_UNKNOWN = "Not specified"
SEM_UNKNOWN = "Unknown"
STUDY_YEAR_UNKNOWN = "Not specified"

STUDY_YEAR_FROM_SEM = {
    1: "1st Year",
    2: "1st Year",
    3: "2nd Year",
    4: "2nd Year",
    5: "3rd Year",
    6: "3rd Year",
    7: "4th Year",
    8: "4th Year",
}

STUDY_YEAR_ORDER = ["1st Year", "2nd Year", "3rd Year", "4th Year", STUDY_YEAR_UNKNOWN]


def infer_semester_from_text(text):
    """Extract semester number (1-8) from filename or result header text."""
    if text is None:
        return None
    s = str(text)
    m = re.search(r"Semester\s*:\s*([1-8])\b", s, re.I)
    if m:
        return int(m.group(1))
    # Filenames like cse_sem_4_result — digit may be followed by '_' (not a word boundary)
    m = re.search(r"sem(?:ester)?[\s_-]*([1-8])", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([1-8])(?:st|nd|rd|th)?\s*sem(?:ester)?\b", s, re.I)
    if m:
        return int(m.group(1))
    # GNDEC PDF names: ...Engineering_1_Regular / _3_makeup / _3_Reappear
    m = re.search(
        r"(?:Engineering|Engg)[_\s.-]*([1-8])[_\s.-]*(?:Regular|Reappear|Re-appear|Supplementary|Makeup|Make-up)",
        s,
        re.I,
    )
    if m:
        return int(m.group(1))
    m = re.search(
        r"[_\s.-]([1-8])[_\s.-]*(?:Regular|Reappear|Re-appear|Supplementary|Makeup|Make-up)\b",
        s,
        re.I,
    )
    if m:
        return int(m.group(1))
    return None


def study_year_from_semester(semester):
    """Map semester 1-8 → 1st/2nd/3rd/4th Year."""
    try:
        sem = int(semester)
    except (TypeError, ValueError):
        return STUDY_YEAR_UNKNOWN
    return STUDY_YEAR_FROM_SEM.get(sem, STUDY_YEAR_UNKNOWN)


def infer_exam_year_from_text(text):
    """Extract examination year from result header / filename."""
    if text is None:
        return None
    s = str(text)
    m = re.search(r"Examination\s+Held\s+in\s*:\s*[A-Za-z]+,?\s*(\d{4})", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"Exam(?:ination)?\s*Year\s*:\s*(\d{4})", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(?:May|Nov|November|December|April|June)[,\s]+(\d{4})\b", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"Dated\s*:\s*\d{1,2}[-/]\d{1,2}[-/](\d{4})", s, re.I)
    if m:
        return int(m.group(1))
    name_only = s.split("\n", 1)[0]
    m = re.search(r"(?:exam|result)[\s_-]*(20[2-3]\d)", name_only, re.I)
    if m:
        return int(m.group(1))
    return None


def infer_batch_year_from_text(text):
    """Extract batch/scheme/admission year from header / filename."""
    if text is None:
        return None
    s = str(text)
    m = re.search(r"Scheme\s*Code\s*:\s*(20\d{2})", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(?:batch|admission|scheme|batch[_-]?year)[\s_:-]*(20\d{2})\b", s, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(?:AY|Academic\s*Year)[\s_-]*(20\d{2})\s*[-_/]\s*(?:20)?(\d{2})\b", s, re.I)
    if m:
        return int(m.group(1))
    name_only = s.split("\n", 1)[0]
    m = re.search(r"(?:batch|scheme|adm)[\s_-]*(20[2-3]\d)", name_only, re.I)
    if m:
        return int(m.group(1))
    return None


def infer_batch_year_from_rolls(rolls):
    """
    Cohort batch year = the admission year shared by the most university rolls.
    First two digits of the roll (2435… → 24 → 2024). Diploma, branch-change,
    and readmission rolls are the minority and do not set the batch.
    """
    from collections import Counter

    years = []
    for roll in rolls or []:
        digits = re.sub(r"\D", "", str(roll))
        if len(digits) < 4:
            continue
        yy = int(digits[:2])
        if 15 <= yy <= 40:
            years.append(2000 + yy)
    if not years:
        return None
    return Counter(years).most_common(1)[0][0]


def apply_majority_batch_year(df):
    """Stamp every row with the cohort batch year from the majority roll prefix."""
    if df is None or df.empty or "Roll_No" not in df.columns:
        return df
    out = df.copy()
    majority = infer_batch_year_from_rolls(out["Roll_No"].tolist())
    if majority is None:
        return out
    out["Batch_Year"] = majority
    out["Batch_Year_Label"] = str(majority)
    return out


def _latest_class_value(group, column):
    """Value from the student's latest semester row. Blank values fall back to an earlier row.

    Diploma, branch-change, and readmission rolls keep the class of the sheet
    they sat, not the year in their own roll number.
    """
    if group is None or len(group) == 0 or column not in group.columns:
        return None
    work = group
    if "Semester" in work.columns:
        work = work.assign(_sem=pd.to_numeric(work["Semester"], errors="coerce").fillna(-1))
        work = work.sort_values("_sem", kind="mergesort")
    values = work[column].dropna().astype(str).str.strip()
    values = values[~values.str.lower().isin(["", "nan", "none", "unknown"])]
    return values.iloc[-1] if not values.empty else None


def _rolls_on_class_sheet(df, batch_label, branch=None):
    """Roll numbers that appear on a result sheet of this class, whatever their own prefix."""
    if df is None or df.empty or "Roll_No" not in df.columns:
        return set()
    work = df
    if branch and branch != "All" and "Branch" in work.columns:
        work = work[work["Branch"].astype(str) == str(branch)]
    if batch_label and batch_label != "All" and "Batch_Year_Label" in work.columns:
        work = work[work["Batch_Year_Label"].astype(str) == str(batch_label)]
    return set(work["Roll_No"].astype(str))


def _peek_result_header_text(file_path, max_chars=2500):
    """Read enough header text from CSV/PDF to detect semester/year."""
    path = Path(file_path)
    try:
        if path.suffix.lower() == ".pdf":
            # Prefer fast pdftotext on page 1 when available
            try:
                import subprocess
                proc = subprocess.run(
                    ["pdftotext", "-f", "1", "-l", "1", "-q", str(path), "-"],
                    capture_output=True,
                    text=True,
                    timeout=12,
                )
                if proc.returncode == 0 and proc.stdout and proc.stdout.strip():
                    return proc.stdout[:max_chars]
            except Exception:
                pass
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                if not pdf.pages:
                    return ""
                return (pdf.pages[0].extract_text() or "")[:max_chars]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return "".join(f.readlines()[:50])[:max_chars]
    except Exception:
        return ""


def _disk_cache_key(file_path, mtime):
    raw = f"{Path(file_path).resolve()}::{mtime}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _peek_result_header_text_cached(file_path, max_chars=2500):
    """Disk-cached header peek so catalog rebuilds don't reopen every PDF."""
    path = Path(file_path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""
    META_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = META_CACHE_DIR / f"{_disk_cache_key(path, mtime)}.txt"
    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception:
            pass
    text = _peek_result_header_text(path, max_chars=max_chars)
    try:
        cache_file.write_text(text or "", encoding="utf-8")
    except Exception:
        pass
    return text


def _sample_rolls_from_file(file_path, limit=400):
    """Collect university rolls so batch year can follow the majority prefix."""
    path = Path(file_path)
    rolls = []
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            roll_cols = [c for c in df.columns if "roll" in str(c).lower()]
            if roll_cols:
                rolls = df[roll_cols[0]].dropna().astype(str).tolist()
        elif path.suffix.lower() == ".pdf":
            text = ""
            try:
                import subprocess
                proc = subprocess.run(
                    ["pdftotext", "-f", "1", "-l", "2", "-q", str(path), "-"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if proc.returncode == 0:
                    text = proc.stdout or ""
            except Exception:
                text = ""
            if not text:
                text = _peek_result_header_text(path, max_chars=8000)
            rolls = re.findall(r"\b(2\d{6})\b", text)
    except Exception:
        return []
    return rolls


def infer_exam_type(text):
    """Regular / Reappear / Makeup from a filename or result header."""
    s = str(text or "")
    if re.search(r"make[\s_-]*up", s, re.I):
        return "Makeup"
    if re.search(r"re[\s_-]*appear", s, re.I):
        return "Reappear"
    if re.search(r"supplement", s, re.I):
        return "Supplementary"
    if re.search(r"\bregular\b", s, re.I):
        return "Regular"
    m = re.search(r"Result\s*Type\s*:\s*([A-Za-z][A-Za-z \-]+)", s, re.I)
    if m:
        return infer_exam_type(m.group(1))
    return "Regular"


# Later attempts replace earlier ones for the same student and semester.
EXAM_TYPE_RANK = {
    "regular": 1,
    "unknown": 1,
    "supplementary": 2,
    "reappear": 3,
    "makeup": 4,
}


def collapse_exam_attempts(df):
    """
    One row per student per branch per semester.
    If Regular, Reappear, and Makeup sheets all exist, keep the latest attempt
    (Makeup > Reappear > Supplementary > Regular). Students who only sat Regular stay on Regular.
    """
    if df is None or df.empty or "Roll_No" not in df.columns:
        return df
    work = df.copy()
    if "Exam_Type" not in work.columns:
        work["Exam_Type"] = "Regular"
    work["_exam_rank"] = (
        work["Exam_Type"].astype(str).str.lower().map(lambda x: EXAM_TYPE_RANK.get(x, 1)).fillna(1)
    )
    work["_sgpa_ord"] = (
        pd.to_numeric(work["SGPA"], errors="coerce").fillna(-1) if "SGPA" in work.columns else -1
    )
    keys = ["Roll_No"]
    if "Branch" in work.columns:
        keys.append("Branch")
    if "Semester" in work.columns:
        keys.append("Semester")
    elif "Semester_Label" in work.columns:
        keys.append("Semester_Label")
    work = work.sort_values(["_exam_rank", "_sgpa_ord"], ascending=True)
    out = work.drop_duplicates(subset=keys, keep="last")
    return out.drop(columns=["_exam_rank", "_sgpa_ord"], errors="ignore")


def parse_dataset_meta(file_path, label=None):
    """Branch / batch / exam / study-year / semester metadata for one result file.

    Fast path: filename only. Opens PDF/CSV header only when semester or batch
    cannot be inferred from the name (results are disk-cached).
    """
    path = Path(file_path)
    name = path.name
    branch = infer_branch_from_filename(name)

    semester = infer_semester_from_text(name)
    exam_year = infer_exam_year_from_text(name)

    need_header = semester is None
    header = ""
    if need_header:
        header = _peek_result_header_text_cached(path)
        blob = f"{name}\n{header}"
        if semester is None:
            semester = infer_semester_from_text(blob)
        if exam_year is None:
            exam_year = infer_exam_year_from_text(blob)

    # Batch year is the majority university-roll prefix, never the scheme code.
    batch_year = infer_batch_year_from_rolls(_sample_rolls_from_file(path))
    if batch_year is None:
        batch_year = infer_batch_year_from_text(name)

    study_year_label = study_year_from_semester(semester)
    semester_label = str(semester) if semester is not None else SEM_UNKNOWN
    batch_label = str(batch_year) if batch_year is not None else YEAR_UNKNOWN
    exam_label = str(exam_year) if exam_year is not None else YEAR_UNKNOWN

    exam_type = infer_exam_type(f"{name}\n{header}")
    display_parts = [branch, f"Batch {batch_label}", study_year_label, f"Sem {semester_label}", exam_type]
    display = " · ".join(display_parts)

    return {
        "label": label or _label_from_filename(name),
        "path": str(path.resolve()),
        "branch": branch,
        "semester": semester,
        "batch_year": batch_year,
        "exam_year": exam_year,
        "year": exam_year,
        "exam_type": exam_type,
        "semester_label": semester_label,
        "batch_year_label": batch_label,
        "exam_year_label": exam_label,
        "study_year_label": study_year_label,
        "year_label": exam_label,
        "display": display,
        "filename": name,
    }


def filter_catalog(
    catalog,
    branch=None,
    batch_year_label=None,
    exam_year_label=None,
    study_year_label=None,
    semester_label=None,
    year_label=None,
):
    rows = list(catalog)
    if branch and branch != "All":
        rows = [r for r in rows if r["branch"] == branch]
    if batch_year_label and batch_year_label != "All":
        rows = [r for r in rows if r.get("batch_year_label") == batch_year_label]
    # year_label kept as alias for exam year (older callers)
    exam_key = exam_year_label if exam_year_label is not None else year_label
    if exam_key and exam_key != "All":
        rows = [r for r in rows if r.get("exam_year_label", r.get("year_label")) == exam_key]
    if study_year_label and study_year_label != "All":
        rows = [r for r in rows if r.get("study_year_label") == study_year_label]
    if semester_label and semester_label != "All":
        rows = [r for r in rows if str(r["semester_label"]) == str(semester_label)]
    return rows


def _sorted_year_labels(labels):
    known = sorted(
        {x for x in labels if x != YEAR_UNKNOWN},
        key=lambda y: int(y) if str(y).isdigit() else y,
    )
    if YEAR_UNKNOWN in labels:
        known.append(YEAR_UNKNOWN)
    return known


def _sorted_sem_labels(labels):
    return sorted(
        set(labels),
        key=lambda x: (x == SEM_UNKNOWN, int(x) if str(x).isdigit() else 99),
    )


def _sorted_study_year_labels(labels):
    present = set(labels)
    return [y for y in STUDY_YEAR_ORDER if y in present]


def sidebar_branch_year_sem_picker(catalog, key_prefix="bys"):
    """Cascading Branch → Batch → Study Year → Semester picker."""
    st.sidebar.subheader("Branch / Batch / Year")
    if not catalog:
        st.sidebar.warning("No CSV/PDF datasets found in the data folder.")
        return None

    branches = sorted({c["branch"] for c in catalog})
    branch = st.sidebar.selectbox("Branch", branches, key=f"{key_prefix}_branch")
    scoped = filter_catalog(catalog, branch=branch)

    batches = _sorted_year_labels([c["batch_year_label"] for c in scoped])
    batch = st.sidebar.selectbox(
        "Batch year",
        batches if batches else [YEAR_UNKNOWN],
        key=f"{key_prefix}_batch",
        help="Admission year from the majority of university roll prefixes (first two digits). Diploma / branch-change rolls do not set the batch.",
    )
    scoped = filter_catalog(scoped, batch_year_label=batch)

    study_years = _sorted_study_year_labels([c["study_year_label"] for c in scoped])
    study_year = st.sidebar.selectbox(
        "Academic year",
        study_years if study_years else [STUDY_YEAR_UNKNOWN],
        key=f"{key_prefix}_study",
        help="1st Year = Sem 1–2, 2nd Year = Sem 3–4, 3rd Year = Sem 5–6, 4th Year = Sem 7–8.",
    )
    scoped = filter_catalog(scoped, study_year_label=study_year)

    sems = _sorted_sem_labels([c["semester_label"] for c in scoped])
    semester = st.sidebar.selectbox(
        "Semester",
        sems if sems else [SEM_UNKNOWN],
        key=f"{key_prefix}_sem",
    )
    scoped = filter_catalog(scoped, semester_label=semester)

    if not scoped:
        st.sidebar.warning("No dataset matches this Branch / Batch / Year / Semester.")
        return None

    exam_types = []
    for row in scoped:
        kind = row.get("exam_type") or infer_exam_type(row.get("filename", ""))
        if kind not in exam_types:
            exam_types.append(kind)
    st.sidebar.caption(
        f"Auto-loaded {len(scoped)} sheet(s): {', '.join(exam_types)}. "
        "Regular, Reappear, and Makeup are combined; each student keeps their latest attempt."
    )
    for row in scoped:
        kind = row.get("exam_type") or infer_exam_type(row.get("filename", ""))
        st.sidebar.caption(f"• {kind}: {row['filename']}")

    pick = dict(scoped[0])
    pick["matches"] = scoped
    pick["exam_types"] = exam_types
    pick["display"] = (
        f"{branch} · Batch {batch} · {study_year} · Sem {semester} · "
        + " + ".join(exam_types)
    )
    return pick


def sidebar_year_sem_filters(catalog, key_prefix="scope"):
    """Optional Batch / Academic year / Semester filters (no exam year)."""
    st.sidebar.subheader("Batch / Year filter")
    batches = ["All"] + _sorted_year_labels([c["batch_year_label"] for c in catalog])
    batch = st.sidebar.selectbox(
        "Batch year",
        batches,
        key=f"{key_prefix}_batch",
        help="Admission year from the majority of university roll prefixes (first two digits).",
    )

    scoped = catalog if batch == "All" else filter_catalog(catalog, batch_year_label=batch)
    study_years = ["All"] + _sorted_study_year_labels([c["study_year_label"] for c in scoped])
    study_year = st.sidebar.selectbox("Academic year", study_years, key=f"{key_prefix}_study")

    if study_year != "All":
        scoped = filter_catalog(scoped, study_year_label=study_year)
    sems = ["All"] + _sorted_sem_labels([c["semester_label"] for c in scoped])
    semester = st.sidebar.selectbox("Semester", sems, key=f"{key_prefix}_sem")
    return {
        "batch_year_label": batch,
        "study_year_label": study_year,
        "exam_year_label": "All",
        "semester_label": semester,
        "year_label": "All",
    }


def apply_year_sem_filter(df, scope=None, **kwargs):
    """Filter a combined frame by batch / study year / exam year / semester."""
    if df is None or df.empty:
        return df

    if isinstance(scope, dict):
        batch = scope.get("batch_year_label", "All")
        study = scope.get("study_year_label", "All")
        exam = scope.get("exam_year_label", scope.get("year_label", "All"))
        sem = scope.get("semester_label", "All")
    else:
        batch = kwargs.get("batch_year_label", "All")
        study = kwargs.get("study_year_label", "All")
        exam = kwargs.get("exam_year_label", kwargs.get("year_label", "All"))
        sem = kwargs.get("semester_label", "All")
        # Legacy: apply_year_sem_filter(df, exam_year, semester=...)
        if scope is not None:
            exam = scope

    out = df
    if batch and batch != "All" and "Batch_Year_Label" in out.columns:
        out = out[out["Batch_Year_Label"].astype(str) == str(batch)]
    if study and study != "All" and "Study_Year_Label" in out.columns:
        out = out[out["Study_Year_Label"].astype(str) == str(study)]
    if exam and exam != "All":
        if "Exam_Year_Label" in out.columns:
            out = out[out["Exam_Year_Label"].astype(str) == str(exam)]
        elif "Year_Label" in out.columns:
            out = out[out["Year_Label"].astype(str) == str(exam)]
    if sem and sem != "All" and "Semester_Label" in out.columns:
        out = out[out["Semester_Label"].astype(str) == str(sem)]
    return out.copy() if out is not None else out


# Backward-compatible alias used by older code paths
def infer_year_from_text(text):
    return infer_exam_year_from_text(text)
