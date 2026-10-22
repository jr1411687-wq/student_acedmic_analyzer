import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

from analyzer.catalog import SEM_UNKNOWN, STUDY_YEAR_FROM_SEM, STUDY_YEAR_UNKNOWN, YEAR_UNKNOWN, _disk_cache_key, infer_branch_from_filename, infer_semester_from_text
from analyzer.constants import BRANCH_ALIASES, META_CACHE_DIR
from analyzer.files import discover_section_list_files

def _normalize_roll(val):
    """Normalize university roll / URN to a comparable string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return None
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    s = re.sub(r"\s+", "", s)
    return s


def parse_section_list_meta(file_path):
    """Infer branch / semester / batch from a section-list filename."""
    path = Path(file_path)
    name = path.name
    branch = infer_branch_from_filename(name)
    # Filename often says "ece" even when infer returns long label
    stem = path.stem.lower().replace(".", " ").replace("_", " ").replace("-", " ").replace("(", " ").replace(")", " ")
    for code, aliases in BRANCH_ALIASES.items():
        if code.lower() in stem.split() or any(a in stem for a in aliases if len(a) > 2):
            branch = code
            break
        if any(tok == a for a in aliases if len(a) <= 3 for tok in stem.split()):
            branch = code
            break

    semester = infer_semester_from_text(name)
    batch = None
    m = re.search(r"batch\s*[:_]?\s*((?:20)?\d{2})\b", name, re.I)
    if m:
        batch = m.group(1)
        if len(batch) == 2:
            batch = "20" + batch
    study_year = STUDY_YEAR_FROM_SEM.get(semester, STUDY_YEAR_UNKNOWN) if semester else STUDY_YEAR_UNKNOWN
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "branch": branch,
        "semester": semester,
        "semester_label": str(semester) if semester else SEM_UNKNOWN,
        "batch_year": int(batch) if batch and str(batch).isdigit() else None,
        "batch_year_label": batch or YEAR_UNKNOWN,
        "study_year_label": study_year,
        "display": f"{branch} · Sem {semester or '?'} · Batch {batch or '?'} · {path.name}",
    }


def _section_cache_path(file_path, file_mtime):
    META_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _disk_cache_key(file_path, file_mtime)
    return META_CACHE_DIR / f"section_map_{key}.csv"


def parse_section_list_tabular(path):
    """Parse CSV/Excel section list into Branch/Section/URN rows."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            raw = pd.read_excel(path)
        except Exception:
            return pd.DataFrame()
    else:
        raw = pd.read_csv(path)

    if raw.empty:
        return pd.DataFrame()

    colmap = {str(c).strip().lower().replace(" ", "_"): c for c in raw.columns}
    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    urn_col = pick("urn", "university_roll_no", "university_roll", "roll_no", "roll", "u_roll")
    crn_col = pick("crn", "college_roll_no", "college_roll", "class_roll")
    sec_col = pick("section", "sec")
    sub_col = pick("subsection", "sub_section", "group")
    branch_col = pick("branch")
    sem_col = pick("semester", "sem")
    batch_col = pick("batch_year", "batch")

    if urn_col is None:
        # Heuristic: a column of 7-digit rolls
        for c in raw.columns:
            sample = raw[c].dropna().astype(str).head(20)
            if sample.empty:
                continue
            if sample.str.replace(r"\.0$", "", regex=True).str.fullmatch(r"\d{6,8}").mean() > 0.7:
                urn_col = c
                break
    if urn_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["URN"] = raw[urn_col].map(_normalize_roll)
    out["CRN"] = raw[crn_col].map(_normalize_roll) if crn_col else None
    if sec_col:
        out["Section"] = raw[sec_col].astype(str).str.strip().str.upper().str[0]
    elif sub_col:
        out["Section"] = raw[sub_col].astype(str).str.strip().str.upper().str[0]
    else:
        out["Section"] = "Unknown"
    out["Subsection"] = raw[sub_col].astype(str).str.strip().str.upper() if sub_col else out["Section"]
    meta = parse_section_list_meta(path)
    out["Branch"] = raw[branch_col].astype(str).str.strip() if branch_col else meta["branch"]
    if sem_col:
        out["Semester"] = pd.to_numeric(raw[sem_col], errors="coerce")
    else:
        out["Semester"] = meta["semester"]
    if batch_col:
        out["Batch_Year"] = pd.to_numeric(raw[batch_col], errors="coerce")
    else:
        out["Batch_Year"] = meta["batch_year"]
    out = out.dropna(subset=["URN"]).drop_duplicates(subset=["URN"], keep="first")
    out["Section"] = out["Section"].replace({"": "Unknown", "N": "Unknown"})
    return out.reset_index(drop=True)


def _ocr_section_list_pdf(path):
    """
    OCR a scanned section-list PDF (CRN|URN under A1/A2/B1/B2 columns).
    Requires pdftoppm + rapidocr_onnxruntime. Returns DataFrame or empty.
    """
    import subprocess
    import tempfile
    import numpy as np
    from collections import defaultdict

    try:
        from rapidocr_onnxruntime import RapidOCR
        from PIL import Image, ImageOps, ImageEnhance
    except Exception:
        return pd.DataFrame(), "OCR packages not installed (rapidocr_onnxruntime / pillow)"

    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="section_ocr_") as tmp:
        out_prefix = str(Path(tmp) / "page")
        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", "200", str(path), out_prefix],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            return pd.DataFrame(), "pdftoppm not found (poppler-utils)"
        except subprocess.CalledProcessError as e:
            return pd.DataFrame(), f"pdftoppm failed: {e}"

        pages = sorted(Path(tmp).glob("page*.png"))
        if not pages:
            return pd.DataFrame(), "No page images from PDF"

        ocr = RapidOCR()
        meta = parse_section_list_meta(path)
        all_rows = []

        for page_png in pages:
            im = Image.open(page_png)
            im2 = im.resize((im.width * 2, im.height * 2), Image.Resampling.LANCZOS)
            im2 = ImageOps.autocontrast(im2.convert("L"))
            im2 = ImageEnhance.Contrast(im2).enhance(1.6)
            enh = Path(tmp) / f"{page_png.stem}_enh.png"
            im2.save(enh)
            result, _ = ocr(str(enh))
            pts = []
            for box, text, score in (result or []):
                cleaned = re.sub(r"(?<=\d)\s+(?=\d)", "", str(text))
                nums = re.findall(r"\d{7}", cleaned)
                if not nums:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                cx, cy = sum(xs) / 4, sum(ys) / 4
                for n in nums:
                    pts.append((cx, cy, n))
            if len(pts) < 4:
                continue

            xs = np.array([p[0] for p in pts], dtype=float)
            centers = np.quantile(xs, [0.125, 0.375, 0.625, 0.875]).astype(float)
            assign = None
            for _ in range(12):
                assign = np.argmin(np.abs(xs[:, None] - centers[None, :]), axis=1)
                for k in range(4):
                    if np.any(assign == k):
                        centers[k] = xs[assign == k].mean()
            order = np.argsort(centers)
            labels = ["A1", "A2", "B1", "B2"]
            center_to_lab = {order[i]: labels[i] for i in range(4)}
            cols = defaultdict(list)
            for (cx, cy, n), a in zip(pts, assign):
                cols[center_to_lab[int(a)]].append({"cx": cx, "cy": cy, "n": n})

            def pair_col(vals, subsection):
                vals = sorted(vals, key=lambda x: (x["cy"], x["cx"]))
                rows, cur, ly = [], [], None
                for v in vals:
                    if ly is None or abs(v["cy"] - ly) < 28:
                        cur.append(v)
                        ly = v["cy"] if ly is None else 0.7 * ly + 0.3 * v["cy"]
                    else:
                        rows.append(cur)
                        cur = [v]
                        ly = v["cy"]
                if cur:
                    rows.append(cur)
                out = []
                for r in rows:
                    r = sorted(r, key=lambda x: x["cx"])
                    uniq = []
                    for x in r:
                        if x["n"] not in uniq:
                            uniq.append(x["n"])
                    if len(uniq) < 2:
                        continue
                    a, b = uniq[0], uniq[1]
                    if a.startswith("2417") and not b.startswith("2417"):
                        crn, urn = a, b
                    elif b.startswith("2417") and not a.startswith("2417"):
                        crn, urn = b, a
                    else:
                        crn, urn = a, b
                    out.append(
                        {
                            "Subsection": subsection,
                            "Section": subsection[0],
                            "CRN": crn,
                            "URN": urn,
                        }
                    )
                return out

            for lab in labels:
                all_rows.extend(pair_col(cols[lab], lab))

        if not all_rows:
            return pd.DataFrame(), "OCR found no CRN/URN pairs"

        df = pd.DataFrame(all_rows)
        df["URN"] = df["URN"].map(_normalize_roll)
        df["CRN"] = df["CRN"].map(_normalize_roll)
        df = df.dropna(subset=["URN"]).drop_duplicates(subset=["URN"], keep="first")
        df["Branch"] = meta["branch"]
        df["Semester"] = meta["semester"]
        df["Batch_Year"] = meta["batch_year"]
        return df.reset_index(drop=True), None


def _parse_section_list_pdf_words(path):
    """
    Parse digital section-list PDFs (CRN/URN columns under A1/A2/B1/B2)
    using word x-positions — works without OCR when the PDF has a text layer.
    """
    import pdfplumber

    path = Path(path)
    rows = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            words = page.extract_words() or []
            if not words:
                continue

            # Subsection headers (A1/A2/B1/B2) define column centers
            sub_hdrs = [
                w for w in words
                if re.fullmatch(r"[AB][12]", str(w["text"]).strip().upper())
            ]
            if len(sub_hdrs) < 2:
                continue
            sub_hdrs = sorted(sub_hdrs, key=lambda w: w["x0"])
            # Dedupe stacked headers at nearly same x
            cols = []
            for w in sub_hdrs:
                lab = str(w["text"]).strip().upper()
                cx = (w["x0"] + w["x1"]) / 2
                if cols and abs(cx - cols[-1]["cx"]) < 20:
                    continue
                cols.append({"label": lab, "cx": cx, "section": lab[0]})
            if len(cols) < 2:
                continue

            # Optional year band headers (2nd / 3rd / 4th) above the grid
            year_hdrs = []
            for w in words:
                t = str(w["text"]).strip().lower()
                if t in ("2nd", "3rd", "4th"):
                    year_hdrs.append({"label": t, "cx": (w["x0"] + w["x1"]) / 2})
            year_hdrs = sorted(year_hdrs, key=lambda x: x["cx"])

            def year_for_cx(cx):
                if not year_hdrs:
                    return None
                # Band edges at midpoints between year headers
                centers = [y["cx"] for y in year_hdrs]
                edges = [0]
                for i in range(len(centers) - 1):
                    edges.append((centers[i] + centers[i + 1]) / 2)
                edges.append(1e9)
                for i, y in enumerate(year_hdrs):
                    if edges[i] <= cx < edges[i + 1]:
                        return {
                            "2nd": "2nd Year",
                            "3rd": "3rd Year",
                            "4th": "4th Year",
                        }.get(y["label"])
                return None

            for c in cols:
                c["study_year"] = year_for_cx(c["cx"])

            # If layout is the standard 10-column sheet (4+4+2), force year by column index
            if len(cols) == 10 and len(year_hdrs) >= 2:
                for i, c in enumerate(cols):
                    if i < 4:
                        c["study_year"] = "2nd Year"
                    elif i < 8:
                        c["study_year"] = "3rd Year"
                    else:
                        c["study_year"] = "4th Year"

            # Number tokens below header row
            hdr_bottom = max(w["bottom"] for w in sub_hdrs) + 5
            nums = []
            for w in words:
                txt = re.sub(r"\s+", "", str(w["text"]))
                if not re.fullmatch(r"\d{6,8}", txt):
                    continue
                if w["top"] < hdr_bottom:
                    continue
                nums.append({
                    "n": txt,
                    "cx": (w["x0"] + w["x1"]) / 2,
                    "cy": (w["top"] + w["bottom"]) / 2,
                })

            def nearest_col(cx):
                return min(cols, key=lambda c: abs(c["cx"] - cx))

            # Pair CRN (left) + URN (right) within each subsection column / row
            # Each subsection spans a CRN column and URN column side by side;
            # numbers near a subsection header alternate CRN/URN by x within that band.
            # Build bands: midpoints between consecutive subsection centers
            ordered = sorted(cols, key=lambda c: c["cx"])
            edges = []
            for i, c in enumerate(ordered):
                left = (ordered[i - 1]["cx"] + c["cx"]) / 2 if i else c["cx"] - 40
                right = (c["cx"] + ordered[i + 1]["cx"]) / 2 if i + 1 < len(ordered) else c["cx"] + 40
                edges.append((left, right, c))

            by_col = {id(c): [] for _, _, c in edges}
            for n in nums:
                for left, right, c in edges:
                    if left <= n["cx"] < right:
                        by_col[id(c)].append(n)
                        break

            for left, right, c in edges:
                vals = sorted(by_col[id(c)], key=lambda x: (round(x["cy"] / 6), x["cx"]))
                # Cluster into rows
                clusters, cur, ly = [], [], None
                for v in vals:
                    if ly is None or abs(v["cy"] - ly) < 8:
                        cur.append(v)
                        ly = v["cy"] if ly is None else 0.6 * ly + 0.4 * v["cy"]
                    else:
                        clusters.append(cur)
                        cur = [v]
                        ly = v["cy"]
                if cur:
                    clusters.append(cur)
                for cluster in clusters:
                    cluster = sorted(cluster, key=lambda x: x["cx"])
                    if len(cluster) < 2:
                        continue
                    crn, urn = cluster[0]["n"], cluster[1]["n"]
                    # Prefer class-roll-like as CRN (same first 3 digits family as neighbors)
                    rows.append({
                        "CRN": crn,
                        "URN": urn,
                        "Section": c["section"],
                        "Subsection": c["label"],
                        "Study_Year_Label": c.get("study_year") or STUDY_YEAR_UNKNOWN,
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["URN"] = df["URN"].map(_normalize_roll)
    df["CRN"] = df["CRN"].map(_normalize_roll)
    df = df.dropna(subset=["URN"]).drop_duplicates(subset=["URN"], keep="first")
    # Swap if URN looks like CRN pattern wrongly ordered (URN usually not 2417/2517/2317 class rolls)
    def _fix_pair(r):
        crn, urn = r["CRN"], r["URN"]
        crn_like = str(crn).startswith(("2417", "2517", "2317", "2217"))
        urn_like_crn = str(urn).startswith(("2417", "2517", "2317", "2217"))
        if (not crn_like) and urn_like_crn:
            return pd.Series({**r, "CRN": urn, "URN": crn})
        return r

    df = df.apply(_fix_pair, axis=1)
    df["URN"] = df["URN"].map(_normalize_roll)
    df["CRN"] = df["CRN"].map(_normalize_roll)
    return df.dropna(subset=["URN"]).drop_duplicates(subset=["URN"], keep="first").reset_index(drop=True)


def enrich_section_map_branches(section_df, results_df=None):
    """
    Auto-detect Branch for each URN by matching against result Roll_No values.
    Falls back to majority branch / filename meta.
    """
    if section_df is None or section_df.empty:
        return section_df
    out = section_df.copy()
    out["URN"] = out["URN"].map(_normalize_roll)

    roll_to_branch = {}
    if results_df is not None and not results_df.empty and "Roll_No" in results_df.columns:
        tmp = results_df.copy()
        tmp["_roll"] = tmp["Roll_No"].map(_normalize_roll)
        if "Branch" in tmp.columns:
            for roll, branch in zip(tmp["_roll"], tmp["Branch"].map(_branch_match_key)):
                if roll and branch and roll not in roll_to_branch:
                    roll_to_branch[roll] = branch

    if roll_to_branch:
        out["Branch"] = out["URN"].map(lambda u: roll_to_branch.get(u))
        # Fill unknowns with majority matched branch
        matched = out["Branch"].dropna()
        if not matched.empty:
            majority = matched.value_counts().idxmax()
            out["Branch"] = out["Branch"].fillna(majority)
        else:
            out["Branch"] = out.get("Branch", "Unknown")
    elif "Branch" not in out.columns:
        out["Branch"] = "Unknown"

    # Drop filename garbage like long labels when we have codes
    out["Branch"] = out["Branch"].map(lambda b: _branch_match_key(b) or b)
    return out


def parse_section_list_pdf(path):
    """Parse section list PDF: prefer text-layer layout parse, then OCR fallback."""
    path = Path(path)
    err = None

    # 1) Digital PDF with text (e.g. "section list.pdf") — no OCR needed
    try:
        df = _parse_section_list_pdf_words(path)
        if df is not None and not df.empty and df["Section"].astype(str).ne("Unknown").any():
            meta = parse_section_list_meta(path)
            if "Branch" not in df.columns or df["Branch"].isna().all():
                df["Branch"] = meta.get("branch") or "Unknown"
            if "Semester" not in df.columns:
                df["Semester"] = meta.get("semester")
            if "Batch_Year" not in df.columns:
                df["Batch_Year"] = meta.get("batch_year")
            return df.reset_index(drop=True), None
    except Exception as e:
        err = f"text parse failed: {e}"

    # 2) OCR fallback (scanned image PDFs)
    df_ocr, ocr_err = _ocr_section_list_pdf(path)
    if df_ocr is not None and not df_ocr.empty:
        return df_ocr, None
    return pd.DataFrame(), ocr_err or err or "Could not parse section list PDF"


@st.cache_data(show_spinner=False)
def load_section_map(file_path, file_mtime, _parser_version=2):
    """Load one section-list file → Roll/URN → Section map (cached)."""
    path = Path(file_path)
    cache = _section_cache_path(file_path, file_mtime)
    if cache.exists():
        try:
            df = pd.read_csv(cache)
            if (
                not df.empty
                and "URN" in df.columns
                and "Section" in df.columns
                and df["Section"].astype(str).str.upper().isin(["A", "B", "C", "D"]).any()
            ):
                df["URN"] = df["URN"].map(_normalize_roll)
                return df, None
        except Exception:
            pass

    err = None
    if path.suffix.lower() in (".csv", ".xlsx", ".xls"):
        df = parse_section_list_tabular(path)
        if df is None or df.empty:
            err = "CSV/Excel section list missing URN + Section columns"
    elif path.suffix.lower() == ".pdf":
        df, err = parse_section_list_pdf(path)
    else:
        df, err = pd.DataFrame(), f"Unsupported section list type: {path.suffix}"

    if df is not None and not df.empty:
        try:
            df.to_csv(cache, index=False)
        except Exception:
            pass
    return (df if df is not None else pd.DataFrame()), err


@st.cache_data(show_spinner=False)
def load_all_section_maps(_mtime_signature, _parser_version=2):
    """Load every discovered section list; prefer CSV over PDF for same branch/sem."""
    frames = []
    errors = []
    seen_keys = set()
    for path in discover_section_list_files():
        meta = parse_section_list_meta(path)
        key = (str(meta["branch"]).upper(), str(meta["semester"]), str(meta["batch_year_label"]))
        # Skip PDF if we already loaded a CSV for same cohort
        if key in seen_keys and path.suffix.lower() == ".pdf":
            continue
        try:
            mtime = path.stat().st_mtime
            df, err = load_section_map(str(path.resolve()), mtime)
            if err:
                errors.append(f"{path.name}: {err}")
            if df is None or df.empty:
                if not err:
                    errors.append(f"{path.name}: empty / unparseable")
                continue
            tagged = df.copy()
            tagged["Section_Source"] = path.name
            for col, val in {
                "Branch": meta["branch"],
                "Semester": meta["semester"],
                "Batch_Year": meta["batch_year"],
            }.items():
                if col not in tagged.columns or tagged[col].isna().all():
                    tagged[col] = val
            frames.append(tagged)
            if path.suffix.lower() == ".csv":
                seen_keys.add(key)
        except Exception as e:
            errors.append(f"{path.name}: {e}")

    if not frames:
        return pd.DataFrame(), errors
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["URN"] = out["URN"].map(_normalize_roll)
    out = out.dropna(subset=["URN"]).drop_duplicates(subset=["URN"], keep="first")
    return out.reset_index(drop=True), errors


def _branch_match_key(name):
    """Normalize branch labels so 'ECE' matches long result-file names."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).strip()
    # Prefer canonical short code when possible
    inferred = infer_branch_from_filename(s)
    if inferred in BRANCH_ALIASES:
        return inferred.upper()
    low = s.lower().replace("&", " and ")
    for code, aliases in BRANCH_ALIASES.items():
        if code.lower() == low or any(a in low for a in aliases):
            return code.upper()
    return s.upper()


def attach_sections(students_df, section_df, branch=None, semester=None):
    """
    Join Section / Subsection onto student rows using University Roll_No ↔ URN.

    Section is permanent for a URN (same student keeps A/B across all semesters),
    so `semester` is ignored for the section map — it only filters the map by branch.
    """
    if students_df is None or students_df.empty:
        return students_df
    out = students_df.copy()
    if section_df is None or section_df.empty:
        out["Section"] = "Unmapped"
        out["Subsection"] = "Unmapped"
        return out

    smap = section_df.copy()
    if branch and branch != "All" and "Branch" in smap.columns:
        bkey = _branch_match_key(branch)
        smap = smap[smap["Branch"].map(_branch_match_key) == bkey]
    # NOTE: do not filter section map by semester — URN→Section is cohort-wide

    if "Roll_No" not in out.columns:
        out["Section"] = "Unmapped"
        out["Subsection"] = "Unmapped"
        return out

    out["_roll"] = out["Roll_No"].map(_normalize_roll)
    cols = ["URN", "Section", "Subsection"]
    for c in ("CRN", "Section_Source"):
        if c in smap.columns:
            cols.append(c)
    smap = smap[[c for c in cols if c in smap.columns]].drop_duplicates(subset=["URN"], keep="first")
    # Avoid duplicate Section/Subsection after merge
    overlap = [c for c in ("Section", "Subsection") if c in out.columns and c in smap.columns]
    if overlap:
        out = out.drop(columns=overlap)
    merged = out.merge(smap, how="left", left_on="_roll", right_on="URN", suffixes=("", "_sec"))
    if "Section" not in merged.columns and "Section_sec" in merged.columns:
        merged["Section"] = merged["Section_sec"]
    if "Section" in merged.columns and isinstance(merged["Section"], pd.DataFrame):
        merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]
    merged["Section"] = merged["Section"].fillna("Unmapped")
    if "Subsection" in merged.columns:
        if isinstance(merged["Subsection"], pd.DataFrame):
            merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]
        merged["Subsection"] = merged["Subsection"].fillna("Unmapped")
    else:
        merged["Subsection"] = merged["Section"]
    return merged.drop(columns=["_roll", "URN", "Section_sec", "Subsection_sec"], errors="ignore")


def _section_roll_coverage(section_map, results_df):
    """How many section-list URNs appear in each result semester."""
    if section_map is None or section_map.empty or results_df is None or results_df.empty:
        return pd.DataFrame()
    urns = set(section_map["URN"].map(_normalize_roll).dropna())
    if not urns or "Roll_No" not in results_df.columns:
        return pd.DataFrame()
    work = results_df.copy()
    work["_roll"] = work["Roll_No"].map(_normalize_roll)
    work["_hit"] = work["_roll"].isin(urns)
    if "Semester" in work.columns:
        work["_sem"] = pd.to_numeric(work["Semester"], errors="coerce")
    else:
        work["_sem"] = None
    rows = []
    for sem, g in work.groupby(work["_sem"], dropna=False):
        rolls = set(g["_roll"].dropna())
        hit = rolls & urns
        rows.append({
            "Semester": "Unknown" if pd.isna(sem) else str(int(sem)),
            "Result students": len(rolls),
            "URN matched to section list": len(hit),
            "Match %": round(100 * len(hit) / max(len(urns), 1), 1),
            "Section-list URNs missing in this sem": len(urns - rolls),
        })
    # Overall unique
    all_rolls = set(work["_roll"].dropna())
    rows.append({
        "Semester": "All (unique)",
        "Result students": len(all_rolls),
        "URN matched to section list": len(all_rolls & urns),
        "Match %": round(100 * len(all_rolls & urns) / max(len(urns), 1), 1),
        "Section-list URNs missing in this sem": len(urns - all_rolls),
    })
    return pd.DataFrame(rows)


def _section_summary_by_semester(df, group_col="_Group"):
    """Section stats broken down by result semester."""
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "Semester" in work.columns:
        work["_sem"] = pd.to_numeric(work["Semester"], errors="coerce")
    else:
        work["_sem"] = None
    rows = []
    for (sem, section), group in work.groupby(["_sem", group_col], dropna=False):
        if str(section) in ("Unmapped", "Unknown"):
            continue
        total = len(group)
        sgpa = pd.to_numeric(group["SGPA"], errors="coerce") if "SGPA" in group.columns else pd.Series(dtype=float)
        graded_n = int(sgpa.notna().sum())
        rows.append({
            "Semester": "Unknown" if pd.isna(sem) else int(sem),
            "Section": section,
            "Students": total,
            "Avg SGPA": round(float(sgpa.mean()), 2) if graded_n else None,
            "Pass Rate %": round(float((sgpa >= 5.0).sum() / total * 100), 1) if total else None,
            "Graded": graded_n,
        })
    return pd.DataFrame(rows)


def _section_summary(df):
    """Per-section SGPA / pass / gender summary."""
    if df is None or df.empty or "Section" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    # Duplicate "Section" columns (from merges/renames) make groupby fail
    if isinstance(work["Section"], pd.DataFrame):
        work = work.loc[:, ~work.columns.duplicated(keep="last")]
    sec = work["Section"]
    if isinstance(sec, pd.DataFrame):
        work["Section"] = sec.iloc[:, -1]
    rows = []
    for section, group in work.groupby(work["Section"].astype(str), sort=True):
        if section in ("Unmapped", "Unknown"):
            continue
        total = len(group)
        sgpa = pd.to_numeric(group["SGPA"], errors="coerce") if "SGPA" in group.columns else pd.Series(dtype=float)
        graded_n = int(sgpa.notna().sum())
        male = int((group["Gender"] == "Male").sum()) if "Gender" in group.columns else 0
        female = int((group["Gender"] == "Female").sum()) if "Gender" in group.columns else 0
        rows.append({
            "Section": section,
            "Students": total,
            "Avg SGPA": round(float(sgpa.mean()), 2) if graded_n else None,
            "Highest SGPA": round(float(sgpa.max()), 2) if graded_n else None,
            "Lowest SGPA": round(float(sgpa.min()), 2) if graded_n else None,
            "Pass Rate %": round(float((sgpa >= 5.0).sum() / total * 100), 1) if total else None,
            "Graded": graded_n,
            "Male": male,
            "Female": female,
            "Male:Female": f"{male}:{female}",
        })
    return pd.DataFrame(rows)

