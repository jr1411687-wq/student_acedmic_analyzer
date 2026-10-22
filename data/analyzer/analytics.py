import pandas as pd
import re
from decimal import Decimal, ROUND_HALF_UP

from analyzer.catalog import _latest_class_value
from analyzer.parsers import _is_grade_value, _is_subject_column_name
from analyzer.sections import _normalize_roll, attach_sections

def _branch_summary(combined_df):
    """Build per-branch SGPA + gender summary for college comparison."""
    rows = []
    for branch, group in combined_df.groupby("Branch", sort=True):
        total = len(group)
        sgpa = pd.to_numeric(group.get("SGPA"), errors="coerce") if "SGPA" in group.columns else pd.Series(dtype=float)
        graded = sgpa.notna()
        graded_n = int(graded.sum())
        male = int((group["Gender"] == "Male").sum()) if "Gender" in group.columns else 0
        female = int((group["Gender"] == "Female").sum()) if "Gender" in group.columns else 0
        unknown = int((group["Gender"] == "Unknown").sum()) if "Gender" in group.columns else 0
        known = male + female
        rows.append({
            "Branch": branch,
            "Students": total,
            "Avg SGPA": round(float(sgpa.mean()), 2) if graded_n else None,
            "Highest SGPA": round(float(sgpa.max()), 2) if graded_n else None,
            "Lowest SGPA": round(float(sgpa.min()), 2) if graded_n else None,
            # Clearance rate: SGPA ≥ 5 among ALL students (blank SGPA = not cleared)
            "Pass Rate %": round(float((sgpa >= 5.0).sum() / total * 100), 1) if total else None,
            "Graded": graded_n,
            "Missing SGPA": total - graded_n,
            "Male": male,
            "Female": female,
            "Unknown": unknown,
            "Male %": round(male / known * 100, 1) if known else 0.0,
            "Female %": round(female / known * 100, 1) if known else 0.0,
            "Male:Female": f"{male}:{female}",
        })
    return pd.DataFrame(rows)


def _unique_student_count(df):
    """Count distinct students by university Roll_No (one person = one roll)."""
    if df is None or df.empty:
        return 0
    if "Roll_No" in df.columns:
        return int(df["Roll_No"].astype(str).nunique())
    return len(df)


def _latest_result_per_student(df):
    """
    Keep one row per student (Roll_No): latest semester / exam year.
    If a roll appears under multiple branches (upgrade / lateral), keep the
    latest semester row only so headcount is not inflated.
    """
    if df is None or df.empty or "Roll_No" not in df.columns:
        return df
    work = df.copy()
    work["_sem_ord"] = pd.to_numeric(work["Semester"], errors="coerce") if "Semester" in work.columns else -1
    work["_sem_ord"] = pd.to_numeric(work["_sem_ord"], errors="coerce").fillna(-1)
    if "Exam_Year" in work.columns:
        work["_exam_ord"] = pd.to_numeric(work["Exam_Year"], errors="coerce").fillna(-1)
    else:
        work["_exam_ord"] = -1
    work = work.sort_values(["_exam_ord", "_sem_ord"], ascending=True)
    # One person = one roll (not Branch+Roll)
    out = work.drop_duplicates(subset=["Roll_No"], keep="last")
    return out.drop(columns=["_sem_ord", "_exam_ord"], errors="ignore")


def _prepare_college_view(df):
    """
    For college metrics: if the same student appears in multiple semesters,
    use latest result per student so Total Students is not inflated.
    Returns (view_df, result_rows, unique_students, deduped).
    """
    if df is None or df.empty:
        return df, 0, 0, False
    result_rows = len(df)
    unique_students = _unique_student_count(df)
    multi_sem = False
    if "Semester_Label" in df.columns:
        multi_sem = df["Semester_Label"].nunique(dropna=False) > 1
    if "Roll_No" in df.columns:
        multi_sem = multi_sem or (df.groupby("Roll_No").size().max() > 1)

    if multi_sem and unique_students < result_rows:
        view = _latest_result_per_student(df)
        return view, result_rows, _unique_student_count(view), True
    return df, result_rows, unique_students, False


PASSING_SGPA_THRESHOLD = 5.0
FAIL_GRADE_PATTERN = re.compile(
    r'^(F|F\(Ext\.\)|F\(Int\.\)|Ab|AB|E|UMC|US|NE)(\s*\([^)]+\))?$',
    re.IGNORECASE,
)
SUBJECT_META_EXCLUDE = {
    "Roll_No", "Name", "Father_Name", "SGPA", "Credits_Earned",
    "Gender", "Branch", "Dataset", "Page", "Pass_Status",
    "Performance_Category", "SGPA_Range", "Risk_Level", "Risk_Reasons",
    "Fail_Count", "At_Risk", "Sr_No", "Sr No", "University_Roll_No",
    "Elective_I_Code",  # subject code text, not a grade
}


def _is_fail_grade(val):
    if pd.isna(val) or val == "":
        return False
    return bool(FAIL_GRADE_PATTERN.match(str(val).strip()))


def _clean_subject_label(col_name):
    """Normalize subject column labels for display (drop paper IDs, etc.)."""
    text = str(col_name).strip()
    # CIT104_96397 -> CIT104 ; CEC104 (96380) already cleaned elsewhere
    text = re.sub(r"[_\s]*\(\d+\)\s*$", "", text)
    text = re.sub(r"_(\d{4,})$", "", text)
    text = re.sub(r"^Subject_\d+$", text, text)
    return text or str(col_name)


def _subject_cols_for_frame(df, min_filled_ratio=0.25, min_filled_count=5):
    """
    Detect subject/grade columns that actually contain data in this frame.
    Critical after pd.concat: other branches' subject columns exist but are empty.
    """
    cols = []
    n = len(df)
    if n == 0:
        return cols

    for col in df.columns:
        if col in SUBJECT_META_EXCLUDE:
            continue
        col_l = str(col).lower()
        if col_l.endswith("_code") or col_l in ("sr_no", "sr no", "page"):
            continue

        series = df[col]
        non_null = series.dropna()
        # empty strings don't count as filled grades
        if non_null.dtype == object or str(series.dtype) == "object":
            non_null = non_null[non_null.astype(str).str.strip().ne("")]
        filled = len(non_null)
        if filled < min_filled_count and (n == 0 or filled / n < min_filled_ratio):
            continue
        if filled == 0:
            continue

        name_ok = _is_subject_column_name(col) or bool(
            re.search(r"elective|grade", str(col), re.I)
        )
        grade_ratio = non_null.astype(str).apply(_is_grade_value).mean() if filled else 0

        # Prefer real grade columns; avoid pure numeric metadata
        if name_ok and grade_ratio >= 0.4:
            cols.append(col)
        elif grade_ratio >= 0.6:
            cols.append(col)

    # Prefer cleaner unique labels while keeping original col for data access
    return cols


def _count_fails(row, subject_cols):
    return sum(1 for c in subject_cols if c in row.index and _is_fail_grade(row.get(c)))


def build_at_risk_table(
    df,
    sgpa_threshold=PASSING_SGPA_THRESHOLD,
    min_fails=2,
    min_credits=None,
):
    """Flag at-risk students with reasons."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    # Count fails using only subjects present for each branch subset
    if "Branch" in work.columns:
        fail_counts = pd.Series(0, index=work.index, dtype=int)
        for _, group in work.groupby("Branch"):
            subs = _subject_cols_for_frame(group)
            if not subs:
                continue
            fail_counts.loc[group.index] = group.apply(lambda r: _count_fails(r, subs), axis=1)
        work["Fail_Count"] = fail_counts
    else:
        subject_cols = _subject_cols_for_frame(work)
        work["Fail_Count"] = (
            work.apply(lambda r: _count_fails(r, subject_cols), axis=1) if subject_cols else 0
        )

    if "SGPA" in work.columns:
        work["SGPA"] = pd.to_numeric(work["SGPA"], errors="coerce")
    if "Credits_Earned" in work.columns:
        work["Credits_Earned"] = pd.to_numeric(work["Credits_Earned"], errors="coerce")

    reasons = []
    at_risk_flags = []
    levels = []
    for _, row in work.iterrows():
        why = []
        sgpa = row.get("SGPA")
        fails = int(row.get("Fail_Count", 0) or 0)
        credits = row.get("Credits_Earned")

        if pd.isna(sgpa):
            why.append("Missing SGPA")
        elif float(sgpa) < sgpa_threshold:
            why.append(f"SGPA < {sgpa_threshold}")

        if fails >= min_fails:
            why.append(f"{fails}+ fail/absent grades")

        if min_credits is not None and pd.notna(credits) and float(credits) < min_credits:
            why.append(f"Credits < {min_credits}")

        at_risk_flags.append(len(why) > 0)
        reasons.append("; ".join(why) if why else "")
        if not why:
            levels.append("OK")
        elif (pd.notna(sgpa) and float(sgpa) < 4.0) or fails >= 4:
            levels.append("High")
        elif (pd.isna(sgpa)) or (pd.notna(sgpa) and float(sgpa) < sgpa_threshold) or fails >= min_fails:
            levels.append("Medium")
        else:
            levels.append("Low")

    work["At_Risk"] = at_risk_flags
    work["Risk_Reasons"] = reasons
    work["Risk_Level"] = levels
    return work[work["At_Risk"]].copy()


def build_subject_fail_matrix(df, by_branch=True):
    """
    Return long dataframe of fail rates using only subjects that have
    real grade values in that branch (avoids cross-branch empty columns).
    Fail rate = fails / students with a grade in that subject.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    if by_branch and "Branch" in df.columns:
        groups = df.groupby("Branch", sort=True)
    else:
        groups = [("All", df)]

    for branch, group in groups:
        subject_cols = _subject_cols_for_frame(group)
        if not subject_cols:
            continue

        for sub in subject_cols:
            series = group[sub]
            # only students who have a grade entry for this subject
            mask = series.notna() & series.astype(str).str.strip().ne("")
            attempted = int(mask.sum())
            if attempted == 0:
                continue
            fails = int(series[mask].apply(_is_fail_grade).sum())
            rows.append({
                "Branch": branch,
                "Subject": _clean_subject_label(sub),
                "Subject_Col": sub,
                "Fail_Count": fails,
                "Graded_Students": attempted,
                "Fail_Rate_%": round(fails / attempted * 100, 1),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # If duplicate cleaned labels exist within a branch, keep the worse fail rate
    out = (
        out.sort_values("Fail_Rate_%", ascending=False)
        .groupby(["Branch", "Subject"], as_index=False)
        .first()
    )
    return out



# =========================================================
# OVERALL CGPA (IKG-PTU / college notice formula)
# =========================================================

MERIT_GRADE_ORDER = ["O", "A+", "A", "B+", "B", "C", "P", "D"]


def _round_half_up(value, places=2):
    """Round half away from zero to a fixed number of decimal places."""
    quant = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def _merit_grade_token(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    raw = str(val).strip().upper()
    if _is_fail_grade(val):
        return None
    for grade in MERIT_GRADE_ORDER:
        if raw == grade or raw.startswith(grade + "(") or raw.startswith(grade + " "):
            return grade
    return None


def _cgpa_notice_example():
    """Worked example from the 20.02.2017 academic notice → 6.42."""
    sgpa = [5.67, 6.28, 7.14, 6.58]
    credits = [26, 28, 27, 25]
    num = sum(s * c for s, c in zip(sgpa, credits))
    den = sum(credits)
    return _round_half_up(num / den)


def compute_overall_cgpa(df):
    """
    CGPA = Σ(SGPA × credits) / Σ(credits), half-up to 2 decimals.

    Every semester for the same roll is included, even if it was in another branch.
    S is the SGPA printed on that semester result. If that result has a reappear
    or no SGPA, S is 0. C is the credits printed on that same result.
    """
    empty = pd.DataFrame()
    if df is None or df.empty or "Roll_No" not in df.columns:
        return empty, empty

    work = df.copy()
    work["Roll_No"] = work["Roll_No"].map(_normalize_roll)
    work = work.dropna(subset=["Roll_No"])
    if work.empty:
        return empty, empty

    work["_sem"] = pd.to_numeric(work["Semester"], errors="coerce") if "Semester" in work.columns else pd.NA
    work["SGPA"] = pd.to_numeric(work["SGPA"], errors="coerce") if "SGPA" in work.columns else pd.NA
    work["Credits_Earned"] = (
        pd.to_numeric(work["Credits_Earned"], errors="coerce")
        if "Credits_Earned" in work.columns else pd.NA
    )
    if "Exam_Year" in work.columns:
        work["_exam"] = pd.to_numeric(work["Exam_Year"], errors="coerce").fillna(-1)
    else:
        work["_exam"] = -1
    work = work.sort_values(["_exam", "_sem"], ascending=True)
    # One attempt per semester, even if an earlier semester was in another branch.
    work = work.drop_duplicates(subset=["Roll_No", "_sem"], keep="last")

    scheme_keys = [c for c in ["Branch", "Batch_Year_Label", "_sem"] if c in work.columns or c == "_sem"]
    scheme = (
        work.groupby(scheme_keys, dropna=False)["Credits_Earned"]
        .max()
        .rename("Scheme_Credits")
        .reset_index()
    )
    work = work.merge(scheme, on=scheme_keys, how="left")

    records = []
    branch_groups = work.groupby("Branch", dropna=False) if "Branch" in work.columns else [(None, work)]
    for _, group in branch_groups:
        subs = _subject_cols_for_frame(group)
        for idx, row in group.iterrows():
            scheme_c = row.get("Scheme_Credits")
            earned = row.get("Credits_Earned")
            sgpa = row.get("SGPA")
            fails = _count_fails(row, subs) if subs else 0
            # Use this semester's own SGPA and credits. A reappear or blank SGPA is 0,
            # and that semester's printed credits still stay in the total.
            uncleared = fails > 0 or pd.isna(sgpa)
            s_used = 0.0 if uncleared else float(sgpa)
            if pd.notna(earned) and float(earned) > 0:
                c_used = float(earned)
            elif pd.notna(scheme_c) and float(scheme_c) > 0:
                c_used = float(scheme_c)
            else:
                c_used = None
            merit = {g: 0 for g in MERIT_GRADE_ORDER}
            for col in subs:
                token = _merit_grade_token(row.get(col))
                if token:
                    merit[token] += 1
            sem_val = row.get("_sem")
            name_val = row.get("Name") if "Name" in work.columns else None
            if pd.isna(name_val) or str(name_val).strip().lower() in {"", "nan", "none"}:
                name_val = None
            records.append({
                "Roll_No": row["Roll_No"],
                "Name": None if name_val is None else str(name_val).strip(),
                "Branch": row.get("Branch") if "Branch" in work.columns else None,
                "Batch_Year_Label": row.get("Batch_Year_Label") if "Batch_Year_Label" in work.columns else None,
                "Gender": row.get("Gender") if "Gender" in work.columns else None,
                "Semester": None if pd.isna(sem_val) else int(sem_val),
                "SGPA": None if pd.isna(sgpa) else round(float(sgpa), 2),
                "SGPA_Used": s_used,
                "Scheme_Credits": c_used,
                "Credits_Earned": None if pd.isna(earned) else float(earned),
                "Uncleared": bool(uncleared),
                "Fail_Count": int(fails),
                **{f"Grade_{g}": merit[g] for g in MERIT_GRADE_ORDER},
            })

    detail = pd.DataFrame(records)
    if detail.empty:
        return empty, detail

    summary_rows = []
    for roll, group in detail.groupby("Roll_No", dropna=False):
        valid = group.dropna(subset=["Scheme_Credits"])
        valid = valid[valid["Scheme_Credits"] > 0]
        if valid.empty:
            continue
        weighted = float((valid["SGPA_Used"] * valid["Scheme_Credits"]).sum())
        total_c = float(valid["Scheme_Credits"].sum())
        cgpa = _round_half_up(weighted / total_c) if total_c else None
        sems = sorted({int(s) for s in valid["Semester"].dropna()})
        uncleared_sems = sorted({int(s) for s in valid.loc[valid["Uncleared"], "Semester"].dropna()})
        current_branch = _latest_class_value(group, "Branch")
        earlier = []
        if "Branch" in group.columns:
            for branch_name in group.sort_values("Semester")["Branch"].dropna().astype(str):
                if branch_name != str(current_branch) and branch_name not in earlier:
                    earlier.append(branch_name)
        rec = {
            "Roll_No": roll,
            "Name": _latest_class_value(group, "Name"),
            "Branch": current_branch,
            "Earlier_Branch": ", ".join(earlier) if earlier else "—",
            "Batch_Year_Label": _latest_class_value(group, "Batch_Year_Label"),
            "Gender": group["Gender"].dropna().mode().iloc[0] if group["Gender"].notna().any() else None,
            "Semesters": ",".join(str(s) for s in sems),
            "Semesters_Count": len(sems),
            "Scheme_Credits": round(total_c, 1),
            "Weighted_Points": round(weighted, 2),
            "CGPA": cgpa,
            "Uncleared_Semesters": ",".join(str(s) for s in uncleared_sems) if uncleared_sems else "—",
            "Fully_Cleared": len(uncleared_sems) == 0,
        }
        for grade in MERIT_GRADE_ORDER:
            rec[f"Grade_{grade}"] = int(group[f"Grade_{grade}"].sum())
        summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary, detail

    ranked_parts = []
    sort_cols = ["CGPA"] + [f"Grade_{g}" for g in MERIT_GRADE_ORDER]
    for _, part in summary.groupby("Branch", dropna=False):
        part = part.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")
        ranks = []
        prev = None
        rank = 0
        for i, (_, row) in enumerate(part.iterrows(), start=1):
            key = tuple(row[c] for c in sort_cols)
            if key != prev:
                rank = i
                prev = key
            ranks.append(rank)
        part = part.copy()
        part["Rank"] = ranks
        ranked_parts.append(part)
    summary = pd.concat(ranked_parts, ignore_index=True)
    return summary, detail


def _feature_criteria_table():
    """Rules behind each column in the combined overview."""
    return pd.DataFrame(
        [
            {
                "Feature": "Overall CGPA",
                "Shown in table": "CGPA, Rank, Uncleared semesters",
                "Criteria": (
                    "CGPA = sum(SGPA used × semester credits) / sum(credits), half-up to 2 decimals. "
                    "Every semester for the same roll is included, including an earlier branch. "
                    "Each semester uses its own printed SGPA and its own printed credits. "
                    "If that result has a reappear or a blank SGPA, that SGPA is 0 and those credits stay in the total. "
                    "Rank is higher CGPA, then more O, then A+, then A."
                ),
            },
            {
                "Feature": "Pass / Fail",
                "Shown in table": "Pass/Fail",
                "Criteria": "Latest published result for that roll. Pass if SGPA is 5.0 or above. Missing SGPA counts as Fail.",
            },
            {
                "Feature": "At-risk",
                "Shown in table": "At-risk, Risk level, Fail subjects, Risk reason",
                "Criteria": (
                    "Checked on the latest result only. At-risk if SGPA is below 5.0 or fail/absent subjects are 2 or more. "
                    "High if SGPA is below 4.0 or fail/absent subjects are 4 or more. Otherwise Medium."
                ),
            },
            {
                "Feature": "Section comparison",
                "Shown in table": "Section",
                "Criteria": "Section-list URN matched to university roll. The same roll keeps the same section in every semester. Unmapped if the roll is not on a section list.",
            },
            {
                "Feature": "Subject heatmap",
                "Shown in table": "Weakest subject",
                "Criteria": (
                    "Fail rate = fail or absent grades ÷ students who have a grade in that subject, on the latest results. "
                    "Weakest subject is the highest fail rate in the branch among subjects with at least 20 graded students."
                ),
            },
        ]
    )


def build_combined_overview(cgpa_view, result_df, section_df):
    """One student row with CGPA, latest SGPA, section, at-risk, and branch subject risk."""
    overview = cgpa_view.copy()
    if result_df is None or result_df.empty or "Roll_No" not in result_df.columns:
        return overview

    scoped = result_df
    if "Roll_No" in overview.columns and "Roll_No" in scoped.columns:
        scoped = scoped[scoped["Roll_No"].astype(str).isin(set(overview["Roll_No"].astype(str)))]
    latest = _latest_result_per_student(scoped)
    if latest is None or latest.empty:
        return overview

    latest = latest.copy()
    fail_counts = pd.Series(0, index=latest.index, dtype=int)
    groups = latest.groupby("Branch") if "Branch" in latest.columns else [(None, latest)]
    for _, group in groups:
        subjects = _subject_cols_for_frame(group)
        if subjects:
            fail_counts.loc[group.index] = group.apply(lambda row: _count_fails(row, subjects), axis=1)
    latest["Fail subjects"] = fail_counts.astype(int)
    latest["Latest SGPA"] = (
        pd.to_numeric(latest["SGPA"], errors="coerce") if "SGPA" in latest.columns else pd.NA
    )

    status, risk, level, reason = [], [], [], []
    for sgpa, fails in zip(latest["Latest SGPA"], latest["Fail subjects"]):
        why = []
        if pd.isna(sgpa):
            why.append("Missing SGPA")
        elif float(sgpa) < PASSING_SGPA_THRESHOLD:
            why.append(f"SGPA < {PASSING_SGPA_THRESHOLD}")
        if int(fails) >= 2:
            why.append(f"{int(fails)} fail/absent subjects")
        status.append("Pass" if pd.notna(sgpa) and float(sgpa) >= PASSING_SGPA_THRESHOLD else "Fail")
        if not why:
            risk.append("No")
            level.append("OK")
            reason.append("")
        else:
            risk.append("Yes")
            high = (pd.notna(sgpa) and float(sgpa) < 4.0) or int(fails) >= 4
            level.append("High" if high else "Medium")
            reason.append("; ".join(why))
    latest["Pass/Fail"] = status
    latest["At-risk"] = risk
    latest["Risk level"] = level
    latest["Risk reason"] = reason

    tagged = attach_sections(latest, section_df)
    if "Section" not in tagged.columns:
        tagged["Section"] = "Unmapped"
    tagged["Section"] = tagged["Section"].fillna("Unmapped")

    weakest = {}
    matrix = build_subject_fail_matrix(latest, by_branch=True)
    if not matrix.empty:
        eligible = matrix[matrix["Graded_Students"] >= 20]
        pick = eligible if not eligible.empty else matrix
        for branch, part in pick.groupby("Branch"):
            top = part.sort_values(["Fail_Rate_%", "Fail_Count"], ascending=False).iloc[0]
            weakest[str(branch)] = f"{top['Subject']} ({top['Fail_Rate_%']}%)"

    keep_cols = [
        "Roll_No", "Latest SGPA", "Pass/Fail", "Fail subjects",
        "At-risk", "Risk level", "Risk reason",
    ]
    keep = latest[keep_cols].copy()
    section_lookup = tagged[["Roll_No", "Section"]].drop_duplicates(subset=["Roll_No"])
    keep = keep.merge(section_lookup, on="Roll_No", how="left")
    keep["_roll"] = keep["Roll_No"].astype(str)
    keep = keep.drop(columns=["Roll_No"])

    overview["_roll"] = overview["Roll_No"].astype(str)
    overview = overview.merge(keep, on="_roll", how="left").drop(columns=["_roll"])
    if "Branch" in overview.columns:
        branch_key = overview["Branch"].astype(str)
        overview["Weakest subject"] = branch_key.map(weakest)
    overview["Section"] = overview["Section"].fillna("Unmapped") if "Section" in overview.columns else "Unmapped"
    return overview


def _pass_fail_counts(frame):
    """Pass is SGPA of 5.0 or more. A blank SGPA counts as a fail."""
    total = len(frame)
    if total == 0 or "SGPA" not in frame.columns:
        return total, 0, total, None
    sgpa = pd.to_numeric(frame["SGPA"], errors="coerce")
    passed = int((sgpa >= PASSING_SGPA_THRESHOLD).sum())
    failed = total - passed
    average = round(float(sgpa.mean()), 2) if sgpa.notna().any() else None
    return total, passed, failed, average


def semester_performance_table(df):
    """Average SGPA and pass rate for each semester that has result rows."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "Semester" in work.columns:
        work["_sem"] = pd.to_numeric(work["Semester"], errors="coerce")
    elif "Semester_Label" in work.columns:
        work["_sem"] = pd.to_numeric(work["Semester_Label"], errors="coerce")
    else:
        return pd.DataFrame()
    rows = []
    for sem, group in work.groupby("_sem", dropna=False):
        total, passed, failed, average = _pass_fail_counts(group)
        if total == 0:
            continue
        label = "Unknown" if pd.isna(sem) else str(int(sem))
        rows.append({
            "Semester": label,
            "Students": total,
            "Avg SGPA": average,
            "Pass %": round(passed / total * 100, 1),
            "Fail %": round(failed / total * 100, 1),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_ord"] = pd.to_numeric(out["Semester"], errors="coerce")
    return out.sort_values("_ord").drop(columns="_ord")


def subject_failure_roster(df):
    """One row per student subject graded F or Ab (and the other fail tokens)."""
    if df is None or df.empty:
        return pd.DataFrame()
    records = []
    if "Branch" in df.columns:
        groups = df.groupby("Branch", sort=True)
    else:
        groups = [("All", df)]
    for branch, group in groups:
        subjects = _subject_cols_for_frame(group)
        for subject in subjects:
            series = group[subject]
            failed = series[series.apply(_is_fail_grade)]
            for idx, grade in failed.items():
                row = group.loc[idx]
                records.append({
                    "Name": row.get("Name") if "Name" in group.columns else None,
                    "Roll_No": row.get("Roll_No") if "Roll_No" in group.columns else None,
                    "Branch": branch,
                    "Semester": row.get("Semester") if "Semester" in group.columns else None,
                    "Subject": _clean_subject_label(subject),
                    "Grade": grade,
                })
    return pd.DataFrame(records)

