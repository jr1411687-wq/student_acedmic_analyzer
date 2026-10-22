import streamlit as st
import pandas as pd
import plotly.express as px

from analyzer.analytics import _prepare_college_view, build_subject_fail_matrix
from analyzer.catalog import _sorted_year_labels
from analyzer.files import discover_section_list_files
from analyzer.nav import FEATURE_VIEWS, selected_view
from analyzer.sections import _branch_match_key, _normalize_roll, _section_roll_coverage, _section_summary, _section_summary_by_semester, attach_sections, enrich_section_map_branches, load_all_section_maps, parse_section_list_meta
from analyzer.store import _data_mtime_signature, load_all_branch_datasets
from analyzer.ui import kpi_cards, show_chart

def render_section_comparison_page():
    """
    Compare Section A vs B using URN→Section from section lists.

    Section is permanent for a student (same URN keeps the same section in every
    semester). The section-list file may say “sem 3”, but mapping is applied to
    ALL result PDFs/CSVs for that branch; then you filter which result semester
    to compare.
    """
    inner = selected_view(FEATURE_VIEWS["Section Comparison"])
    st.title(inner)
    st.caption(
        "Section Comparison · Section list URN → Section is applied to all result files for the branch. "
        "A student keeps the same section in Sem 1–4."
    )

    sig = _data_mtime_signature()
    section_files = discover_section_list_files()
    if not section_files:
        st.warning(
            "No section list found. Add a file named like "
            "`section list of ece sem 3 (batch 2024).pdf` or "
            "`section_list_ECE_sem3_batch2024.csv` (columns: URN, Section) into `data/all results/`."
        )
        return

    with st.spinner("Auto-detecting section list from PDF + matching result rolls…"):
        section_map, sec_errors = load_all_section_maps(sig)

    with st.spinner("Loading all result datasets…"):
        combined, load_errors = load_all_branch_datasets(sig)

    if load_errors:
        with st.expander("Result load notes", expanded=False):
            for err in load_errors:
                st.write(f"- {err}")

    # Auto-tag Branch on each URN by matching University Roll No in results
    if not section_map.empty:
        section_map = enrich_section_map_branches(section_map, combined if not combined.empty else None)

    with st.expander(f"Detected section lists ({len(section_files)})", expanded=True):
        for f in section_files:
            meta = parse_section_list_meta(f)
            st.write(f"- **{f.name}** (auto-parsed from PDF text / layout)")
            if not section_map.empty:
                st.caption(
                    f"Parsed {len(section_map):,} URNs · sections "
                    f"{sorted(section_map['Section'].dropna().astype(str).unique())} · "
                    f"branches matched: "
                    f"{sorted(section_map['Branch'].dropna().astype(str).unique())}"
                )
        if sec_errors:
            st.caption("Notes:")
            for err in sec_errors:
                st.write(f"- {err}")

    if section_map.empty:
        st.error(
            "Section list PDF was found but could not be auto-parsed. "
            "Use a digital (text) PDF section sheet, or install OCR deps for scans."
        )
        if sec_errors:
            for err in sec_errors:
                st.caption(err)
        return

    # Branch from section lists (after result matching)
    branches = sorted(
        {str(b) for b in section_map["Branch"].dropna().astype(str) if str(b) and str(b).lower() != "unknown"}
    )
    if not branches:
        branches = sorted({str(b) for b in section_map["Branch"].dropna().astype(str)})
    if not branches:
        st.error("Section list parsed, but no branch could be matched from result Roll_No / URN.")
        return
    branch = st.sidebar.selectbox("Branch", branches, key="sec_cmp_branch")
    scoped_map = section_map[section_map["Branch"].map(_branch_match_key) == _branch_match_key(branch)].copy()
    scoped_map["URN"] = scoped_map["URN"].map(_normalize_roll)
    scoped_map = scoped_map.dropna(subset=["URN"]).drop_duplicates(subset=["URN"], keep="first")

    detail_level = st.sidebar.radio(
        "Compare by",
        ["Section (A / B)", "Subsection (A1 / A2 / B1 / B2)"],
        key="sec_cmp_level",
    )
    group_col = "Subsection" if "Subsection" in detail_level else "Section"

    if combined.empty:
        st.warning("No result datasets loaded.")
        return

    bkey = _branch_match_key(branch)
    results_all = combined[combined["Branch"].map(_branch_match_key) == bkey].copy()
    if results_all.empty:
        st.warning(f"No result rows for branch **{branch}**.")
        return

    # Optional batch filter from section list / results
    batch_opts = ["All"]
    if "Batch_Year_Label" in results_all.columns:
        batch_opts += _sorted_year_labels(results_all["Batch_Year_Label"].dropna().astype(str).tolist())
    # Prefer section-list batch as default when present
    default_batch = "All"
    if "Batch_Year" in scoped_map.columns and scoped_map["Batch_Year"].notna().any():
        by = str(int(scoped_map["Batch_Year"].dropna().iloc[0])) if pd.api.types.is_numeric_dtype(scoped_map["Batch_Year"]) else str(scoped_map["Batch_Year"].dropna().iloc[0])
        if by in batch_opts:
            default_batch = by
    batch_pick = st.sidebar.selectbox(
        "Batch year",
        batch_opts,
        index=batch_opts.index(default_batch) if default_batch in batch_opts else 0,
        key="sec_cmp_batch",
        help="Usually match the batch on the section list (e.g. 2024).",
    )
    if batch_pick != "All" and "Batch_Year_Label" in results_all.columns:
        results_all = results_all[results_all["Batch_Year_Label"].astype(str) == str(batch_pick)]

    # Tag EVERY result row for this branch with Section via URN (all semesters)
    tagged_all = attach_sections(results_all, scoped_map, branch=branch, semester=None)

    # Coverage: section-list URNs vs each result semester
    coverage = _section_roll_coverage(scoped_map, results_all)
    st.subheader("URN match check (section list ↔ results)")
    st.caption(
        f"Section list has **{len(scoped_map)} unique URNs** for {branch}. "
        "Same URN must appear in result PDFs/CSVs to compare that student."
    )
    if not coverage.empty:
        st.dataframe(coverage, use_container_width=True, hide_index=True)

    # Result semester filter (does NOT change section assignment)
    sem_in_results = sorted(
        {int(s) for s in pd.to_numeric(tagged_all.get("Semester"), errors="coerce").dropna().unique()}
    )
    sem_opts = ["All"] + [str(s) for s in sem_in_results]
    sem_pick = st.sidebar.selectbox(
        "Result semester",
        sem_opts,
        key="sec_cmp_result_sem",
        help="Filter which exam results to compare. Section labels still come from the section list (all semesters).",
    )

    tagged = tagged_all
    if sem_pick != "All" and "Semester" in tagged.columns:
        tagged = tagged[pd.to_numeric(tagged["Semester"], errors="coerce") == float(sem_pick)]

    if tagged.empty:
        st.warning(f"No {branch} results for semester **{sem_pick}**.")
        return

    # For a single semester: one row per student. For All: keep all semester rows
    # so per-semester charts stay accurate; also build a latest-only headcount view.
    if sem_pick == "All":
        view_latest, result_rows, unique_students, deduped = _prepare_college_view(tagged)
        work_rows = tagged[tagged["Section"].astype(str) != "Unmapped"].copy()
        mapped_latest = view_latest[view_latest["Section"].astype(str) != "Unmapped"].copy()
        unmapped_n = int((view_latest["Section"].astype(str) == "Unmapped").sum())
        mapped_for_kpis = mapped_latest
    else:
        view_latest, result_rows, unique_students, deduped = _prepare_college_view(tagged)
        work_rows = view_latest[view_latest["Section"].astype(str) != "Unmapped"].copy()
        mapped_for_kpis = work_rows
        unmapped_n = int((view_latest["Section"].astype(str) == "Unmapped").sum())
        deduped = False

    if work_rows.empty or mapped_for_kpis.empty:
        st.error(
            f"Section list has {len(scoped_map)} URNs, but none matched result Roll_No for "
            f"**{branch}** (batch {batch_pick}, result sem {sem_pick}). "
            "Confirm the list uses University Roll No (URN), not CRN."
        )
        st.dataframe(scoped_map.head(20), use_container_width=True, hide_index=True)
        return

    # Group key
    def _assign_group(df):
        out = df.copy()
        if group_col == "Subsection" and "Subsection" in out.columns:
            sub = out["Subsection"]
            out["_Group"] = (sub.iloc[:, -1] if isinstance(sub, pd.DataFrame) else sub).astype(str)
        else:
            sec = out["Section"]
            out["_Group"] = (sec.iloc[:, -1] if isinstance(sec, pd.DataFrame) else sec).astype(str)
        return out

    work = _assign_group(work_rows if sem_pick == "All" else mapped_for_kpis)
    kpi_work = _assign_group(mapped_for_kpis)
    summary_src = kpi_work.drop(columns=["Section"], errors="ignore").rename(columns={"_Group": "Section"})
    summary = _section_summary(summary_src)

    avg = pd.to_numeric(mapped_for_kpis["SGPA"], errors="coerce").mean() if "SGPA" in mapped_for_kpis.columns else None
    kpi_cards([
        ("Mapped students", f"{len(mapped_for_kpis):,}"),
        ("Sections", kpi_work["_Group"].nunique()),
        ("Avg SGPA", f"{avg:.2f}" if pd.notna(avg) else "—"),
        ("Unmapped in results", unmapped_n),
    ])

    st.caption(
        f"{branch} · batch {batch_pick} · result sem **{sem_pick}** · "
        f"section-list URNs: {len(scoped_map):,} · "
        f"mapped: {len(mapped_for_kpis):,}"
        + (f" · {result_rows:,} result rows across semesters" if sem_pick == "All" else "")
    )

    if summary.empty:
        st.info("Not enough mapped students with Section labels.")
        return

    if inner == "SGPA by Section":
        title_bit = f"result Sem {sem_pick}" if sem_pick != "All" else "latest result per student"
        fig = px.bar(
            summary.sort_values("Section"),
            x="Section",
            y="Avg SGPA",
            text="Avg SGPA",
            color="Pass Rate %",
            color_continuous_scale=["#B91C1C", "#D97706", "#15803D"],
            title=f"{branch} — Avg SGPA by section ({title_bit})",
            hover_data=["Students", "Highest SGPA", "Lowest SGPA"],
        )
        fig.update_traces(textposition="outside")
        show_chart(fig, use_container_width=True)

        if "SGPA" in kpi_work.columns:
            fig_box = px.box(
                kpi_work.dropna(subset=["SGPA"]),
                x="_Group",
                y="SGPA",
                color="_Group",
                title="SGPA distribution",
                labels={"_Group": "Section"},
            )
            fig_box.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="Pass = 5.0")
            show_chart(fig_box, use_container_width=True)

        st.dataframe(summary, use_container_width=True, hide_index=True)

    if inner == "By Semester":
        st.caption("Same URN keeps the same section in every semester. Below: section stats from each result file.")
        by_sem = _section_summary_by_semester(_assign_group(work_rows), group_col="_Group")
        if by_sem.empty:
            st.info("No per-semester section stats available.")
        else:
            fig_s = px.bar(
                by_sem.sort_values(["Semester", "Section"]),
                x="Semester",
                y="Avg SGPA",
                color="Section",
                barmode="group",
                text="Avg SGPA",
                title=f"{branch} — Section Avg SGPA across semesters",
                hover_data=["Students", "Pass Rate %"],
            )
            fig_s.update_traces(textposition="outside")
            show_chart(fig_s, use_container_width=True)

            fig_pr = px.line(
                by_sem.sort_values(["Semester", "Section"]),
                x="Semester",
                y="Pass Rate %",
                color="Section",
                markers=True,
                title=f"{branch} — Section pass rate across semesters",
            )
            show_chart(fig_pr, use_container_width=True)
            st.dataframe(by_sem, use_container_width=True, hide_index=True)
            st.download_button(
                "Download per-semester section summary",
                data=by_sem.to_csv(index=False),
                file_name=f"section_by_semester_{branch}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if inner == "Pass Rate":
        fig_p = px.bar(
            summary.sort_values("Section"),
            x="Section",
            y="Pass Rate %",
            text="Pass Rate %",
            color="Section",
            title="Pass / clearance rate by section (SGPA ≥ 5)",
        )
        fig_p.update_traces(textposition="outside")
        show_chart(fig_p, use_container_width=True)
        if "Gender" in kpi_work.columns:
            g = kpi_work.groupby(["_Group", "Gender"]).size().reset_index(name="Count")
            fig_g = px.bar(
                g,
                x="_Group",
                y="Count",
                color="Gender",
                barmode="stack",
                title="Gender mix by section",
                labels={"_Group": "Section"},
            )
            show_chart(fig_g, use_container_width=True)

    if inner == "Subject Pressure":
        fail_src = kpi_work if sem_pick != "All" else work
        fail_matrix = build_subject_fail_matrix(fail_src.assign(Branch=fail_src["_Group"]), by_branch=True)
        if fail_matrix.empty:
            st.info("No subject grades available for section heat comparison.")
        else:
            top = (
                fail_matrix.sort_values(["Branch", "Fail_Rate_%"], ascending=[True, False])
                .groupby("Branch", group_keys=False)
                .head(5)
            )
            fig_f = px.bar(
                top,
                x="Fail_Rate_%",
                y="Subject",
                color="Branch",
                orientation="h",
                barmode="group",
                title="Top subject fail rates by section",
                labels={"Branch": "Section"},
                hover_data=["Fail_Count", "Graded_Students"],
            )
            show_chart(fig_f, use_container_width=True)
            st.dataframe(
                fail_matrix.sort_values(["Branch", "Fail_Rate_%"], ascending=[True, False])[
                    [c for c in ["Branch", "Subject", "Fail_Count", "Graded_Students", "Fail_Rate_%"] if c in fail_matrix.columns]
                ].rename(columns={"Branch": "Section"}),
                use_container_width=True,
                hide_index=True,
            )

    if inner == "Students":
        show = work.copy() if sem_pick == "All" else kpi_work.copy()
        cols = [
            c for c in [
                "Section", "Subsection", "Roll_No", "Name", "Gender", "SGPA", "Credits_Earned",
                "Semester", "Semester_Label", "Batch_Year_Label",
            ] if c in show.columns
        ]
        sort_cols = [c for c in ["Semester", "Section", "Subsection", "SGPA"] if c in cols]
        ascending = [c != "SGPA" for c in sort_cols]
        st.dataframe(
            show[cols].sort_values(sort_cols, ascending=ascending) if sort_cols else show[cols],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download section-tagged results (CSV)",
            data=show[cols].to_csv(index=False),
            file_name=f"section_comparison_{branch}_sem{sem_pick}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download section summary (CSV)",
            data=summary.to_csv(index=False),
            file_name=f"section_summary_{branch}_sem{sem_pick}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("**How this works**")
    st.markdown(
        "- Drop a **section list PDF** into `all results/` (name contains `section list`)\n"
        "- App **auto-parses** CRN/URN columns and Section A/B (A1/A2/…) from the PDF\n"
        "- Each URN is matched to **University Roll No** in all result PDFs/CSVs → Branch auto-detected\n"
        "- Same URN keeps the same section across every semester\n"
        "- Use **By semester** to compare Section A vs B on Sem 1–4 results"
    )

