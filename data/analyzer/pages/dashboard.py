"""College executive view: totals, branch and semester charts, subject difficulty, reports."""

import io

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from analyzer.analytics import (
    PASSING_SGPA_THRESHOLD,
    _branch_summary,
    _pass_fail_counts,
    build_subject_fail_matrix,
    semester_performance_table,
    subject_failure_roster,
)
from analyzer.catalog import _sorted_sem_labels, _sorted_year_labels
from analyzer.nav import FEATURE_VIEWS, selected_view
from analyzer.store import _data_mtime_signature, load_all_branch_datasets
from analyzer.ui import kpi_cards, show_chart


def _pdf_table(rows):
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#201060")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E4D3B8")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FBF8F4")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _executive_pdf(title, kpis, insights, branch_table, semester_table, subject_table):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Guru Nanak Dev Engineering College, Ludhiana", styles["Title"]),
        Paragraph(title, styles["Heading2"]),
        Spacer(1, 8),
    ]
    for line in kpis + insights:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 10))
    for heading, frame in (
        ("Branch comparison", branch_table),
        ("Semester comparison", semester_table),
        ("Subject difficulty (F and Ab count as failures)", subject_table),
    ):
        story.append(Paragraph(heading, styles["Heading3"]))
        if frame is None or frame.empty:
            story.append(Paragraph("No rows for this filter.", styles["Normal"]))
        else:
            show = frame.head(15)
            rows = [list(show.columns)] + show.astype(str).values.tolist()
            story.append(_pdf_table(rows))
        story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Pass means SGPA of 5.0 or above. A blank SGPA is counted as a fail. "
        "The full student list is in the CSV download.",
        styles["Italic"],
    ))
    doc.build(story)
    return buffer.getvalue()


def render_dashboard_page():
    inner = selected_view(FEATURE_VIEWS["Executive Dashboard"])
    st.title(inner)
    st.caption(
        "Executive Dashboard · College summary for one semester. "
        "F and Ab both count as a failure."
    )

    signature = _data_mtime_signature()
    combined, load_errors = load_all_branch_datasets(signature)
    if load_errors:
        with st.expander("Dataset load notes", expanded=False):
            for err in load_errors:
                st.write(f"- {err}")
    if combined.empty:
        st.warning("Add result CSV or PDF files first.")
        return

    batch_opts = ["All"]
    if "Batch_Year_Label" in combined.columns:
        batch_opts += _sorted_year_labels(combined["Batch_Year_Label"].dropna().astype(str).tolist())
    sem_opts = ["All"]
    if "Semester_Label" in combined.columns:
        sem_opts += _sorted_sem_labels(combined["Semester_Label"].dropna().astype(str).tolist())
    default_sem = sem_opts[-1] if len(sem_opts) > 1 else "All"

    f1, f2 = st.columns(2)
    batch_pick = f1.selectbox("Batch year", batch_opts, key="dash_batch")
    sem_pick = f2.selectbox(
        "Semester",
        sem_opts,
        index=sem_opts.index(default_sem),
        key="dash_sem",
    )

    batch_view = combined
    if batch_pick != "All" and "Batch_Year_Label" in combined.columns:
        batch_view = combined[combined["Batch_Year_Label"].astype(str) == batch_pick]
    view = batch_view
    if sem_pick != "All" and "Semester_Label" in batch_view.columns:
        view = batch_view[batch_view["Semester_Label"].astype(str) == sem_pick]
    if view.empty:
        st.info("No result rows match this batch and semester.")
        return

    total, passed, failed, average = _pass_fail_counts(view)
    pass_pct = round(passed / total * 100, 1) if total else 0
    fail_pct = round(failed / total * 100, 1) if total else 0

    kpi_cards([
        ("Total students", f"{total:,}"),
        ("Average SGPA", f"{average:.2f}" if average is not None else "—"),
        ("Pass percentage", f"{pass_pct:.1f}%"),
        ("Fail percentage", f"{fail_pct:.1f}%"),
    ])
    st.caption(
        f"Pass is SGPA {PASSING_SGPA_THRESHOLD:.1f} or above. "
        "Students with no SGPA are included in the fail percentage. "
        "Average SGPA uses only published SGPA values."
    )

    branch_table = _branch_summary(view)
    semester_table = semester_performance_table(batch_view)
    subject_matrix = build_subject_fail_matrix(view, by_branch=True)
    if not subject_matrix.empty:
        subject_matrix = subject_matrix.sort_values(
            ["Fail_Rate_%", "Fail_Count"], ascending=[False, False]
        )

    eligible = subject_matrix[subject_matrix["Graded_Students"] >= 20] if not subject_matrix.empty else subject_matrix
    insight_subjects = eligible if eligible is not None and not eligible.empty else subject_matrix
    hardest = None
    if insight_subjects is not None and not insight_subjects.empty:
        hardest = insight_subjects.iloc[0]
    best_branch = None
    if not branch_table.empty and branch_table["Pass Rate %"].notna().any():
        sized = branch_table[branch_table["Students"] >= 20]
        pool = sized if not sized.empty else branch_table
        best_branch = pool.loc[pool["Pass Rate %"].idxmax()]

    c1, c2 = st.columns(2)
    if hardest is not None:
        c1.info(
            f"**Highest-failure subject:** {hardest['Subject']} in {hardest['Branch']} — "
            f"{int(hardest['Fail_Count'])} failures out of {int(hardest['Graded_Students'])} graded "
            f"({hardest['Fail_Rate_%']}%). F and Ab are both failures."
        )
    else:
        c1.info("No subject grades in this filter yet.")
    if best_branch is not None:
        c2.success(
            f"**Highest pass rate:** {best_branch['Branch']} at {best_branch['Pass Rate %']}% "
            f"({int(best_branch['Students'])} students, average SGPA {best_branch['Avg SGPA']})."
        )

    if inner == "Branch Comparison":
        st.caption(f"Branches compared for batch {batch_pick}, semester {sem_pick}.")
        if branch_table.empty:
            st.info("No branch rows.")
        else:
            plot_df = branch_table.dropna(subset=["Avg SGPA"])
            left, right = st.columns(2)
            fig_avg = px.bar(
                plot_df,
                x="Branch",
                y="Avg SGPA",
                text="Avg SGPA",
                color="Branch",
                title="Average SGPA by branch",
            )
            fig_avg.update_layout(showlegend=False)
            show_chart(fig_avg, left)
            fig_pass = px.bar(
                branch_table.dropna(subset=["Pass Rate %"]),
                x="Branch",
                y="Pass Rate %",
                text="Pass Rate %",
                color="Branch",
                title="Pass percentage by branch",
            )
            fig_pass.update_layout(showlegend=False)
            show_chart(fig_pass, right)
            st.dataframe(
                branch_table[["Branch", "Students", "Avg SGPA", "Pass Rate %", "Missing SGPA"]],
                use_container_width=True,
                hide_index=True,
            )

    if inner == "Semester Comparison":
        st.caption("Every semester in the selected batch. The semester filter above does not hide this chart.")
        if semester_table.empty:
            st.info("Semester numbers were not detected on these result files.")
        else:
            fig_sem = px.line(
                semester_table,
                x="Semester",
                y=["Avg SGPA", "Pass %"],
                markers=True,
                title="Performance across semesters",
            )
            show_chart(fig_sem, use_container_width=True)
            st.dataframe(semester_table, use_container_width=True, hide_index=True)

    if inner == "Subject Difficulty":
        st.caption("Failure rate = F, F(Int.), F(Ext.), and Ab ÷ students who have a grade in that subject.")
        if subject_matrix.empty:
            st.info("No subject grades in this filter.")
        else:
            top = subject_matrix.head(12)
            fig_sub = px.bar(
                top.sort_values("Fail_Rate_%"),
                x="Fail_Rate_%",
                y="Subject",
                color="Branch",
                orientation="h",
                text="Fail_Count",
                title="Highest failure rates",
                hover_data=["Graded_Students"],
            )
            show_chart(fig_sub, use_container_width=True)
            show_cols = ["Branch", "Subject", "Fail_Count", "Graded_Students", "Fail_Rate_%"]
            st.dataframe(subject_matrix[show_cols], use_container_width=True, hide_index=True)

            roster = subject_failure_roster(view)
            if not roster.empty:
                choices = (
                    subject_matrix[["Branch", "Subject"]]
                    .drop_duplicates()
                    .apply(lambda r: f"{r['Branch']} · {r['Subject']}", axis=1)
                    .tolist()
                )
                picked = st.selectbox("Failed students in", choices, key="dash_fail_subject")
                branch_name, subject_name = picked.split(" · ", 1)
                people = roster[
                    (roster["Branch"].astype(str) == branch_name)
                    & (roster["Subject"].astype(str) == subject_name)
                ]
                st.dataframe(people, use_container_width=True, hide_index=True)

    if inner == "Reports":
        students = view.copy()
        sgpa = pd.to_numeric(students["SGPA"], errors="coerce") if "SGPA" in students.columns else pd.Series(dtype=float)
        students["Pass/Fail"] = [
            "Pass" if pd.notna(value) and float(value) >= PASSING_SGPA_THRESHOLD else "Fail"
            for value in sgpa
        ]
        export_cols = [
            c for c in ["Name", "Roll_No", "Branch", "Semester", "Batch_Year_Label", "SGPA", "Pass/Fail"]
            if c in students.columns
        ]
        student_csv = students[export_cols].to_csv(index=False)
        subject_csv = (
            subject_matrix[["Branch", "Subject", "Fail_Count", "Graded_Students", "Fail_Rate_%"]].to_csv(index=False)
            if not subject_matrix.empty else "Branch,Subject,Fail_Count,Graded_Students,Fail_Rate_%\n"
        )
        insight_lines = []
        if hardest is not None:
            insight_lines.append(
                f"Highest-failure subject: {hardest['Subject']} ({hardest['Branch']}), "
                f"{hardest['Fail_Rate_%']}%."
            )
        if best_branch is not None:
            insight_lines.append(
                f"Highest pass rate: {best_branch['Branch']}, {best_branch['Pass Rate %']}%."
            )
        pdf_bytes = _executive_pdf(
            f"Executive report · Batch {batch_pick} · Semester {sem_pick}",
            [
                f"Students: {total}",
                f"Average SGPA: {average if average is not None else '—'}",
                f"Pass percentage: {pass_pct}%",
                f"Fail percentage: {fail_pct}%",
            ],
            insight_lines,
            branch_table[["Branch", "Students", "Avg SGPA", "Pass Rate %"]] if not branch_table.empty else branch_table,
            semester_table,
            subject_matrix[["Branch", "Subject", "Fail_Count", "Graded_Students", "Fail_Rate_%"]]
            if not subject_matrix.empty else subject_matrix,
        )
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Download student results (CSV)",
            data=student_csv,
            file_name=f"gndec_students_batch{batch_pick}_sem{sem_pick}.csv",
            mime="text/csv",
        )
        d2.download_button(
            "Download subject difficulty (CSV)",
            data=subject_csv,
            file_name=f"gndec_subjects_batch{batch_pick}_sem{sem_pick}.csv",
            mime="text/csv",
        )
        d3.download_button(
            "Download executive report (PDF)",
            data=pdf_bytes,
            file_name=f"gndec_executive_batch{batch_pick}_sem{sem_pick}.pdf",
            mime="application/pdf",
        )
