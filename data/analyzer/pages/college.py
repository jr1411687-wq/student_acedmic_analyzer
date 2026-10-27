import streamlit as st
import pandas as pd
import plotly.express as px

from analyzer.analytics import _branch_summary, _prepare_college_view
from analyzer.catalog import apply_year_sem_filter, sidebar_year_sem_filters
from analyzer.nav import FEATURE_VIEWS, selected_view
from analyzer.store import _data_mtime_signature, build_dataset_catalog, load_all_branch_datasets
from analyzer.ui import kpi_cards, show_chart

def render_college_comparison_page():
    """College-level comparison across all branch CSVs/PDFs."""
    inner = selected_view(FEATURE_VIEWS["College Comparison"])
    st.title(inner)
    st.caption("College Comparison · Compare SGPA, gender ratio, and top performers across all branches")

    sig = _data_mtime_signature()
    catalog = build_dataset_catalog(sig)
    scope = sidebar_year_sem_filters(catalog, key_prefix="college")
    combined, load_errors = load_all_branch_datasets(sig)
    combined = apply_year_sem_filter(combined, scope)

    if load_errors:
        with st.expander("Some datasets could not be loaded", expanded=False):
            for err in load_errors:
                st.write(f"- {err}")

    if combined.empty:
        st.warning("No students match the selected Batch / Academic year / Semester. Try All, or add more result files.")
        return

    # When Academic year / Semester = All, same roll appears in Sem1–4 → don't inflate headcount
    view, result_rows, unique_students, deduped = _prepare_college_view(combined)
    summary = _branch_summary(view)
    total_students = unique_students
    n_branches = view["Branch"].nunique() if "Branch" in view.columns else 0
    overall_avg = pd.to_numeric(view["SGPA"], errors="coerce").mean() if "SGPA" in view.columns else None

    if not summary.empty and summary["Avg SGPA"].notna().any():
        best = summary.loc[summary["Avg SGPA"].idxmax()]
        top_branch = f"{best['Branch']} ({best['Avg SGPA']})"
    else:
        top_branch = "N/A"
    kpi_cards([
        ("Branches Compared", n_branches),
        ("Total Students", total_students),
        ("College Avg SGPA", f"{overall_avg:.2f}" if pd.notna(overall_avg) else "N/A"),
        ("Top Branch (Avg SGPA)", top_branch),
    ])

    if deduped:
        st.info(
            f"Showing **{unique_students:,} unique students** (latest semester each). "
            f"Filter includes **{result_rows:,} result rows** across multiple semesters — "
            f"the same student is not counted once per semester."
        )
    elif result_rows != unique_students:
        st.caption(f"{unique_students:,} unique students · {result_rows:,} result rows")

    st.divider()

    # Use deduped view for fair branch comparison charts/tables
    combined = view
    if inner == "SGPA Comparison":
        st.subheader("SGPA by Branch")
        if "SGPA" not in combined.columns or combined["SGPA"].isna().all():
            st.info("No SGPA data available across datasets.")
        else:
            plot_df = summary.dropna(subset=["Avg SGPA"])
            c1, c2 = st.columns(2)
            with c1:
                fig_avg = px.bar(
                    plot_df,
                    x="Branch",
                    y="Avg SGPA",
                    color="Branch",
                    text="Avg SGPA",
                    title="Average SGPA by Branch",
                )
                fig_avg.update_traces(textposition="outside")
                fig_avg.update_layout(showlegend=False)
                show_chart(fig_avg, use_container_width=True)
            with c2:
                fig_high = px.bar(
                    plot_df,
                    x="Branch",
                    y=["Highest SGPA", "Lowest SGPA"],
                    barmode="group",
                    title="Highest vs Lowest SGPA by Branch",
                )
                show_chart(fig_high, use_container_width=True)

            if "Pass Rate %" in summary.columns and summary["Pass Rate %"].notna().any():
                fig_pass = px.bar(
                    summary.dropna(subset=["Pass Rate %"]),
                    x="Branch",
                    y="Pass Rate %",
                    color="Branch",
                    text="Pass Rate %",
                    title="Pass Rate by Branch (SGPA ≥ 5.0)",
                )
                fig_pass.update_traces(textposition="outside")
                fig_pass.update_layout(showlegend=False)
                show_chart(fig_pass, use_container_width=True)

    if inner == "Gender Ratio":
        st.subheader("Gender Ratio by Branch")
        if "Gender" not in combined.columns:
            st.info("No gender data available.")
        else:
            ratio_df = summary[["Branch", "Male %", "Female %"]].melt(
                id_vars=["Branch"], var_name="Metric", value_name="Percentage"
            )
            fig_pct = px.bar(
                ratio_df,
                x="Branch",
                y="Percentage",
                color="Metric",
                barmode="group",
                title="Male vs Female Percentage by Branch",
            )
            show_chart(fig_pct, use_container_width=True)

            st.dataframe(
                summary[["Branch", "Students", "Male", "Female", "Unknown", "Male:Female", "Male %", "Female %"]],
                use_container_width=True,
                hide_index=True,
            )

            college_gender = combined["Gender"].value_counts()
            fig_pie = px.pie(
                names=college_gender.index,
                values=college_gender.values,
                title=f"College Overall Gender Distribution ({total_students} students)",
                hole=0.35,
                color_discrete_sequence=["#4ECDC4", "#FF6B6B", "#95A5A6"],
            )
            show_chart(fig_pie, use_container_width=True)

    if inner == "Top Performers":
        st.subheader("College Top Performers")
        if "SGPA" not in combined.columns:
            st.info("No SGPA data for ranking.")
        else:
            top_n = st.slider("Show top N students (college-wide)", 5, 50, 15, key="college_top_n")
            ranked = combined.dropna(subset=["SGPA"]).sort_values("SGPA", ascending=False).head(top_n)
            display_cols = [c for c in ["Branch", "Roll_No", "Name", "Gender", "SGPA", "Credits_Earned"] if c in ranked.columns]
            st.dataframe(ranked[display_cols], use_container_width=True, hide_index=True)

            st.subheader("Top Performer per Branch")
            per_branch = (
                combined.dropna(subset=["SGPA"])
                .sort_values("SGPA", ascending=False)
                .groupby("Branch", as_index=False)
                .first()
            )
            per_cols = [c for c in ["Branch", "Roll_No", "Name", "Gender", "SGPA", "Credits_Earned"] if c in per_branch.columns]
            st.dataframe(per_branch[per_cols].sort_values("SGPA", ascending=False), use_container_width=True, hide_index=True)

    if inner == "Overview Table":
        st.subheader("Branch Comparison Summary")
        overview = summary.drop(columns=["Graded", "Male:Female"], errors="ignore")
        st.dataframe(overview, use_container_width=True, hide_index=True)

        csv_bytes = overview.to_csv(index=False)
        st.download_button(
            "Download Branch Comparison CSV",
            data=csv_bytes,
            file_name="college_branch_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.caption("College-Level Branch Comparison | Auto-loaded from CSV/PDF files in data folder")


# =========================================================
# SECTION COMPARISON PAGE
# =========================================================

