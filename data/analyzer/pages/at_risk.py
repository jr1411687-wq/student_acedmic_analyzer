import streamlit as st
import plotly.express as px

from analyzer.analytics import PASSING_SGPA_THRESHOLD, _prepare_college_view, build_at_risk_table
from analyzer.catalog import _sorted_sem_labels, _sorted_study_year_labels, _sorted_year_labels, apply_year_sem_filter
from analyzer.nav import FEATURE_VIEWS, selected_view
from analyzer.store import _data_mtime_signature, load_all_branch_datasets
from analyzer.ui import kpi_cards, show_chart

def render_at_risk_page():
    """Mentoring-focused at-risk student list."""
    inner = selected_view(FEATURE_VIEWS["At-Risk Students"])
    st.title(inner)
    st.caption("At-Risk Students · Flag students who may need mentoring based on SGPA, fails, and credits")

    sig = _data_mtime_signature()
    combined, load_errors = load_all_branch_datasets(sig)
    if load_errors:
        with st.expander("Dataset load notes", expanded=False):
            for err in load_errors:
                st.write(f"- {err}")
    if combined.empty:
        st.warning("Add CSV/PDF files to the data folder first.")
        return

    # On-page filters for clear academic scope (same idea as Subject Heatmap)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        batch_opts = ["All"] + _sorted_year_labels(
            combined["Batch_Year_Label"].dropna().astype(str).tolist()
            if "Batch_Year_Label" in combined.columns else []
        )
        batch_pick = st.selectbox("Batch year", batch_opts, key="atrisk_batch")
    with f2:
        scoped_batch = combined
        if batch_pick != "All" and "Batch_Year_Label" in combined.columns:
            scoped_batch = combined[combined["Batch_Year_Label"].astype(str) == batch_pick]
        study_opts = ["All"] + _sorted_study_year_labels(
            scoped_batch["Study_Year_Label"].dropna().astype(str).tolist()
            if "Study_Year_Label" in scoped_batch.columns else []
        )
        study_pick = st.selectbox("Academic year", study_opts, key="atrisk_study")
    with f3:
        scoped_study = scoped_batch
        if study_pick != "All" and "Study_Year_Label" in scoped_batch.columns:
            scoped_study = scoped_batch[scoped_batch["Study_Year_Label"].astype(str) == study_pick]
        sem_opts = ["All"] + _sorted_sem_labels(
            scoped_study["Semester_Label"].dropna().astype(str).tolist()
            if "Semester_Label" in scoped_study.columns else []
        )
        sem_pick = st.selectbox("Semester", sem_opts, key="atrisk_sem")
    with f4:
        branch_opts = ["All"] + (
            sorted(scoped_study["Branch"].dropna().unique().tolist())
            if "Branch" in scoped_study.columns else []
        )
        branch_pick = st.selectbox("Branch", branch_opts, key="atrisk_branch")

    scoped = apply_year_sem_filter(
        combined,
        {
            "batch_year_label": batch_pick,
            "study_year_label": study_pick,
            "exam_year_label": "All",
            "semester_label": sem_pick,
        },
    )
    if branch_pick != "All" and "Branch" in scoped.columns:
        scoped = scoped[scoped["Branch"] == branch_pick]

    if scoped.empty:
        st.warning("No students match this Batch / Academic year / Semester / Branch. Try All.")
        return

    # Avoid counting the same roll once per semester when filters are broad
    view, result_rows, unique_students, deduped = _prepare_college_view(scoped)

    st.sidebar.subheader("At-risk rules")
    sgpa_cut = st.sidebar.slider("SGPA below", 0.0, 10.0, float(PASSING_SGPA_THRESHOLD), 0.1)
    min_fails = st.sidebar.slider("Minimum fail/absent subjects", 1, 8, 2)
    use_credit_rule = st.sidebar.checkbox("Also flag low credits", value=False)
    min_credits = (
        st.sidebar.number_input("Credits below", min_value=0, max_value=40, value=16)
        if use_credit_rule else None
    )

    at_risk_df = build_at_risk_table(
        view,
        sgpa_threshold=sgpa_cut,
        min_fails=min_fails,
        min_credits=min_credits,
    )

    scope_bits = []
    if batch_pick != "All":
        scope_bits.append(f"Batch {batch_pick}")
    if study_pick != "All":
        scope_bits.append(study_pick)
    if sem_pick != "All":
        scope_bits.append(f"Sem {sem_pick}")
    if branch_pick != "All":
        scope_bits.append(branch_pick)
    st.caption("Scope: " + (" · ".join(scope_bits) if scope_bits else "All students in data folder"))

    high = int((at_risk_df["Risk_Level"] == "High").sum()) if not at_risk_df.empty else 0
    med = int((at_risk_df["Risk_Level"] == "Medium").sum()) if not at_risk_df.empty else 0
    kpi_cards([
        ("Students scanned", unique_students),
        ("At-risk", len(at_risk_df)),
        ("High risk", high),
        ("Medium risk", med),
    ])

    if deduped:
        st.info(
            f"Scanning **{unique_students:,} unique students** (latest semester each). "
            f"Selection has **{result_rows:,} result rows** across multiple semesters — "
            f"the same student is not listed once per semester."
        )

    if at_risk_df.empty:
        st.success("No students match the current rules for this selection.")
        return

    at_risk_rate = (len(at_risk_df) / unique_students * 100) if unique_students else 0
    st.caption(f"At-risk rate: {at_risk_rate:.1f}% of scanned students")

    if inner == "Risk Overview":
        c1, c2 = st.columns(2)
        with c1:
            level_counts = at_risk_df["Risk_Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
            fig_l = px.bar(
                x=level_counts.index,
                y=level_counts.values,
                title="Risk level breakdown",
                labels={"x": "Risk level", "y": "Students"},
                color=level_counts.index,
                color_discrete_map={"High": "#B91C1C", "Medium": "#D97706", "Low": "#64748B"},
            )
            fig_l.update_layout(showlegend=False)
            show_chart(fig_l, use_container_width=True)
        with c2:
            if "Branch" in at_risk_df.columns and at_risk_df["Branch"].nunique() > 1:
                by_b = at_risk_df.groupby("Branch").size().reset_index(name="At_Risk")
                fig_b = px.bar(by_b, x="Branch", y="At_Risk", title="At-risk by branch", color="Branch")
                fig_b.update_layout(showlegend=False)
                show_chart(fig_b, use_container_width=True)
            elif "SGPA" in at_risk_df.columns:
                fig_s = px.histogram(at_risk_df, x="SGPA", nbins=12, title="At-risk SGPA distribution")
                show_chart(fig_s, use_container_width=True)

        st.markdown("**How this page works**")
        st.markdown(
            "- Choose **Batch**, **Academic year**, and **Semester** on this page for a clear cohort\n"
            "- When Semester / Academic year is **All**, each student is checked on their **latest** result only\n"
            "- Sidebar rules control SGPA / fail / credit thresholds"
        )

    if inner == "Mentoring List":
        st.subheader("Mentoring list")
        show_cols = [
            c for c in [
                "Branch", "Roll_No", "Name", "Batch_Year_Label", "Study_Year_Label", "Semester_Label",
                "Gender", "SGPA", "Credits_Earned", "Fail_Count", "Risk_Level", "Risk_Reasons", "Dataset",
            ]
            if c in at_risk_df.columns
        ]
        level_order = {"High": 0, "Medium": 1, "Low": 2}
        sorted_risk = at_risk_df.assign(_ord=at_risk_df["Risk_Level"].map(level_order)).sort_values(
            ["_ord", "Fail_Count", "SGPA"], ascending=[True, False, True]
        ).drop(columns=["_ord"])

        st.dataframe(sorted_risk[show_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download at-risk list (CSV)",
            data=sorted_risk[show_cols].to_csv(index=False),
            file_name="at_risk_students.csv",
            mime="text/csv",
            use_container_width=True,
        )


