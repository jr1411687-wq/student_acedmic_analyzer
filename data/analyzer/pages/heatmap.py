import streamlit as st
import plotly.express as px

from analyzer.analytics import _clean_subject_label, _subject_cols_for_frame, build_subject_fail_matrix
from analyzer.catalog import _sorted_sem_labels, _sorted_study_year_labels, apply_year_sem_filter
from analyzer.nav import FEATURE_VIEWS, selected_view
from analyzer.store import _data_mtime_signature, load_all_branch_datasets
from analyzer.ui import kpi_cards, show_chart

def render_subject_heatmap_page():
    """Subject weakness heatmap across branches."""
    inner = selected_view(FEATURE_VIEWS["Subject Heatmap"])
    st.title(inner)
    st.caption("Subject Heatmap · Fail / absent rate by subject — only subjects that actually appear in each branch")

    sig = _data_mtime_signature()
    combined, load_errors = load_all_branch_datasets(sig)
    if load_errors:
        with st.expander("Dataset load notes", expanded=False):
            for err in load_errors:
                st.write(f"- {err}")
    if combined.empty:
        st.warning("Add CSV/PDF files to the data folder first.")
        return

    mode = st.radio(
        "View",
        ["Single branch", "College overall"],
        horizontal=True,
        key="heatmap_mode",
    )

    # On-page scope filters (clearer than sidebar)
    f1, f2, f3 = st.columns(3)
    with f1:
        study_opts = ["All"] + _sorted_study_year_labels(
            combined["Study_Year_Label"].dropna().astype(str).tolist()
            if "Study_Year_Label" in combined.columns else []
        )
        study_pick = st.selectbox("Academic year", study_opts, key="heatmap_study")
    with f2:
        scoped_for_sem = combined
        if study_pick != "All" and "Study_Year_Label" in combined.columns:
            scoped_for_sem = combined[combined["Study_Year_Label"].astype(str) == study_pick]
        sem_opts = ["All"] + _sorted_sem_labels(
            scoped_for_sem["Semester_Label"].dropna().astype(str).tolist()
            if "Semester_Label" in scoped_for_sem.columns else []
        )
        sem_pick = st.selectbox("Semester", sem_opts, key="heatmap_sem")
    with f3:
        if mode == "Single branch" and "Branch" in combined.columns:
            branch_pick = st.selectbox(
                "Branch",
                sorted(combined["Branch"].unique()),
                key="heatmap_branch_pick",
            )
        else:
            branch_pick = None
            st.selectbox("Branch", ["All branches"], disabled=True, key="heatmap_branch_disabled")

    scoped = apply_year_sem_filter(
        combined,
        {
            "batch_year_label": "All",
            "study_year_label": study_pick,
            "exam_year_label": "All",
            "semester_label": sem_pick,
        },
    )
    if mode == "Single branch" and branch_pick is not None and "Branch" in scoped.columns:
        scoped = scoped[scoped["Branch"] == branch_pick]

    if scoped.empty:
        st.warning("No data for this Academic year / Semester selection. Try All.")
        return

    if mode == "Single branch":
        fail_matrix = build_subject_fail_matrix(scoped, by_branch=True)
    else:
        fail_matrix = build_subject_fail_matrix(scoped, by_branch=False)

    if fail_matrix.empty:
        st.info("No subject grade columns with enough data were detected for this selection.")
        return

    scope_label = []
    if study_pick != "All":
        scope_label.append(study_pick)
    if sem_pick != "All":
        scope_label.append(f"Sem {sem_pick}")
    if mode == "Single branch" and branch_pick:
        scope_label.append(branch_pick)
    else:
        scope_label.append("College overall")
    st.caption("Showing: " + " · ".join(scope_label))

    # Detected subjects (for transparency)
    with st.expander("Subjects detected", expanded=False):
        if mode == "Single branch" and "Branch" in scoped.columns:
            for b, g in scoped.groupby("Branch"):
                subs = [_clean_subject_label(c) for c in _subject_cols_for_frame(g)]
                st.markdown(f"**{b}** ({len(subs)}): " + (", ".join(subs) if subs else "none"))
        else:
            subs = [_clean_subject_label(c) for c in _subject_cols_for_frame(scoped)]
            st.write(", ".join(subs) if subs else "none")

    flat = fail_matrix.sort_values("Fail_Rate_%", ascending=False)
    kpi_cards([
        ("Subject entries", len(fail_matrix)),
        ("Unique subjects", fail_matrix["Subject"].nunique()),
        (
            "Highest fail rate",
            f"{flat.iloc[0]['Subject']} · {flat.iloc[0]['Fail_Rate_%']}%" if not flat.empty else "—",
        ),
    ])

    if inner == "Fail Rates":
        chart_df = fail_matrix.sort_values("Fail_Rate_%", ascending=True)
        by_branch = mode == "College overall" and "Branch" in chart_df.columns
        fig = px.bar(
            chart_df,
            x="Fail_Rate_%",
            y="Subject",
            color="Branch" if by_branch else "Fail_Rate_%",
            color_continuous_scale=None if by_branch else ["#F3E6E6", "#9E1010"],
            orientation="h",
            text="Fail_Rate_%",
            hover_data=[c for c in ["Fail_Count", "Graded_Students"] if c in chart_df.columns],
            title="Subject fail / absent rates",
        )
        fig.update_layout(height=max(400, 28 * max(len(chart_df), 8)))
        show_chart(fig, use_container_width=True)
        st.markdown("**How to read this**")
        st.markdown(
            "- Pick **Academic year** (1st/2nd/…) and **Semester**, then **Single branch** or **College overall**\n"
            "- Rate = (F / F(Ext.) / Ab / E / …) ÷ students who have a grade in that subject\n"
            "- Each branch only shows its own subjects (CSE ≠ Civil codes)"
        )

    if inner == "Subject Table":
        st.subheader("Detailed subject table")
        detail_cols = [
            c for c in ["Branch", "Subject", "Fail_Count", "Graded_Students", "Fail_Rate_%"]
            if c in fail_matrix.columns
        ]
        detail = fail_matrix.sort_values(["Fail_Rate_%", "Fail_Count"], ascending=[False, False])
        st.dataframe(detail[detail_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download subject fail matrix (CSV)",
            data=detail[detail_cols].to_csv(index=False),
            file_name="subject_fail_heatmap.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# COLLEGE COMPARISON PAGE
# =========================================================

