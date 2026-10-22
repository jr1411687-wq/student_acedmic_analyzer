import streamlit as st
import plotly.express as px

from analyzer.analytics import MERIT_GRADE_ORDER, _cgpa_notice_example, _feature_criteria_table, build_combined_overview, compute_overall_cgpa
from analyzer.catalog import _rolls_on_class_sheet, _sorted_year_labels
from analyzer.nav import FEATURE_VIEWS, selected_view
from analyzer.sections import load_all_section_maps
from analyzer.store import _data_mtime_signature, load_all_branch_datasets
from analyzer.ui import kpi_cards, show_chart

def render_cgpa_page():
    """Overall CGPA using the college academic-section formula."""
    inner = selected_view(FEATURE_VIEWS["Overall CGPA"])
    st.title(inner)
    st.caption(
        "Overall CGPA · Credit-weighted CGPA from every loaded semester result. "
        "Same formula as the academic-section notice (SGPA × semester credits)."
    )

    example = _cgpa_notice_example()
    with st.expander("Formula used (college notice)", expanded=True):
        st.latex(
            r"CGPA=\frac{S_1C_1+S_2C_2+\cdots+S_nC_n}{C_1+C_2+\cdots+C_n}"
        )
        st.markdown(
            f"""
            - **S** = SGPA printed on that semester result. If the semester has a reappear (F / Ab) or the SGPA cell is blank, **S = 0**.
            - **C** = credits printed on that same semester result, including semesters taken in an earlier branch.
            - CGPA = (S1×C1 + S2×C2 + …) / (C1 + C2 + …), rounded to **2 decimal places**. Official example: (5.67×26 + 6.28×28 + 7.14×27 + 6.58×25) / 106 = **{example:.2f}**.
            - If two students tie on CGPA, higher rank goes to more **O** grades across all semesters, then **A+**, then **A**, and so on.
            """
        )

    sig = _data_mtime_signature()
    with st.spinner("Loading all semester results…"):
        combined, load_errors = load_all_branch_datasets(sig)
    if load_errors:
        with st.expander("Dataset load notes", expanded=False):
            for err in load_errors:
                st.write(f"- {err}")
    if combined.empty:
        st.warning("Add result CSV/PDF files first.")
        return

    with st.spinner("Calculating CGPA from every semester SGPA and credits…"):
        summary, detail = compute_overall_cgpa(combined)
    if summary.empty:
        st.warning("Could not compute CGPA. Need Roll_No, SGPA, and credits in the result files.")
        return

    f1, f2, f3 = st.columns(3)
    branches = ["All"] + sorted(summary["Branch"].dropna().astype(str).unique())
    branch_pick = f1.selectbox("Branch", branches, key="cgpa_branch")
    batches = ["All"]
    if "Batch_Year_Label" in combined.columns:
        batches += _sorted_year_labels(combined["Batch_Year_Label"].dropna().astype(str).tolist())
    batch_pick = f2.selectbox("Batch year", batches, key="cgpa_batch")
    cleared_only = f3.checkbox("Fully cleared students only", value=False, key="cgpa_cleared")

    view = summary.copy()
    detail_view = detail.copy()
    if branch_pick != "All":
        view = view[view["Branch"].astype(str) == branch_pick]
    if batch_pick != "All":
        # Any student on that class sheet, including a different roll-year prefix.
        class_rolls = _rolls_on_class_sheet(combined, batch_pick, branch_pick)
        view = view[view["Roll_No"].astype(str).isin(class_rolls)]
    if cleared_only:
        view = view[view["Fully_Cleared"]]
    # Keep every semester for these rolls, including an earlier branch.
    detail_view = detail_view[detail_view["Roll_No"].astype(str).isin(set(view["Roll_No"].astype(str)))]

    if view.empty:
        st.info("No students match these filters.")
        return

    # Re-rank inside the visible cohort so Rank 1 is the top of this screen
    sort_cols = ["CGPA"] + [f"Grade_{g}" for g in MERIT_GRADE_ORDER]
    view = view.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")
    ranks = []
    prev = None
    rank = 0
    for i, (_, row) in enumerate(view.iterrows(), start=1):
        key = tuple(row[c] for c in sort_cols)
        if key != prev:
            rank = i
            prev = key
        ranks.append(rank)
    view = view.copy()
    view["Rank"] = ranks

    kpi_cards([
        ("Students", f"{len(view):,}"),
        ("Average CGPA", f"{view['CGPA'].mean():.2f}"),
        ("Highest CGPA", f"{view['CGPA'].max():.2f}"),
        ("Fully cleared", int(view["Fully_Cleared"].sum())),
    ])

    st.caption(
        "CGPA uses only semesters present in the result files, including an earlier branch for the same roll. "
        "Each semester contributes its printed SGPA × its printed credits, including an earlier branch. "
        "A semester with a reappear or a blank SGPA contributes 0 × that semester's credits."
    )


    show_cols = [
        c for c in [
            "Rank", "Branch", "Earlier_Branch", "Roll_No", "Name", "Batch_Year_Label", "Gender",
            "CGPA", "Semesters", "Semesters_Count", "Scheme_Credits",
            "Uncleared_Semesters", "Grade_O", "Grade_A+",
        ] if c in view.columns
    ]
    rename = {
        "Batch_Year_Label": "Batch",
        "Earlier_Branch": "Earlier branch",
        "Semesters_Count": "Semesters counted",
        "Scheme_Credits": "Scheme credits",
        "Uncleared_Semesters": "Uncleared semesters",
        "Grade_O": "O grades",
        "Grade_A+": "A+ grades",
    }

    if inner == "Combined Overview":
        st.subheader("Combined overview")
        st.caption(
            "One row per student. Each column uses the same rule as that feature. "
            "Branch and batch filters on this page apply here too."
        )
        criteria = _feature_criteria_table()
        st.dataframe(criteria, use_container_width=True, hide_index=True)
        section_map, _sec_errors = load_all_section_maps(sig)
        scoped_results = combined
        if branch_pick != "All" and "Branch" in scoped_results.columns:
            scoped_results = scoped_results[scoped_results["Branch"].astype(str) == branch_pick]
        if batch_pick != "All":
            class_rolls = _rolls_on_class_sheet(scoped_results, batch_pick, branch_pick)
            scoped_results = scoped_results[scoped_results["Roll_No"].astype(str).isin(class_rolls)]
        combined_view = build_combined_overview(view, scoped_results, section_map)
        overview_cols = [
            c for c in [
                "Rank", "Name", "Roll_No", "Branch", "Earlier_Branch", "Section", "Batch_Year_Label",
                "CGPA", "Pass/Fail", "Uncleared_Semesters",
                "Fail subjects", "At-risk", "Risk level", "Risk reason",
                "Weakest subject",
            ] if c in combined_view.columns
        ]
        overview_table = combined_view[overview_cols].rename(columns={
            "Batch_Year_Label": "Batch",
            "Earlier_Branch": "Earlier branch",
            "Uncleared_Semesters": "Uncleared semesters",
        })
        st.dataframe(overview_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download combined overview (CSV)",
            data=overview_table.to_csv(index=False),
            file_name="combined_overview.csv",
            mime="text/csv",
            key="download_combined_overview",
        )

    if inner == "Rank List":
        st.dataframe(
            view[show_cols].rename(columns=rename),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download CGPA ranking (CSV)",
            data=view[show_cols].rename(columns=rename).to_csv(index=False),
            file_name="overall_cgpa_ranking.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if inner == "Student Working":
        rolls = sorted(view["Roll_No"].astype(str).unique())
        roll = st.selectbox("Roll number", rolls, key="cgpa_roll")
        student = view[view["Roll_No"].astype(str) == roll].iloc[0]
        steps = detail_view[detail_view["Roll_No"].astype(str) == roll].sort_values("Semester")
        kpi_cards([("CGPA", f"{student['CGPA']:.2f}")])
        st.caption(f"Rank {int(student['Rank'])} in this list")
        steps_show = steps[[
            c for c in [
                "Semester", "Branch", "SGPA", "SGPA_Used", "Scheme_Credits", "Credits_Earned",
                "Uncleared", "Fail_Count",
            ] if c in steps.columns
        ]].copy()
        if "SGPA_Used" in steps_show.columns and "Scheme_Credits" in steps_show.columns:
            steps_show["SGPA × Credits"] = (
                steps_show["SGPA_Used"] * steps_show["Scheme_Credits"]
            ).round(2)
        st.dataframe(
            steps_show.rename(columns={
                "SGPA_Used": "SGPA used",
                "Scheme_Credits": "Credits used (C)",
                "Credits_Earned": "Credits earned",
                "Fail_Count": "Reappear subjects",
            }),
            use_container_width=True,
            hide_index=True,
        )
        num = float((steps["SGPA_Used"] * steps["Scheme_Credits"]).sum()) if not steps.empty else 0
        den = float(steps["Scheme_Credits"].sum()) if not steps.empty else 0
        st.markdown(
            f"**CGPA** = {num:.2f} / {den:.2f} = **{student['CGPA']:.2f}** "
            f"(semesters {student['Semesters']})"
        )

    if inner == "By Branch":
        branch_avg = (
            view.groupby("Branch", dropna=False)
            .agg(Students=("Roll_No", "count"), **{"Avg CGPA": ("CGPA", "mean"), "Top CGPA": ("CGPA", "max")})
            .reset_index()
        )
        branch_avg["Avg CGPA"] = branch_avg["Avg CGPA"].round(2)
        branch_avg["Top CGPA"] = branch_avg["Top CGPA"].round(2)
        fig = px.bar(
            branch_avg.sort_values("Avg CGPA", ascending=False),
            x="Branch",
            y="Avg CGPA",
            text="Avg CGPA",
            color="Students",
            title="Average CGPA by branch",
        )
        fig.update_traces(textposition="outside")
        show_chart(fig, use_container_width=True)
        st.dataframe(branch_avg, use_container_width=True, hide_index=True)
