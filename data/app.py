import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st

st.set_page_config(
    page_title="Student Academic Analyzer",
    page_icon="🎓",
    layout="wide",
)

from analyzer.look import apply_look
from analyzer.nav import FEATURE_VIEWS, render_page_nav, render_view_nav, selected_page
from analyzer.ui import mark_transition
from analyzer.pages.home import render_home_page
from analyzer.pages.at_risk import render_at_risk_page
from analyzer.pages.branch import render_branch_analysis_page
from analyzer.pages.cgpa import render_cgpa_page
from analyzer.pages.college import render_college_comparison_page
from analyzer.pages.heatmap import render_subject_heatmap_page
from analyzer.pages.sections import render_section_comparison_page

apply_look()
page = selected_page()
render_page_nav(page)
view_name = None
if page in FEATURE_VIEWS:
    view_name = render_view_nav(page, FEATURE_VIEWS[page])
mark_transition(f"{page}|{view_name or ''}")

if page == "Home":
    render_home_page()
elif page == "Overall CGPA":
    render_cgpa_page()
elif page == "At-Risk Students":
    render_at_risk_page()
elif page == "Subject Heatmap":
    render_subject_heatmap_page()
elif page == "Section Comparison":
    render_section_comparison_page()
elif page == "College Comparison":
    render_college_comparison_page()
else:
    render_branch_analysis_page()

if st.session_state.pop("_pending_view_rerun", False):
    st.rerun()
