"""HTML landing page. Cards open the existing analysis screens."""

import streamlit as st

from analyzer.constants import DATA_DIR
from analyzer.nav import LOGO_PATH, _logo_src, page_href

CAMPUS_PATH = DATA_DIR / "data_assets" / "WhatsApp Image 2026-10-21 at 6.28.58 PM.jpeg"

TOOLS = [
    ("Executive Dashboard", "College totals, pass and fail rates, branch and semester charts, and a PDF report."),
    ("Overall CGPA", "Credit-weighted CGPA from every semester, including an earlier branch."),
    ("At-Risk Students", "Students who need mentoring from SGPA, fails, and credits."),
    ("Subject Heatmap", "Which subjects are pulling a branch down."),
    ("Section Comparison", "Section A and B compared from the official section list."),
    ("Branch Analysis", "One branch, with Regular, Reappear, and Makeup together."),
    ("College Comparison", "How branches stand against each other."),
]


def render_home_page():
    cards = "".join(
        f'''<a class="home-card" href="{page_href(title)}" target="_self">
            <span>ANALYSIS</span>
            <h3>{title}</h3>
            <p>{blurb}</p>
            <em>Open tool →</em>
        </a>'''
        for title, blurb in TOOLS
    )
    logo = ""
    if LOGO_PATH.exists():
        logo = f'<img src="{_logo_src(str(LOGO_PATH), LOGO_PATH.stat().st_mtime)}" alt="Guru Nanak Dev Engineering College logo">'
    st.markdown(
        f"""
        <style>
        .gndec-home {{ color: #1A1230; margin-top: 8px; }}
        .home-badge {{
            display: inline-block;
            background: #FDECEC;
            color: #9E1010;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }}
        .gndec-home h1 {{
            color: #201060;
            font-size: clamp(1.8rem, 4vw, 2.6rem);
            line-height: 1.1;
            margin: 12px 0 8px 0;
        }}
        .gndec-home .lead {{ color: #5C4A42; margin: 0 0 18px 0; font-size: 1.05rem; }}
        .home-tools {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}
        .home-brand {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin: 8px 0 12px;
        }}
        .home-brand img {{
            width: 72px;
            height: 72px;
            object-fit: contain;
        }}
        .home-card {{
            display: block;
            text-decoration: none;
            background: #fff;
            border: 1px solid #EEE6DC;
            border-radius: 16px;
            padding: 18px 18px 14px;
            box-shadow: 0 8px 24px rgba(32,16,96,0.04);
        }}
        .home-card span {{
            color: #9E1010;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }}
        .home-card h3 {{ color: #201060; margin: 8px 0 6px; font-size: 1.15rem; }}
        .home-card p {{ color: #5C4A42; margin: 0 0 12px; font-size: 0.92rem; }}
        .home-card em {{ color: #D01010; font-style: normal; font-weight: 700; font-size: 0.88rem; }}
        @media (max-width: 1100px) {{
            .home-tools {{ grid-template-columns: 1fr 1fr; }}
        }}
        @media (max-width: 700px) {{
            .home-tools {{ grid-template-columns: 1fr; }}
            .home-brand {{ align-items: flex-start; }}
        }}
        </style>
        <div class="gndec-home">
          <div class="home-brand">
            {logo}
            <div>
              <div class="home-badge">ESTABLISHED 1956 · PUNJAB GOVERNMENT AIDED</div>
              <h1>Guru Nanak Dev Engineering College</h1>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if CAMPUS_PATH.exists():
        with st.container(key="campus_section"):
            st.image(str(CAMPUS_PATH), width="stretch")
    st.markdown(
        f"""
        <div class="gndec-home">
          <p class="lead">Student Academic Analyzer reads the college result sheets already in this project and turns them into pass rates, subject difficulty, and branch comparisons. No student list is shown on this page.</p>
          <div class="home-tools">{cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
