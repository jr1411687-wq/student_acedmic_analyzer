"""HTML page menu. Analysis pages stay in Streamlit."""

import base64
from urllib.parse import quote

import streamlit as st

from analyzer.constants import DATA_DIR

PAGES = [
    "Home",
    "Executive Dashboard",
    "Overall CGPA",
    "At-Risk Students",
    "Subject Heatmap",
    "Section Comparison",
    "Branch Analysis",
    "College Comparison",
]

LOGO_PATH = DATA_DIR / "data_assets" / "Guru_Nanak_Dev_Engineering_College,_Ludhiana_logo.png"

BRANCH_VIEWS = [
    "Gender Ratio",
    "SGPA Distribution",
    "Subject-wise Grades",
    "Top Performers",
    "Academic Status",
    "Student Results",
    "Pass / Fail",
]

FEATURE_VIEWS = {
    "Executive Dashboard": [
        "Branch Comparison",
        "Semester Comparison",
        "Subject Difficulty",
        "Reports",
    ],
    "Overall CGPA": [
        "Combined Overview",
        "Rank List",
        "Student Working",
        "By Branch",
    ],
    "At-Risk Students": [
        "Risk Overview",
        "Mentoring List",
    ],
    "Subject Heatmap": [
        "Fail Rates",
        "Subject Table",
    ],
    "Section Comparison": [
        "SGPA by Section",
        "By Semester",
        "Pass Rate",
        "Subject Pressure",
        "Students",
    ],
    "Branch Analysis": BRANCH_VIEWS,
    "College Comparison": [
        "SGPA Comparison",
        "Gender Ratio",
        "Top Performers",
        "Overview Table",
    ],
}


def page_href(name):
    return f"?page={quote(name)}"


def selected_page():
    """Read the page from the address bar. The HTML links set that value."""
    requested = st.query_params.get("page")
    if requested in PAGES:
        return requested
    st.query_params["page"] = PAGES[0]
    return PAGES[0]


@st.cache_data(show_spinner=False)
def _logo_src(path_str, mtime):
    raw = base64.b64encode(open(path_str, "rb").read()).decode()
    return f"data:image/png;base64,{raw}"


def render_page_nav(active):
    """College bar used to switch pages. Filters stay on each analysis page."""
    logo = ""
    if LOGO_PATH.exists():
        logo = f'<img src="{_logo_src(str(LOGO_PATH), LOGO_PATH.stat().st_mtime)}" alt="GNDEC logo">'
    links = "".join(
        f'<a href="{page_href(name)}" target="_self" class="{"on" if name == active else ""}">{name}</a>'
        for name in PAGES
    )
    st.markdown(
        f"""
        <style>
        .gndec-nav {{
            display: flex;
            align-items: center;
            gap: 18px;
            flex-wrap: wrap;
            background: #ffffff;
            border-bottom: 1px solid #eee6dc;
            padding: 8px 4px 10px 4px;
            margin: -1rem 0 1rem 0;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .gndec-brand {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-right: auto;
            text-decoration: none;
        }}
        .gndec-brand img {{
            height: 46px;
            width: 46px;
            object-fit: contain;
        }}
        .gndec-brand strong {{
            display: block;
            color: #201060;
            font-size: 0.95rem;
            line-height: 1.1;
        }}
        .gndec-brand small {{
            color: #9E1010;
            letter-spacing: 0.12em;
            font-size: 0.68rem;
            font-weight: 700;
        }}
        .gndec-nav a {{
            color: #201060;
            text-decoration: none;
            font-size: 0.82rem;
            font-weight: 700;
            padding-bottom: 4px;
            border-bottom: 2px solid transparent;
            transition: color 0.15s ease, border-color 0.15s ease;
        }}
        .gndec-nav a:hover {{
            color: #D01010;
        }}
        .gndec-nav a.on {{
            color: #D01010;
            border-bottom-color: #D01010;
        }}
        .gndec-nav a:focus-visible {{
            outline: 2px solid #201060;
            outline-offset: 3px;
        }}
        @media (prefers-reduced-motion: reduce) {{
            .gndec-nav a {{ transition: none; }}
        }}
        </style>
        <div class="gndec-nav">
            <a class="gndec-brand" href="{page_href(PAGES[0])}" target="_self">
                {logo}
                <span><small>GNDEC</small><strong>Ludhiana</strong></span>
            </a>
            {links}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _view_state_key(views):
    return "inner_view_" + "|".join(views)


def selected_view(views):
    """Inner page for this feature. Kept in the session so filters are not cleared."""
    key = _view_state_key(views)
    if key not in st.session_state or st.session_state[key] not in views:
        requested = st.query_params.get("view")
        if isinstance(requested, list):
            requested = requested[0] if requested else None
        st.session_state[key] = requested if requested in views else views[0]
    return st.session_state[key]


def _view_button_key(page, name):
    slug = "".join(ch if ch.isalnum() else "_" for ch in f"{page}_{name}")
    return f"gndecview_{slug}"


def render_view_nav(page, views):
    """Pages inside one feature. Clicks stay in the same session so filters remain."""
    active = selected_view(views)
    st.markdown(
        """
        <style>
        .gndec-views-label {
            color: #9E1010;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .st-key-gndec_view_bar {
            border-bottom: 1px solid #eee6dc;
            padding: 0 0 8px 0;
            margin: -8px 0 12px 0;
        }
        .st-key-gndec_view_bar button {
            background: transparent !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
            border-radius: 0 !important;
            color: #201060 !important;
            font-weight: 700 !important;
            box-shadow: none !important;
            padding: 6px 2px 4px 2px !important;
        }
        .st-key-gndec_view_bar button[data-testid="stBaseButton-primary"] {
            color: #D01010 !important;
            border-bottom-color: #D01010 !important;
        }
        .st-key-gndec_view_bar button:hover {
            color: #D01010 !important;
            border-bottom-color: #D01010 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(horizontal=True, vertical_alignment="center", gap="medium", key="gndec_view_bar"):
        st.markdown(f'<span class="gndec-views-label">{page}</span>', unsafe_allow_html=True)
        for name in views:
            if st.button(
                name,
                key=_view_button_key(page, name),
                type="primary" if name == active else "secondary",
            ) and name != active:
                st.session_state[_view_state_key(views)] = name
                st.session_state["_pending_view_rerun"] = True
    return st.session_state[_view_state_key(views)]
