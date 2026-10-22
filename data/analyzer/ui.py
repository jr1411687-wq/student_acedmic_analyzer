"""Presentation helpers. Numbers are computed by the pages before they reach here."""

import json
import re

import streamlit as st

_NUMBER = re.compile(r"^(-?\d[\d,]*\.?\d*)(%)?(\s*\(.*\))?$")


def _kpi_spec(label, value):
    text = "" if value is None else str(value).strip()
    spec = {"label": str(label), "text": text, "animate": False}
    match = _NUMBER.match(text)
    if not match:
        return spec
    raw = match.group(1).replace(",", "")
    spec.update({
        "animate": True,
        "num": float(raw),
        "decimals": len(raw.split(".")[1]) if "." in raw else 0,
        "suffix": (match.group(2) or "") + (match.group(3) or ""),
        "grouped": "," in match.group(1),
    })
    return spec


def kpi_cards(items):
    """Count numbers up when a page opens or a figure changes."""
    fresh = bool(st.session_state.get("_kpi_fresh"))
    screen = st.session_state.get("_screen_token", "")
    specs = []
    for index, (label, value) in enumerate(items):
        spec = _kpi_spec(label, value)
        seen_key = "kpi_" + re.sub(r"\W+", "_", f"{screen}_{spec['label']}")[:64]
        changed = st.session_state.get(seen_key) != spec["text"]
        st.session_state[seen_key] = spec["text"]
        play = fresh or changed
        if not play:
            spec["animate"] = False
        spec["enter"] = bool(play and (spec["animate"] or fresh))
        spec["delay"] = index * 45 if spec["enter"] else 0
        specs.append(spec)
    cols = st.columns(len(specs) or 1)
    for col, spec in zip(cols, specs):
        with col:
            st.iframe(_kpi_html(spec), height=112)


def _kpi_html(spec):
    motion = ""
    if spec.get("enter"):
        motion = f"animation: rise 0.28s ease both; animation-delay: {spec.get('delay', 0)}ms;"
    payload = json.dumps(spec).replace("<", "\\u003c")
    return f"""
    <style>
      html, body {{ margin: 0; padding: 0; background: transparent; }}
      .card {{
        background: #ffffff;
        border: 1px solid #E7E0D6;
        border-top: 3px solid #D01010;
        border-radius: 12px;
        padding: 12px 14px 10px;
        box-shadow: 0 8px 20px rgba(32, 16, 96, 0.05);
        font-family: sans-serif;
        min-height: 78px;
        {motion}
      }}
      @keyframes rise {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: none; }}
      }}
      .card:hover {{ transform: translateY(-2px); box-shadow: 0 12px 24px rgba(32, 16, 96, 0.08); }}
      .label {{ margin: 0; color: #5C4A42; font-size: 13px; font-weight: 650; }}
      .value {{ margin: 6px 0 0; color: #201060; font-size: 28px; font-weight: 700; letter-spacing: -0.03em; }}
      @media (prefers-reduced-motion: reduce) {{
        .card {{ animation: none; }}
        .card:hover {{ transform: none; }}
      }}
    </style>
    <article class="card" id="card"></article>
    <script>
      const spec = {payload};
      const card = document.getElementById("card");
      card.setAttribute("aria-label", spec.label + ": " + spec.text);
      const label = document.createElement("p");
      label.className = "label";
      label.textContent = spec.label;
      const value = document.createElement("p");
      value.className = "value";
      value.setAttribute("aria-hidden", "true");
      card.append(label, value);
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      function format(num) {{
        const digits = spec.decimals || 0;
        const shown = spec.grouped
          ? num.toLocaleString(undefined, {{ minimumFractionDigits: digits, maximumFractionDigits: digits }})
          : num.toFixed(digits);
        return shown + (spec.suffix || "");
      }}
      if (!spec.animate || reduce) {{
        value.textContent = spec.text;
      }} else {{
        const start = performance.now();
        const tick = (now) => {{
          const t = Math.min(1, (now - start) / 650);
          const eased = 1 - Math.pow(1 - t, 3);
          value.textContent = format(spec.num * eased);
          if (t < 1) requestAnimationFrame(tick);
          else value.textContent = spec.text;
        }};
        requestAnimationFrame(tick);
      }}
    </script>
    """


def show_chart(fig, where=None, **kwargs):
    """Apply the shared chart frame, then draw the figure the page already built."""
    fig.update_layout(
        font=dict(family="sans-serif", color="#1C1430"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#201060", font=dict(color="#ffffff"), bordercolor="#201060"),
    )
    if kwargs.pop("use_container_width", True):
        kwargs["width"] = "stretch"
    else:
        kwargs.setdefault("width", "content")
    (where or st).plotly_chart(fig, **kwargs)


def mark_transition(token):
    """Fade the screen only when the open page changes, not when a filter changes."""
    fresh = st.session_state.get("_screen_token") != token
    st.session_state["_screen_token"] = token
    st.session_state["_kpi_fresh"] = fresh
    if not fresh:
        return
    st.markdown(
        """
        <style>
        @keyframes gndec-enter {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: none; }
        }
        [data-testid="stMainBlockContainer"] {
          animation: gndec-enter 0.28s ease;
        }
        @media (prefers-reduced-motion: reduce) {
          [data-testid="stMainBlockContainer"] { animation: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
