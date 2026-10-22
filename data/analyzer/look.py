"""Window colors and spacing. Feature calculations stay in the other modules."""

from pathlib import Path

import streamlit as st

_CSS = Path(__file__).with_name("look.css")


def apply_look():
    st.markdown(f"<style>{_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
