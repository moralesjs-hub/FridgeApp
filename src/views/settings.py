"""Settings view: configure model confidence, matching criteria, servings, and engine info."""

from __future__ import annotations

import streamlit as st

from src.backend import apply_custom_styles, init_session_state


def show() -> None:
    apply_custom_styles()
    init_session_state()

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="page-header">
            <h2>⚙️ Engine Configuration</h2>
            <p class="page-subtitle">Tune recognition confidence, recipe matching, and serving sizes to your liking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Detection settings ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="settings-section-header">
            <span class="settings-section-icon">🔬</span>
            <div>
                <strong>Detection &amp; Recognition</strong>
                <p>Controls how the AI vision model interprets your fridge photos.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        confidence = st.slider(
            "Model confidence threshold",
            min_value=0.05,
            max_value=0.90,
            value=float(st.session_state.get("confidence", 0.15)),
            step=0.05,
            help="Filter detected ingredients based on model confidence. Lower = more detections (noisier). Higher = fewer (stricter).",
            key="cfg_confidence",
        )
        st.session_state["confidence"] = confidence

    with col_s2:
        servings = st.number_input(
            "Recipe Servings",
            1,
            12,
            int(st.session_state.get("servings", 1)),
            help="Number of servings used to scale nutrition macros on recipe cards.",
            key="cfg_servings",
        )
        st.session_state["servings"] = servings

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recipe matching settings ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="settings-section-header">
            <span class="settings-section-icon">🍳</span>
            <div>
                <strong>Recipe Matching</strong>
                <p>Adjust how many results appear and how strict the ingredient match must be.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        max_results = st.slider(
            "Maximum recipe results",
            min_value=1,
            max_value=12,
            value=int(st.session_state.get("max_results", 6)),
            help="Cap the number of recipe cards displayed in Find Recipes.",
            key="cfg_max_results",
        )
        st.session_state["max_results"] = max_results

    with col_s4:
        min_match = st.slider(
            "Minimum recipe match",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("min_match", 25)),
            step=5,
            format="%d%%",
            help="Hide recipes whose ingredient overlap falls below this threshold.",
            key="cfg_min_match",
        )
        st.session_state["min_match"] = min_match

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── About / team footer ────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="about-card">
            <div class="about-card-title">🧑‍💻 About Ingredia</div>
            <p>Built by <strong>Philip · Snovi · Jaden · Amelia</strong></p>
            <p style="color:#6B7280;font-size:0.85rem;margin-top:0.25rem;">
                CSCI 270 Applied AI &nbsp;·&nbsp; College of Charleston
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__" or "st" in globals():
    show()