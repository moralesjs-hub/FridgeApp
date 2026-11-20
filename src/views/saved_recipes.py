"""Saved Recipes view: quickly access and remove pinned favorite recipes."""

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
            <h2>🔖 Saved Bookmarks</h2>
            <p class="page-subtitle">Your pinned favorite recipes, always one click away.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    favs: list[str] = st.session_state["favorites"]

    if favs:
        # ── Count badge ────────────────────────────────────────────────────────
        st.markdown(
            f'<p style="color:#1B4D3E;font-weight:600;margin-bottom:0.75rem;">'
            f'⭐ {len(favs)} saved recipe{"s" if len(favs) != 1 else ""}</p>',
            unsafe_allow_html=True,
        )

        # ── Bookmark cards ─────────────────────────────────────────────────────
        for fav in favs[:]:
            fc1, fc2 = st.columns([5, 1])
            with fc1:
                st.markdown(
                    f'<div class="bookmark-card">🍽️ &nbsp;<strong>{fav}</strong></div>',
                    unsafe_allow_html=True,
                )
            with fc2:
                if st.button("✕", key=f"fav_{fav}", type="tertiary", help="Remove bookmark"):
                    st.session_state["favorites"].remove(fav)
                    st.rerun()

    else:
        # ── Empty state ────────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="empty-state-card">
                <div class="empty-state-icon">🔖</div>
                <h3>No bookmarks yet</h3>
                <p>Click <strong>☆ Save</strong> on any recipe card in
                   <em>Find Recipes</em> or <em>Explore Catalog</em>
                   to pin it here for quick access.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__" or "st" in globals():
    show()
