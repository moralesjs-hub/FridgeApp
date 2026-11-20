"""Shopping List view: manage, check off, clear, and export needed ingredients."""

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
            <h2>🛒 Shopping List</h2>
            <p class="page-subtitle">Manage and export the missing ingredients needed for your cooking sessions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    items: list[str] = st.session_state["shopping_list"]

    if items:
        # ── Item count badge ───────────────────────────────────────────────────
        st.markdown(
            f'<p style="color:#1B4D3E;font-weight:600;margin-bottom:0.75rem;">'
            f'📦 {len(items)} item{"s" if len(items) != 1 else ""} in your list</p>',
            unsafe_allow_html=True,
        )

        # ── Item rows ──────────────────────────────────────────────────────────
        st.markdown('<div class="shopping-list-container">', unsafe_allow_html=True)
        for item in items[:]:
            c_chk, c_del = st.columns([5, 1])
            with c_chk:
                st.checkbox(item.capitalize(), key=f"chk_{item}")
            with c_del:
                if st.button("✕", key=f"del_{item}", type="tertiary", help=f"Remove {item}"):
                    st.session_state["shopping_list"].remove(item)
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Action buttons ─────────────────────────────────────────────────────
        col_exp, col_clr = st.columns(2)
        with col_exp:
            st.download_button(
                "⬇️ Export List (TXT)",
                data="\n".join(items),
                file_name="shopping_list.txt",
                mime="text/plain",
                width="stretch",
            )
        with col_clr:
            if st.button("🗑️ Clear Shopping List", type="secondary", width="stretch"):
                st.session_state["shopping_list"].clear()
                st.rerun()

    else:
        # ── Empty state ────────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="empty-state-card">
                <div class="empty-state-icon">🛒</div>
                <h3>Your shopping list is empty</h3>
                <p>Missing ingredients are added here automatically when you click
                   <strong>Add to Shopping List</strong> on a recipe card.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__" or "st" in globals():
    show()
