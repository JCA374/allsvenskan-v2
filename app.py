import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.config import RESULTS_PATH, MODEL_PATH, SIM_PATH
from core.ui.helpers import STEPS, nav, step_done

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Allsvenskan Forecast",
    page_icon="\u26bd",
    layout="wide",
    initial_sidebar_state="auto",
)

# Responsive CSS: stack Streamlit columns on narrow screens
st.markdown("""
<style>
@media (max-width: 640px) {
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* Prevent wide tables / charts from overflowing */
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
        max-width: 100vw;
        overflow-x: auto;
    }
}
</style>
""", unsafe_allow_html=True)

# Prevent iOS keyboard from appearing when tapping a selectbox.
components.html("""
<script>
(function () {
    function patch() {
        window.parent.document
            .querySelectorAll('[data-baseweb="select"] input')
            .forEach(function (el) {
                el.setAttribute('inputmode', 'none');
            });
    }
    patch();
    new MutationObserver(patch).observe(
        window.parent.document.body,
        { childList: true, subtree: true }
    );
})();
</script>
""", height=0)


# ── Session-state defaults ────────────────────────────────────────────────────
for key, default in {
    "data_loaded":    False,
    "model_trained":  False,
    "sim_complete":   False,
    "active_page":    "Forecast",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Detect on-disk state so page refreshes don't lose context
if not st.session_state.data_loaded and RESULTS_PATH.exists():
    try:
        if len(pd.read_csv(RESULTS_PATH)) > 0:
            st.session_state.data_loaded = True
    except Exception:
        pass

if not st.session_state.model_trained and MODEL_PATH.exists():
    st.session_state.model_trained = True

if not st.session_state.sim_complete and SIM_PATH.exists():
    try:
        if len(pd.read_csv(SIM_PATH)) > 0:
            st.session_state.sim_complete = True
    except Exception:
        pass

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("\u26bd Allsvenskan")
    st.caption("Monte Carlo Forecast")
    st.divider()

    for i, (name, icon, gate) in enumerate(STEPS, 1):
        done   = step_done(gate)
        active = st.session_state.active_page == name
        if active:
            label = f"**{icon} {name}**"
        elif done:
            label = f"{icon} {name} \u2705"
        else:
            label = f"{icon} {name}"
        st.button(
            label,
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if active else "secondary",
            on_click=nav,
            args=(name,),
        )

    st.divider()
    _upd_active = st.session_state.active_page == "Update"
    st.button(
        "**\U0001f504 Update Everything**" if _upd_active else "\U0001f504 Update Everything",
        key="nav_Update",
        use_container_width=True,
        type="primary" if _upd_active else "secondary",
        on_click=nav,
        args=("Update",),
    )

page = st.session_state.active_page

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Data":
    from core.ui.pages.data import render
    render()
elif page == "Model":
    from core.ui.pages.model import render
    render()
elif page == "Simulate":
    from core.ui.pages.simulate import render
    render()
elif page == "Forecast":
    from core.ui.pages.forecast import render
    render()
elif page == "Predictions":
    from core.ui.pages.predictions import render
    render()
elif page == "Update":
    from core.ui.pages.update import render
    render()
