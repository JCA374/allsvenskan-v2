"""Step 5 -- Predictions page."""
import pandas as pd
import streamlit as st

from core.config import MODEL_PATH, FORECAST_CACHE_PATH
from core.ui.helpers import (
    stepper, blocked, load_fixtures, load_model, load_forecast_cache,
)


def render():
    stepper()
    st.title("Step 5 \u2014 Predictions")

    # Show updated-at from forecast cache
    try:
        if FORECAST_CACHE_PATH.exists():
            _fc_data = load_forecast_cache()
            _pred_updated_at = _fc_data[-1]
            if _pred_updated_at:
                st.caption(f"Updated at: {_pred_updated_at.strftime('%Y-%m-%d %H:%M')}")
    except Exception:
        pass

    if not st.session_state.model_trained:
        blocked("Model", "Train the model first to generate fixture predictions.")

    fixtures = load_fixtures()
    if fixtures.empty:
        st.info("No upcoming fixtures found. Download data for the current season on the Data page.")
        st.stop()

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    all_teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))

    # ── Filters ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        team1 = fc1.selectbox("Team 1 (highlight)", ["All teams"] + all_teams, key="pred_team1")
    with fc2:
        team2_opts = ["All teams"] + [t for t in all_teams if t != team1] if team1 != "All teams" else ["All teams"] + all_teams
        team2 = fc2.selectbox("Team 2", team2_opts, key="pred_team2")
    with fc3:
        if "Date" in fixtures.columns and fixtures["Date"].notna().any():
            dates       = sorted(fixtures["Date"].dt.date.dropna().unique())
            date_filter = fc3.selectbox("Date", ["All dates"] + [str(d) for d in dates], key="pred_date")
        else:
            date_filter = "All dates"

    # ── Build predictions ────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _predict_fixtures(fixture_records: tuple, model_mtime: float):
        rows = []
        for home, away, date in fixture_records:
            try:
                pred = model.predict_outcome_probabilities(home, away)
                rows.append({
                    "Date":     date,
                    "Home":     home,
                    "Away":     away,
                    "Home Win": pred["home_win"],
                    "Draw":     pred["draw"],
                    "Away Win": pred["away_win"],
                    "xG Home":  pred["mu_home"],
                    "xG Away":  pred["mu_away"],
                })
            except Exception:
                pass
        return rows

    _model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
    _all_records = tuple(
        (r["HomeTeam"], r["AwayTeam"], r["Date"].date() if pd.notna(r.get("Date")) else "\u2014")
        for _, r in fixtures.iterrows()
    )
    all_rows = _predict_fixtures(_all_records, _model_mtime)

    if not all_rows:
        st.info("No predictions could be generated.")
        st.stop()

    pred_df = pd.DataFrame(all_rows)

    # Apply filters
    disp_df = pred_df.copy()
    if team1 != "All teams" and team2 != "All teams":
        disp_df = disp_df[
            ((disp_df["Home"] == team1) & (disp_df["Away"] == team2)) |
            ((disp_df["Home"] == team2) & (disp_df["Away"] == team1))
        ]
    elif team1 != "All teams":
        disp_df = disp_df[(disp_df["Home"] == team1) | (disp_df["Away"] == team1)]
    elif team2 != "All teams":
        disp_df = disp_df[(disp_df["Home"] == team2) | (disp_df["Away"] == team2)]
    if date_filter != "All dates":
        disp_df = disp_df[disp_df["Date"].astype(str) == date_filter]

    st.caption(f"Showing {len(disp_df)} fixture{'s' if len(disp_df) != 1 else ''}")

    if disp_df.empty:
        st.info("No fixtures match the selected filters.")
        st.stop()

    # ── Match cards ──────────────────────────────────────────────────────────
    st.subheader("Match Details")
    cards_html = ["""<style>
.match-card {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #e0e0e0;
}
.match-team { flex: 1 1 30%; min-width: 80px; }
.match-team-away { text-align: right; }
.match-center { flex: 1 1 30%; min-width: 120px; text-align: center; }
@media (max-width: 480px) {
    .match-card { flex-direction: column; align-items: stretch; }
    .match-team, .match-team-away, .match-center { text-align: left; min-width: unset; }
}
</style>"""]
    for r in disp_df.to_dict("records"):
        hw   = float(r["Home Win"]) * 100
        dw   = float(r["Draw"])     * 100
        aw   = float(r["Away Win"]) * 100
        xgh  = float(r["xG Home"])
        xga  = float(r["xG Away"])
        t1_home = team1 != "All teams" and r["Home"] == team1
        t1_away = team1 != "All teams" and r["Away"] == team1
        hw_str = f"<span style='font-weight:900;color:#111'>{hw:.0f}%</span>" if t1_home else f"{hw:.0f}%"
        aw_str = f"<span style='font-weight:900;color:#111'>{aw:.0f}%</span>" if t1_away else f"{aw:.0f}%"
        dw_str = f"{dw:.0f}%"
        cards_html.append(f"""
<div class="match-card">
  <div class="match-team">
    <strong>{r['Home']}</strong><br>
    <span style="font-size:0.8em">xG {xgh:.2f}</span>
  </div>
  <div class="match-center">
    <div style="color:#888;font-size:0.75em;margin-bottom:4px">{r['Date']}</div>
    <div style="font-size:1em;margin-bottom:6px">{hw_str} \u00b7 {dw_str} \u00b7 {aw_str}</div>
    <div style="display:flex;height:10px;border-radius:4px;overflow:hidden">
      <div style="width:{hw:.1f}%;background:#2196F3"></div>
      <div style="width:{dw:.1f}%;background:#9E9E9E"></div>
      <div style="width:{aw:.1f}%;background:#F44336"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.7em;color:#888;margin-top:2px">
      <span>Home</span><span>Draw</span><span>Away</span>
    </div>
  </div>
  <div class="match-team match-team-away">
    <strong>{r['Away']}</strong><br>
    <span style="font-size:0.8em">xG {xga:.2f}</span>
  </div>
</div>""")
    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

    st.divider()

    # ── Table view ───────────────────────────────────────────────────────────
    st.subheader("All Fixtures")

    def _bold_team1_odds(row):
        styles = [""] * len(row)
        cols   = list(row.index)
        if team1 != "All teams":
            if row.get("Home") == team1 and "Home Win" in cols:
                styles[cols.index("Home Win")] = "font-weight: 900; color: #111"
            elif row.get("Away") == team1 and "Away Win" in cols:
                styles[cols.index("Away Win")] = "font-weight: 900; color: #111"
        return styles

    table_df = disp_df.copy()
    table_df["Home Win"] = table_df["Home Win"].map(lambda x: f"{x:.0%}")
    table_df["Draw"]     = table_df["Draw"].map(lambda x: f"{x:.0%}")
    table_df["Away Win"] = table_df["Away Win"].map(lambda x: f"{x:.0%}")
    table_df["xG Home"]  = table_df["xG Home"].map(lambda x: f"{x:.2f}")
    table_df["xG Away"]  = table_df["xG Away"].map(lambda x: f"{x:.2f}")
    styled = table_df.style.apply(_bold_team1_odds, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
