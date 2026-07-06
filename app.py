
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

st.set_page_config(page_title="Aeronautics Dashboard", layout="wide")

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

DATA_PATH = "flights_aero.csv"
df = load_data(DATA_PATH)

st.markdown("## <div style='text-align: center;'>✈️ Aeronautics Operations Dashboard</div>", unsafe_allow_html=True)

st.markdown(
    """
This dashboard explores a **synthetic** aeronautics dataset.
Use the sidebar to filter by airline, aircraft type, and date range.
Below, charts are organized **two per row** for readability.
"""
)

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

airlines = ["All"] + sorted(df["airline"].dropna().unique().tolist())
selected_airline = st.sidebar.selectbox("Airline", airlines, key="airline_sel_v4")

aircrafts = ["All"] + sorted(df["aircraft_type"].dropna().unique().tolist())
selected_ac = st.sidebar.selectbox("Aircraft Type", aircrafts, key="ac_sel_v4")

# Date range
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="date_range_sel_v4",
)

# Apply filters
mask = pd.Series([True] * len(df))
if selected_airline != "All":
    mask &= (df["airline"] == selected_airline)
if selected_ac != "All":
    mask &= (df["aircraft_type"] == selected_ac)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask &= (df["date"].between(start, end))

fdf = df.loc[mask].copy()

st.caption(f"Filtered rows: **{len(fdf)}** / {len(df)}")

# ---------------------------
# KPI cards
# ---------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Flights", f"{len(fdf):,}")
with col2:
    st.metric("Avg Delay (min)", f"{fdf['delay_min'].mean():.1f}")
with col3:
    st.metric("Avg Fuel Burn (kg)", f"{fdf['fuel_burn_kg'].mean():.0f}")
with col4:
    st.metric("Total CO₂ (kg)", f"{fdf['co2_kg'].sum():,.0f}")
with col5:
    st.metric("Load Factor", f"{fdf['load_factor'].mean():.2f}")

st.divider()

# ===========================
# ROW 1
# Time Series || Top Routes
# ===========================
left1, right1 = st.columns(2)

with left1:
    st.subheader("Time Series — Average Delay per Day (Matplotlib)")
    ts = fdf.groupby("date", as_index=False)["delay_min"].mean()

    fig1, ax1 = plt.subplots()
    ax1.plot(ts["date"], ts["delay_min"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Average delay (min)")
    ax1.set_title("Average Delay per Day")
    st.pyplot(fig1, clear_figure=True)

with right1:
    st.subheader("Top 10 Routes by Number of Flights (Seaborn)")
    fdf["route"] = fdf["origin"] + " → " + fdf["destination"]
    route_counts = (
        fdf["route"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    route_counts.columns = ["route", "count"]

    fig2, ax2 = plt.subplots()
    sns.barplot(data=route_counts, x="count", y="route", ax=ax2)
    ax2.set_xlabel("Flights")
    ax2.set_ylabel("Route")
    ax2.set_title("Top 10 Routes")
    st.pyplot(fig2, clear_figure=True)

st.divider()

# ===========================
# ROW 2
# Boxplot || Correlation Heatmap
# ===========================
left2, right2 = st.columns(2)

with left2:
    st.subheader("Delay Distribution by Airline (Seaborn — Boxplot)")
    # Boxplot of delay by airline (limit to top 10 airlines by frequency for readability)
    top_airlines = fdf["airline"].value_counts().head(10).index
    box_df = fdf[fdf["airline"].isin(top_airlines)]
    figb, axb = plt.subplots()
    sns.boxplot(data=box_df, x="delay_min", y="airline", ax=axb)
    axb.set_xlabel("Delay (min)")
    axb.set_ylabel("Airline")
    axb.set_title("Boxplot of Delay by Airline")
    st.pyplot(figb, clear_figure=True)

with right2:
    st.subheader("Correlation Heatmap (Seaborn)")
    num_cols = ["distance_km", "duration_min", "delay_min", "cruise_altitude_ft", "cruise_speed_kts", "fuel_burn_kg", "co2_kg", "load_factor", "passengers"]
    corr = fdf[num_cols].corr(numeric_only=True)

    figh, axh = plt.subplots()
    sns.heatmap(corr, annot=False, ax=axh)
    axh.set_title("Correlation Heatmap")
    st.pyplot(figh, clear_figure=True)

st.divider()

# ===========================
# ROW 3
# Flight Statistics by Airline || Aircraft Type
# ===========================
left3, right3 = st.columns(2)

with left3:
    st.subheader("Flight Statistics by Airline (Matplotlib — Bar)")
    avg_delay_airline = (
        fdf.groupby("airline", as_index=False)["delay_min"]
        .mean()
        .sort_values("delay_min", ascending=False)
    )
    fig5, ax5 = plt.subplots()
    ax5.bar(avg_delay_airline["airline"], avg_delay_airline["delay_min"])
    ax5.set_xlabel("Airline")
    ax5.set_ylabel("Average Delay (min)")
    ax5.set_title("Average Delay by Airline")
    plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig5, clear_figure=True)

with right3:
    st.subheader("Aircraft Type (Matplotlib — Pie)")
    ac_counts = fdf["aircraft_type"].value_counts().reset_index()
    ac_counts.columns = ["aircraft_type", "count"]
    fig6, ax6 = plt.subplots()
    ax6.pie(ac_counts["count"], labels=ac_counts["aircraft_type"], autopct="%1.1f%%", startangle=90)
    ax6.axis("equal")
    ax6.set_title("Share of Flights by Aircraft Type")
    st.pyplot(fig6, clear_figure=True)

st.divider()

# ===========================
# ROW 4
# Map — Last Known Positions || Another Map
# ===========================
left4, right4 = st.columns(2)

with left4:
    st.subheader("Map — Last Known Positions")
    st.caption("Quick scatter using st.map")
    st.map(fdf[["lat", "lon"]].dropna().rename(columns={"lat": "latitude", "lon": "longitude"}))

with right4:
    st.subheader("Route Flows (PyDeck — Arc Layer)")
    st.caption("Aggregated origin→destination arcs")

    # Airport coordinates dictionary (for routes)
    airport_coords = {
        "CMN": (33.3675, -7.5897),
        "RAK": (31.6069, -8.0363),
        "TNG": (35.7269, -5.9169),
        "AGA": (30.3250, -9.4131),
        "ORY": (48.7233, 2.3794),
        "MAD": (40.4722, -3.5608),
        "LHR": (51.4700, -0.4543),
        "IST": (41.2603, 28.7420),
    }

    # Build aggregated route dataset
    fdf["route"] = fdf["origin"] + "-" + fdf["destination"]
    agg = fdf.groupby(["origin", "destination"], as_index=False).size().rename(columns={"size": "count"})

    # Map to coordinates and drop routes without known coords
    def add_coords(row):
        o = airport_coords.get(row["origin"])
        d = airport_coords.get(row["destination"])
        if o and d:
            return pd.Series({"o_lat": o[0], "o_lon": o[1], "d_lat": d[0], "d_lon": d[1]})
        return pd.Series({"o_lat": np.nan, "o_lon": np.nan, "d_lat": np.nan, "d_lon": np.nan})

    coords = agg.apply(add_coords, axis=1)
    arcs = pd.concat([agg, coords], axis=1).dropna()

    if not arcs.empty:
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=arcs,
            get_source_position=["o_lon", "o_lat"],
            get_target_position=["d_lon", "d_lat"],
            get_width="count",
            get_source_color=[255, 215, 0],   # jaune doré (Golden Yellow)
            get_target_color=[255, 255, 0],   # jaune clair
            pickable=True,
        )
        # Center view between mean of origins
        center_lat = float(np.mean([airport_coords[k][0] for k in airport_coords]))
        center_lon = float(np.mean([airport_coords[k][1] for k in airport_coords]))
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=2.8, pitch=0)
        deck = pdk.Deck(layers=[arc_layer], initial_view_state=view_state, tooltip={"text": "{origin} → {destination} ({count})"})
        st.pydeck_chart(deck)
    else:
        st.info("No arcs to display for the current filters.")

st.divider()

# ---------------------------
# Raw Data (expandable)
# ---------------------------
with st.expander("Show filtered data table"):
    st.dataframe(fdf.reset_index(drop=True))
