import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.constants import *


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Passenger Aircraft Comparison",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):
        st.error(
            f"Could not find `{DATA_FILE}`. "
            "Make sure it is in the same directory as app2.py."
        )
        st.stop()
    try:
        df = pd.read_csv(
            DATA_FILE,
            na_values=[
                "",
                " ",
                "n/a",
                "na",
                "not avaialble",
                "NA",
                "N/A",
                "None",
                "null",
            ],
        )

    except Exception as e:
        st.error(
            f"Could not read `{DATA_FILE}`:\n\n{e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            "The CSV is missing the following required columns:"
        )

        st.code(
            "\n".join(missing_columns)
        )

        st.stop()


    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    for col in NUMERIC_COLUMNS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Company",
            "Family",
            "Variant",
        ]
    ).copy()


    # --------------------------------------------------------
    # Preserve original ordering
    # --------------------------------------------------------

    df.insert(
        0,
        "_Original_Order",
        range(len(df)),
    )


    return df


df = load_data()


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(data):

    warnings = []


    # Range consistency
    range_check = (
        data["Range_nm"].notna()
        & data["Range_km"].notna()
    )

    if range_check.any():

        calculated_km = (
            data.loc[range_check, "Range_nm"]
            * 1.852
        )

        supplied_km = (
            data.loc[range_check, "Range_km"]
        )

        difference = (
            calculated_km - supplied_km
        ).abs()

        inconsistent = difference > 10

        if inconsistent.any():

            variants = data.loc[
                range_check
            ].loc[
                inconsistent,
                "Variant"
            ].tolist()

            warnings.append(
                "Range nm/km mismatch detected for: "
                + ", ".join(variants)
            )


    # Orders vs deliveries
    od_check = (
        data["Orders"].notna()
        & data["Deliveries"].notna()
    )

    if od_check.any():

        invalid_od = (
            data.loc[od_check, "Deliveries"]
            > data.loc[od_check, "Orders"]
        )

        if invalid_od.any():

            variants = data.loc[
                od_check
            ].loc[
                invalid_od,
                "Variant"
            ].tolist()

            warnings.append(
                "Deliveries exceed orders for: "
                + ", ".join(variants)
            )


    return warnings


validation_warnings = validate_data(df)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_min(column):

    values = df[column].dropna()
    if len(values) == 0:
        return None
    return float(values.min())

def safe_max(column):

    values = df[column].dropna()
    if len(values) == 0:
        return None
    return float(values.max())

def format_range(row):

    nm = row.get("Range_nm")
    km = row.get("Range_km")
    if pd.isna(nm) and pd.isna(km):
        return "N/A"
    if pd.isna(nm):
        return f"{km:,.0f} km"
    if pd.isna(km):
        return f"{nm:,.0f} nm"
    return (
        f"{nm:,.0f} nm "
        f"({km:,.0f} km)"
    )

def format_optional(value, suffix=""):

    if pd.isna(value):
        return "N/A"
    return f"{value}{suffix}"

def make_hover_data(data):

    return np.column_stack(
        [
            data["Company"].fillna(""),
            data["Family"].fillna(""),
            data["Generation"].fillna(""),
            data["Typical_Seat_Config"].fillna(""),
            data["Seats"].fillna(np.nan),
            data["Range_nm"].fillna(np.nan),
            data["Range_km"].fillna(np.nan),
            data["MTOW_t"].fillna(np.nan),
            data["Orders"].fillna(np.nan),
            data["Deliveries"].fillna(np.nan),
            data["Fuel_kg_h"].fillna(np.nan),
            data["Fuel_L_h"].fillna(np.nan),
            data["Fuel_kg_seat_h"].fillna(np.nan),
            data["Backlog"].fillna(np.nan),
            data["Delivery_pct"].fillna(np.nan),
            data["Number_of_Engines"].fillna(np.nan),
            data["Launch_Year"].fillna(np.nan),
            data["Entry_Into_Service_Year"].fillna(np.nan),
            data["List_Price_USD_M"].fillna(np.nan),
            data["List_Price_Reference_Year"].fillna(np.nan),
            data["Empty_Weight_t"].fillna(np.nan),
            data["Max_Flight_Altitude_ft"].fillna(np.nan),
        ]
    )


# HOVER_TEMPLATE = """
# <b>%{text}</b><br>
# <br>
# <b>Company:</b> %{customdata[0]}<br>
# <b>Family:</b> %{customdata[1]}<br>
# <b>Generation:</b> %{customdata[2]}<br>
# <b>Configuration:</b> %{customdata[3]}<br>
# <br>
# <b>Seats:</b> %{customdata[4]:,.0f}<br>
# <b>Range:</b> %{customdata[5]:,.0f} nm
# (%{customdata[6]:,.0f} km)<br>
# <b>MTOW:</b> %{customdata[7]:,.1f} t<br>
# <br>
# <b>Fuel:</b> %{customdata[10]:,.0f} kg/hr<br>
# <b>Fuel:</b> %{customdata[11]:,.0f} L/hr<br>
# <b>Fuel/seat:</b> %{customdata[12]:,.2f} kg/seat/hr<br>
# <br>
# <b>Orders:</b> %{customdata[8]:,.0f}<br>
# <b>Deliveries:</b> %{customdata[9]:,.0f}<br>
# <b>Backlog:</b> %{customdata[13]:,.0f}<br>
# <b>Delivery:</b> %{customdata[14]:,.1f}%<br>
# <br>
# <b>Engines:</b> %{customdata[15]:,.0f}<br>
# <b>Launch:</b> %{customdata[16]:,.0f}<br>
# <b>EIS:</b> %{customdata[17]:,.0f}<br>
# <b>List price:</b> %{customdata[18]:,.1f}M USD<br>
# <b>Price reference:</b> %{customdata[19]:,.0f}<br>
# <b>Empty weight:</b> %{customdata[20]:,.1f} t<br>
# <b>Max altitude:</b> %{customdata[21]:,.0f} ft
# <extra></extra>
# """

# ============================================================
# STANDARD CHART LEGEND
# ============================================================

STANDARD_LEGEND = dict(
    itemsizing="constant",
    font=dict(size=12),
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("✈️ Filters")


# ============================================================
# COMPANY FILTER
# ============================================================

companies = sorted(
    df["Company"]
    .dropna()
    .unique()
    .tolist()
)

selected_companies = st.sidebar.multiselect(
    "Manufacturer",
    options=companies,
    default=companies,
)


# ============================================================
# FAMILY FILTER
# ============================================================

families = sorted(
    df["Family"]
    .dropna()
    .unique()
    .tolist()
)

selected_families = st.sidebar.multiselect(
    "Brand Family",
    options=families,
    default=families,
)

# ============================================================
# AIRCRAFT BUILD TYPE FILTER
# ============================================================

build_types = sorted(
    df["Aircraft_Build_Type"]
    .dropna()
    .unique()
    .tolist()
)

selected_build_types = st.sidebar.multiselect(
    "Aircraft Build Type",
    options=build_types,
    default=build_types,
)


# ============================================================
# GENERATION FILTER
# ============================================================

generations = sorted(
    df["Generation"]
    .dropna()
    .unique()
    .tolist()
)

with st.sidebar.expander(
    "Generation",
    expanded=False,
):

    selected_generations = st.multiselect(
        "Generation",
        options=generations,
        default=generations,
        label_visibility="collapsed",
    )



# ============================================================
# AIRCRAFT VARIANT FILTER
# ============================================================

all_variants = sorted(
    df["Variant"]
    .dropna()
    .unique()
    .tolist()
)

with st.sidebar.expander(
    "Aircraft Variant",
    expanded=False,
):

    selected_variants = st.multiselect(
        "Aircraft Variant",
        options=all_variants,
        default=all_variants,
        label_visibility="collapsed",
    )


# ============================================================
# SLIDER FUNCTION
# ============================================================

def numeric_slider(
    label,
    column,
    step,
    number_format,
):

    values = df[column].dropna()

    if len(values) == 0:

        st.sidebar.caption(
            f"{label}: no data available"
        )

        return None


    minimum = float(values.min())
    maximum = float(values.max())


    # Avoid invalid slider if only one value exists

    if minimum == maximum:

        st.sidebar.caption(
            f"{label}: {minimum:{number_format}}"
        )

        return (
            minimum,
            maximum,
        )


    return st.sidebar.slider(
        label,

        min_value=minimum,
        max_value=maximum,

        value=(
            minimum,
            maximum,
        ),

        step=step,

        format=number_format,
    )


# ============================================================
# CORE PERFORMANCE FILTERS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "Performance"
)


seat_filter = numeric_slider(
    "Seating capacity",
    "Seats",
    step=1.0,
    number_format="%.0f",
)


range_nm_filter = numeric_slider(
    "Range (nm)",
    "Range_nm",
    step=50.0,
    number_format="%.0f",
)


range_km_filter = numeric_slider(
    "Range (km)",
    "Range_km",
    step=100.0,
    number_format="%.0f",
)


mtow_filter = numeric_slider(
    "MTOW (tonnes)",
    "MTOW_t",
    step=1.0,
    number_format="%.1f",
)


# ============================================================
# FUEL FILTERS
# ============================================================

st.sidebar.subheader(
    "Fuel burn"
)


fuel_l_filter = numeric_slider(
    "Fuel Burn (litres/hr)",
    "Fuel_L_h",
    step=100.0,
    number_format="%.0f",
)


fuel_kg_filter = numeric_slider(
    "Fuel Burn (kg/hr)",
    "Fuel_kg_h",
    step=100.0,
    number_format="%.0f",
)


fuel_seat_filter = numeric_slider(
    "Fuel Burn (kg/seat/hr)",
    "Fuel_kg_seat_h",
    step=0.5,
    number_format="%.2f",
)


# ============================================================
# ADDITIONAL CHARACTERISTICS
# ============================================================

st.sidebar.subheader(
    "Aircraft characteristics"
)


engine_options = sorted(
    df["Number_of_Engines"]
    .dropna()
    .unique()
    .astype(int)
    .tolist()
)

selected_engines = st.sidebar.multiselect(
    "Number of Engines",
    options=engine_options,
    default=engine_options,
    format_func=lambda x: f"{x} engines",
)

launch_year_filter = numeric_slider(
    "Launch Year",
    "Launch_Year",
    step=1.0,
    number_format="%.0f",
)


eis_filter = numeric_slider(
    "Entry Into Service",
    "Entry_Into_Service_Year",
    step=1.0,
    number_format="%.0f",
)


empty_weight_filter = numeric_slider(
    "Empty Weight (tonnes)",
    "Empty_Weight_t",
    step=1.0,
    number_format="%.1f",
)


altitude_filter = numeric_slider(
    "Max Flight Altitude (ft)",
    "Max_Flight_Altitude_ft",
    step=500.0,
    number_format="%.0f",
)


price_filter = numeric_slider(
    "List Price (USD million)",
    "List_Price_USD_M",
    step=5.0,
    number_format="%.0f",
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = df.copy()


# ------------------------------------------------------------
# Categorical filters
# ------------------------------------------------------------

filtered = filtered[
    filtered["Company"].isin(
        selected_companies
    )
]


filtered = filtered[
    filtered["Family"].isin(
        selected_families
    )
]

filtered = filtered[
    filtered["Aircraft_Build_Type"].isin(
        selected_build_types
    )
]

filtered = filtered[
    filtered["Generation"].isin(
        selected_generations
    )
]


filtered = filtered[
    filtered["Variant"].isin(
        selected_variants
    )
]


# ------------------------------------------------------------
# Numeric filter helper
#
# IMPORTANT:
# Missing values are retained rather than automatically
# removing the aircraft. This is useful for historical fields
# where price/weight may genuinely be unavailable.
# ------------------------------------------------------------

def apply_numeric_filter(
    data,
    column,
    selection,
    keep_missing=False,
):

    if selection is None:
        return data

    minimum, maximum = selection

    if keep_missing:

        return data[
            data[column].isna()
            |
            data[column].between(
                minimum,
                maximum,
            )
        ]

    return data[
        data[column].between(
            minimum,
            maximum,
        )
    ]


filtered = apply_numeric_filter(
    filtered,
    "Seats",
    seat_filter,
)


filtered = apply_numeric_filter(
    filtered,
    "Range_nm",
    range_nm_filter,
)


filtered = apply_numeric_filter(
    filtered,
    "Range_km",
    range_km_filter,
)


filtered = apply_numeric_filter(
    filtered,
    "MTOW_t",
    mtow_filter,
)


filtered = apply_numeric_filter(
    filtered,
    "Fuel_L_h",
    fuel_l_filter,
)


filtered = apply_numeric_filter(
    filtered,
    "Fuel_kg_h",
    fuel_kg_filter,
)


filtered = apply_numeric_filter(
    filtered,
    "Fuel_kg_seat_h",
    fuel_seat_filter,
)


if selected_engines:
    filtered = filtered[
        filtered["Number_of_Engines"].isin(
            selected_engines
        )
    ]
else:
    filtered = filtered.iloc[0:0]


filtered = apply_numeric_filter(
    filtered,
    "Launch_Year",
    launch_year_filter,
    keep_missing=True,
)


filtered = apply_numeric_filter(
    filtered,
    "Entry_Into_Service_Year",
    eis_filter,
    keep_missing=True,
)


filtered = apply_numeric_filter(
    filtered,
    "Empty_Weight_t",
    empty_weight_filter,
    keep_missing=True,
)


filtered = apply_numeric_filter(
    filtered,
    "Max_Flight_Altitude_ft",
    altitude_filter,
    keep_missing=True,
)


filtered = apply_numeric_filter(
    filtered,
    "List_Price_USD_M",
    price_filter,
    keep_missing=True,
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "✈️ Global Passenger Aircraft Comparison"
)

st.markdown(
    """
**Airbus vs Boeing**

A220 • A319 • A320 • A321 • A330 • A340 • A350 • A380 •
717 • 737 • 747 • 757 • 767 • 777 • 787 • E-Jets E2
"""
)


# ============================================================
# DATASET INFO
# ============================================================

if validation_warnings:

    with st.expander(
        "⚠️ Data validation warnings",
        expanded=False,
    ):

        for warning in validation_warnings:

            st.warning(warning)


# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)


with k1:

    st.metric(
        "Variants",
        len(filtered),
        f"of {len(df)}",
    )


with k2:

    if len(filtered):

        largest = filtered.loc[
            filtered["Seats"].idxmax()
        ]

        st.metric(
            "Most seats",
            largest["Variant"],
            f"{largest['Seats']:,.0f}",
        )

    else:

        st.metric(
            "Most seats",
            "—",
        )


with k3:

    if len(filtered):

        longest = filtered.loc[
            filtered["Range_km"].idxmax()
        ]

        st.metric(
            "Longest range",
            longest["Variant"],
            f"{longest['Range_km']:,.0f} km",
        )

    else:

        st.metric(
            "Longest range",
            "—",
        )


with k4:

    if len(filtered):

        heaviest = filtered.loc[
            filtered["MTOW_t"].idxmax()
        ]

        st.metric(
            "Highest MTOW",
            heaviest["Variant"],
            f"{heaviest['MTOW_t']:,.1f} t",
        )

    else:

        st.metric(
            "Highest MTOW",
            "—",
        )


with k5:

    if len(filtered):

        best_eff = filtered.loc[
            filtered["Fuel_kg_seat_h"].idxmin()
        ]

        st.metric(
            "Best fuel efficiency",
            best_eff["Variant"],
            f"{best_eff['Fuel_kg_seat_h']:.2f} kg/seat/hr",
        )

    else:

        st.metric(
            "Best efficiency",
            "—",
        )


with k6:

    if len(filtered):

        ordered = filtered.loc[
            filtered["Orders"].idxmax()
        ]

        st.metric(
            "Most ordered",
            ordered["Variant"],
            f"{ordered['Orders']:,.0f}",
        )

    else:

        st.metric(
            "Most ordered",
            "—",
        )


# ============================================================
# TABS
# ============================================================

(
    tab_dashboard,
    tab_performance,
    tab_fuel,
    tab_commercial,
    tab_characteristics,
    tab_rankings,
    tab_data,
    tab_sources,
) = st.tabs(
    [
        "📊 Dashboard",
        "✈️ Performance",
        "⛽ Fuel & Efficiency",
        "📦 Commercial",
        "🔧 Aircraft Characteristics",
        "🏆 Rankings",
        "🗃️ Data",
        "📚 Sources & Methodology",
    ]
)

# ============================================================
# DASHBOARD
# ============================================================


with tab_dashboard:

    st.subheader("Top Highlghts/Visuals on one screen")

    if filtered.empty:
        st.warning("No aircraft match the current filters.")
    else:
        # ============================================================
        # TAB-LEVEL MARKER SETTINGS
        # Shared marker configuration used by charts in this tab.
        # ============================================================

        # Marker Size Limits
        MARKER_SIZE_MIN = 8
        MARKER_SIZE_MAX = 45
        MARKER_SIZE_SINGLE = 25

        # Marker Size Scaling
        ## This toggle is switched off by default.
        scale_marker_to_selection = st.toggle(
            "Scale marker size based on User Selection?", value=False,
        )

        if scale_marker_to_selection:
            st.info(
                "Marker sizes are scaled relative to the "
                "currently selected aircraft."
            )

            size_min = filtered["MTOW_t"].min()
            size_max = filtered["MTOW_t"].max()

        else:
            st.info(
                "Default marker sizing is based on the "
                "entire aircraft dataset."
            )

            # df is the complete, unfiltered dataset containing all aircraft.
            size_min = df["MTOW_t"].min()
            size_max = df["MTOW_t"].max()

        # Calculate Marker Sizes
        filtered_plot = filtered.copy()
        valid_mtow = filtered_plot["MTOW_t"].notna()

        if size_min == size_max:

            # Single aircraft / identical MTOW values.
            filtered_plot["Marker_Size"] = MARKER_SIZE_SINGLE

        else:

            filtered_plot["Marker_Size"] = MARKER_SIZE_SINGLE

            filtered_plot.loc[valid_mtow, "Marker_Size"] = np.interp(
                np.log(
                    filtered_plot.loc[
                        valid_mtow,
                        "MTOW_t",
                    ]
                ),
                [
                    np.log(size_min),
                    np.log(size_max),
                ],
                [
                    MARKER_SIZE_MIN,
                    MARKER_SIZE_MAX,
                ],
            )

        # ============================================================
        # RANGE VS SEATING CAPACITY
        # ============================================================

        fig_range_seats = go.Figure()

        for company in selected_companies:

            d = filtered_plot[
                filtered_plot["Company"] == company
            ]

            if d.empty:
                continue

            fig_range_seats.add_trace(
                go.Scatter(
                    x=d["Range_km"],
                    y=d["Seats"],

                    mode="markers+text",

                    text=d["Variant"],
                    textposition="top center",
                    textfont=dict(
                        size=8,
                    ),

                    marker=dict(
                        size=d["Marker_Size"],

                        # Manufacturer-specific marker shape.
                        # Circle is the fallback for unknown companies.
                        symbol=MARKER_SHAPE_CONSTANTS.get(
                            company,
                            "circle",
                        ),

                        color=COMPANY_COLORS.get(
                            company,
                            "#666666",
                        ),

                        opacity=0.82,

                        line=dict(
                            color="white",
                            width=1,
                        ),
                    ),

                    customdata=make_hover_data(d),
                    hovertemplate=HOVER_TEMPLATE,

                    name=company,
                    legendgroup=company,
                )
            )

        fig_range_seats.update_layout(
            title="Range vs Seating Capacity",

            xaxis_title="Range (km)",
            yaxis_title="Typical Seats",

            height=600,

            template="plotly_white",

            margin=dict(
                l=60,
                r=30,
                t=80,
                b=50,
            ),

            legend=dict(
                orientation="h",
                y=1.08,
                x=0.5,
                xanchor="center",
                itemsizing="constant",
            ),
        )

        st.plotly_chart(
            fig_range_seats,
            use_container_width=True,
        )

