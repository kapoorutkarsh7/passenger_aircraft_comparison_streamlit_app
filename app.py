import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
# CONSTANTS
# ============================================================

DATA_FILE = "aircraft_data.csv"
JET_A_DENSITY_KG_L = 0.80
COMPANY_COLORS = {
    "Airbus": "#0879B8",
    "Boeing": "#C62828",
}

FAMILY_COLORS = {
    "A220": "#009E73",
    "A319": "#56B4E9",
    "A320": "#0072B2",
    "A330": "#0072B2",
    "A340": "#56B4E9",
    "A350": "#009E73",
    "A380": "#00A6A6",

    "717": "#D55E00",
    "737": "#E69F00",
    "747": "#F0E442",
    "757": "#CC79A7",
    "767": "#8C6BB1",
    "777": "#CC79A7",
    "787": "#7A3E9D",

    "E-Jets E2": "#009E73",
}


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Company",
    "Family",
    "Variant",
    "Generation",
    "Aircraft_Build_Type",
    "Typical_Seat_Config",
    "Seats",
    "Range_nm",
    "Range_km",
    "MTOW_t",
    "Orders",
    "Deliveries",
    "Fuel_kg_h",
    "Fuel_L_h",
    "Fuel_kg_seat_h",
    "Fuel_L_seat_h",
    "Backlog",
    "Delivery_pct",
    "Number_of_Engines",
    "Launch_Year",
    "Entry_Into_Service_Year",
    "List_Price_USD_M",
    "List_Price_Reference_Year",
    "Empty_Weight_t",
    "Max_Flight_Altitude_ft",
    "Technical_Data_Source",
    "Commercial_Data_Source",
    "Notes",
]


# ============================================================
# NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "Seats",
    "Range_nm",
    "Range_km",
    "MTOW_t",
    "Orders",
    "Deliveries",
    "Fuel_kg_h",
    "Fuel_L_h",
    "Fuel_kg_seat_h",
    "Fuel_L_seat_h",
    "Backlog",
    "Delivery_pct",
    "Number_of_Engines",
    "Launch_Year",
    "Entry_Into_Service_Year",
    "List_Price_USD_M",
    "List_Price_Reference_Year",
    "Empty_Weight_t",
    "Max_Flight_Altitude_ft",
]


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


HOVER_TEMPLATE = """
<b>%{text}</b><br>
<br>
<b>Company:</b> %{customdata[0]}<br>
<b>Family:</b> %{customdata[1]}<br>
<b>Generation:</b> %{customdata[2]}<br>
<b>Configuration:</b> %{customdata[3]}<br>
<br>
<b>Seats:</b> %{customdata[4]:,.0f}<br>
<b>Range:</b> %{customdata[5]:,.0f} nm
(%{customdata[6]:,.0f} km)<br>
<b>MTOW:</b> %{customdata[7]:,.1f} t<br>
<br>
<b>Fuel:</b> %{customdata[10]:,.0f} kg/hr<br>
<b>Fuel:</b> %{customdata[11]:,.0f} L/hr<br>
<b>Fuel/seat:</b> %{customdata[12]:,.2f} kg/seat/hr<br>
<br>
<b>Orders:</b> %{customdata[8]:,.0f}<br>
<b>Deliveries:</b> %{customdata[9]:,.0f}<br>
<b>Backlog:</b> %{customdata[13]:,.0f}<br>
<b>Delivery:</b> %{customdata[14]:,.1f}%<br>
<br>
<b>Engines:</b> %{customdata[15]:,.0f}<br>
<b>Launch:</b> %{customdata[16]:,.0f}<br>
<b>EIS:</b> %{customdata[17]:,.0f}<br>
<b>List price:</b> %{customdata[18]:,.1f}M USD<br>
<b>Price reference:</b> %{customdata[19]:,.0f}<br>
<b>Empty weight:</b> %{customdata[20]:,.1f} t<br>
<b>Max altitude:</b> %{customdata[21]:,.0f} ft
<extra></extra>
"""


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("✈️ Aircraft Filters")


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
    "Company",
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
# GENERATION FILTER
# ============================================================

generations = sorted(
    df["Generation"]
    .dropna()
    .unique()
    .tolist()
)

selected_generations = st.sidebar.multiselect(
    "Generation",
    options=generations,
    default=generations,
)


# ============================================================
# VARIANT FILTER
# ============================================================

with st.sidebar.expander(
    "Aircraft Variant",
    expanded=False,
):

    all_variants = sorted(
        df["Variant"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_variants = st.multiselect(
        "Variants",
        options=all_variants,
        default=all_variants,
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

    st.subheader(
        "All major dimensions on one screen"
    )

    if len(filtered) == 0:

        st.warning(
            "No aircraft match the current filters."
        )

    else:

        fig = make_subplots(
            rows=2,
            cols=2,

            subplot_titles=[
                "Range vs Seating Capacity",
                "MTOW vs Seating Capacity",
                "Orders vs Deliveries",
                "Fuel Burn vs Seating Capacity",
            ],

            horizontal_spacing=0.08,
            vertical_spacing=0.14,
        )


        # ----------------------------------------------------
        # RANGE / SEATS
        # ----------------------------------------------------

        for company in selected_companies:

            d = filtered[
                filtered["Company"] == company
            ]

            if len(d) == 0:
                continue

            fig.add_trace(

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
                        size=np.maximum(
                            d["MTOW_t"] / 7,
                            8,
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
                ),

                row=1,
                col=1,
            )


        # ----------------------------------------------------
        # MTOW / SEATS
        # ----------------------------------------------------

        for company in selected_companies:

            d = filtered[
                filtered["Company"] == company
            ]

            if len(d) == 0:
                continue

            fig.add_trace(

                go.Scatter(

                    x=d["MTOW_t"],
                    y=d["Seats"],

                    mode="markers+text",

                    text=d["Variant"],

                    textposition="top center",

                    textfont=dict(
                        size=8,
                    ),

                    marker=dict(
                        size=np.maximum(
                            d["Range_km"] / 150,
                            8,
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

                    showlegend=False,
                ),

                row=1,
                col=2,
            )


        # ----------------------------------------------------
        # ORDERS / DELIVERIES
        # ----------------------------------------------------

        for company in selected_companies:

            d = filtered[
                filtered["Company"] == company
            ]

            if len(d) == 0:
                continue

            fig.add_trace(

                go.Scatter(

                    x=d["Orders"],
                    y=d["Deliveries"],

                    mode="markers+text",

                    text=d["Variant"],

                    textposition="top center",

                    textfont=dict(
                        size=8,
                    ),

                    marker=dict(
                        size=np.maximum(
                            d["MTOW_t"] / 7,
                            8,
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

                    showlegend=False,
                ),

                row=2,
                col=1,
            )


        # ----------------------------------------------------
        # FUEL / SEATS
        # ----------------------------------------------------

        for company in selected_companies:

            d = filtered[
                filtered["Company"] == company
            ]

            if len(d) == 0:
                continue

            fig.add_trace(

                go.Scatter(

                    x=d["Fuel_kg_h"],
                    y=d["Seats"],

                    mode="markers+text",

                    text=d["Variant"],

                    textposition="top center",

                    textfont=dict(
                        size=8,
                    ),

                    marker=dict(
                        size=np.maximum(
                            d["Range_km"] / 150,
                            8,
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

                    showlegend=False,
                ),

                row=2,
                col=2,
            )


        # ----------------------------------------------------
        # AXES
        # ----------------------------------------------------

        fig.update_xaxes(
            title_text="Range (km)",
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title_text="Typical seats",
            row=1,
            col=1,
        )


        fig.update_xaxes(
            title_text="MTOW (tonnes)",
            row=1,
            col=2,
        )

        fig.update_yaxes(
            title_text="Typical seats",
            row=1,
            col=2,
        )


        fig.update_xaxes(
            title_text="Orders",
            row=2,
            col=1,
        )

        fig.update_yaxes(
            title_text="Deliveries",
            row=2,
            col=1,
        )


        fig.update_xaxes(
            title_text="Fuel burn (kg/hr)",
            row=2,
            col=2,
        )

        fig.update_yaxes(
            title_text="Typical seats",
            row=2,
            col=2,
        )


        fig.update_layout(
            height=850,

            template="plotly_white",

            margin=dict(
                l=60,
                r=30,
                t=90,
                b=50,
            ),

            legend=dict(
                orientation="h",
                y=1.08,
                x=0.5,
                xanchor="center",
            ),
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        st.caption(
            "Bubble size represents a third metric and varies by chart. "
            "Hover over any aircraft for the full dataset record."
        )


# ============================================================
# PERFORMANCE
# ============================================================

with tab_performance:

    st.subheader(
        "Performance comparison"
    )


    if len(filtered) == 0:

        st.warning(
            "No aircraft match the current filters."
        )

    else:

        p1, p2 = st.columns(2)


        # ----------------------------------------------------
        # RANGE / SEATS
        # ----------------------------------------------------

        with p1:

            fig = go.Figure()

            for company in selected_companies:

                d = filtered[
                    filtered["Company"] == company
                ]

                if len(d) == 0:
                    continue

                fig.add_trace(

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
                            size=np.maximum(
                                d["MTOW_t"] / 7,
                                8,
                            ),

                            color=COMPANY_COLORS.get(
                                company,
                                "#666666",
                            ),
                        ),

                        customdata=make_hover_data(d),

                        hovertemplate=HOVER_TEMPLATE,

                        name=company,
                    )
                )


            fig.update_layout(
                title="Range vs Seating",

                xaxis_title="Range (km)",

                yaxis_title="Typical seats",

                height=600,

                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ----------------------------------------------------
        # MTOW / SEATS
        # ----------------------------------------------------

        with p2:

            fig = go.Figure()

            for company in selected_companies:

                d = filtered[
                    filtered["Company"] == company
                ]

                if len(d) == 0:
                    continue

                fig.add_trace(

                    go.Scatter(

                        x=d["MTOW_t"],
                        y=d["Seats"],

                        mode="markers+text",

                        text=d["Variant"],

                        textposition="top center",

                        textfont=dict(
                            size=8,
                        ),

                        marker=dict(
                            size=np.maximum(
                                d["Range_km"] / 150,
                                8,
                            ),

                            color=COMPANY_COLORS.get(
                                company,
                                "#666666",
                            ),
                        ),

                        customdata=make_hover_data(d),

                        hovertemplate=HOVER_TEMPLATE,

                        name=company,
                    )
                )


            fig.update_layout(
                title="MTOW vs Seating",

                xaxis_title="MTOW (tonnes)",

                yaxis_title="Typical seats",

                height=600,

                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ----------------------------------------------------
        # RANGE / MTOW
        # ----------------------------------------------------

        st.subheader(
            "Range vs MTOW"
        )


        fig = go.Figure()


        for company in selected_companies:

            d = filtered[
                filtered["Company"] == company
            ]

            if len(d) == 0:
                continue


            fig.add_trace(

                go.Scatter(

                    x=d["Range_km"],
                    y=d["MTOW_t"],

                    mode="markers+text",

                    text=d["Variant"],

                    textposition="top center",

                    textfont=dict(
                        size=8,
                    ),

                    marker=dict(
                        size=np.maximum(
                            d["Seats"] / 10,
                            8,
                        ),

                        color=COMPANY_COLORS.get(
                            company,
                            "#666666",
                        ),
                    ),

                    customdata=make_hover_data(d),

                    hovertemplate=HOVER_TEMPLATE,

                    name=company,
                )
            )


        fig.update_layout(
            title="Range vs MTOW — bubble size = seats",

            xaxis_title="Range (km)",

            yaxis_title="MTOW (tonnes)",

            height=650,

            template="plotly_white",
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# FUEL
# ============================================================

with tab_fuel:

    st.subheader(
        "Fuel burn & efficiency"
    )


    st.info(
        "Fuel-flow figures are representative estimates rather "
        "than certified aircraft performance. Actual fuel flow "
        "varies with aircraft weight, engine variant, altitude, "
        "weather, payload, flight phase and operating conditions."
    )


    if len(filtered) == 0:

        st.warning(
            "No aircraft match the current filters."
        )

    else:

        f1, f2 = st.columns(2)


        # ----------------------------------------------------
        # ABSOLUTE FUEL
        # ----------------------------------------------------

        with f1:

            fig = go.Figure()


            for company in selected_companies:

                d = filtered[
                    filtered["Company"] == company
                ]

                if len(d) == 0:
                    continue


                fig.add_trace(

                    go.Scatter(

                        x=d["Fuel_kg_h"],
                        y=d["Seats"],

                        mode="markers+text",

                        text=d["Variant"],

                        textposition="top center",

                        textfont=dict(
                            size=8,
                        ),

                        marker=dict(
                            size=np.maximum(
                                d["Range_km"] / 150,
                                8,
                            ),

                            color=COMPANY_COLORS.get(
                                company,
                                "#666666",
                            ),
                        ),

                        customdata=make_hover_data(d),

                        hovertemplate=HOVER_TEMPLATE,

                        name=company,
                    )
                )


            fig.update_layout(
                title="Absolute Fuel Burn",

                xaxis_title="Fuel burn (kg/hour)",

                yaxis_title="Typical seats",

                height=600,

                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ----------------------------------------------------
        # PER-SEAT FUEL
        # ----------------------------------------------------

        with f2:

            fig = go.Figure()


            for company in selected_companies:

                d = filtered[
                    filtered["Company"] == company
                ]

                if len(d) == 0:
                    continue


                fig.add_trace(

                    go.Scatter(

                        x=d["Fuel_kg_seat_h"],
                        y=d["Range_km"],

                        mode="markers+text",

                        text=d["Variant"],

                        textposition="top center",

                        textfont=dict(
                            size=8,
                        ),

                        marker=dict(
                            size=np.maximum(
                                d["Seats"] / 10,
                                8,
                            ),

                            color=COMPANY_COLORS.get(
                                company,
                                "#666666",
                            ),
                        ),

                        customdata=make_hover_data(d),

                        hovertemplate=HOVER_TEMPLATE,

                        name=company,
                    )
                )


            fig.update_layout(
                title="Fuel Efficiency vs Range",

                xaxis_title="Fuel burn (kg / seat / hour)",

                yaxis_title="Range (km)",

                height=600,

                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ----------------------------------------------------
        # LITRES / SEAT
        # ----------------------------------------------------

        st.subheader(
            "Fuel burn in litres"
        )


        fig = go.Figure()


        for company in selected_companies:

            d = filtered[
                filtered["Company"] == company
            ]

            if len(d) == 0:
                continue


            fuel_l_seat = (
                d["Fuel_L_h"] /
                d["Seats"]
            )


            fig.add_trace(

                go.Scatter(

                    x=fuel_l_seat,

                    y=d["Range_km"],

                    mode="markers+text",

                    text=d["Variant"],

                    textposition="top center",

                    textfont=dict(
                        size=8,
                    ),

                    marker=dict(
                        size=np.maximum(
                            d["MTOW_t"] / 7,
                            8,
                        ),

                        color=COMPANY_COLORS.get(
                            company,
                            "#666666",
                        ),
                    ),

                    customdata=make_hover_data(d),

                    hovertemplate=HOVER_TEMPLATE,

                    name=company,
                )
            )


        fig.update_layout(
            title="Fuel Burn per Seat vs Range",

            xaxis_title="Fuel burn (L / seat / hour)",

            yaxis_title="Range (km)",

            height=600,

            template="plotly_white",
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# COMMERCIAL
# ============================================================

with tab_commercial:

    st.subheader(
        "Orders, deliveries & backlog"
    )


    if len(filtered) == 0:

        st.warning(
            "No aircraft match the current filters."
        )

    else:

        c1, c2 = st.columns(2)


        # ----------------------------------------------------
        # ORDERS / DELIVERIES
        # ----------------------------------------------------

        with c1:

            fig = go.Figure()


            for company in selected_companies:

                d = filtered[
                    filtered["Company"] == company
                ]

                if len(d) == 0:
                    continue


                fig.add_trace(

                    go.Scatter(

                        x=d["Orders"],
                        y=d["Deliveries"],

                        mode="markers+text",

                        text=d["Variant"],

                        textposition="top center",

                        textfont=dict(
                            size=8,
                        ),

                        marker=dict(
                            size=np.maximum(
                                d["MTOW_t"] / 7,
                                8,
                            ),

                            color=COMPANY_COLORS.get(
                                company,
                                "#666666",
                            ),
                        ),

                        customdata=make_hover_data(d),

                        hovertemplate=HOVER_TEMPLATE,

                        name=company,
                    )
                )


            maximum = max(
                filtered["Orders"].max(),
                filtered["Deliveries"].max(),
            )


            fig.add_shape(

                type="line",

                x0=0,
                y0=0,

                x1=maximum,
                y1=maximum,

                line=dict(
                    color="gray",
                    dash="dash",
                ),
            )


            fig.update_layout(
                title="Orders vs Deliveries",

                xaxis_title="Orders",

                yaxis_title="Deliveries",

                height=600,

                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ----------------------------------------------------
        # BACKLOG
        # ----------------------------------------------------

        with c2:

            backlog = (
                filtered
                .sort_values(
                    "Backlog",
                    ascending=True,
                )
            )


            fig = go.Figure()


            fig.add_trace(

                go.Bar(

                    x=backlog["Backlog"],

                    y=backlog["Variant"],

                    orientation="h",

                    marker_color=[
                        COMPANY_COLORS.get(
                            company,
                            "#666666",
                        )
                        for company
                        in backlog["Company"]
                    ],

                    text=backlog["Backlog"],

                    textposition="outside",

                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Backlog: %{x:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )


            fig.update_layout(
                title="Aircraft Backlog",

                xaxis_title="Aircraft remaining to deliver",

                height=600,

                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ----------------------------------------------------
        # COMMERCIAL TABLE
        # ----------------------------------------------------

        st.subheader(
            "Commercial ranking"
        )


        commercial = (
            filtered[
                [
                    "Company",
                    "Family",
                    "Variant",
                    "Seats",
                    "Orders",
                    "Deliveries",
                    "Backlog",
                    "Delivery_pct",
                ]
            ]
            .sort_values(
                "Orders",
                ascending=False,
            )
            .reset_index(drop=True)
        )


        st.dataframe(

            commercial,

            column_config={

                "Company": "Company",

                "Family": "Family",

                "Variant": "Variant",

                "Seats": st.column_config.NumberColumn(
                    "Seats",
                    format="%.0f",
                ),

                "Orders": st.column_config.NumberColumn(
                    "Orders",
                    format="%,.0f",
                ),

                "Deliveries": st.column_config.NumberColumn(
                    "Deliveries",
                    format="%,.0f",
                ),

                "Backlog": st.column_config.NumberColumn(
                    "Backlog",
                    format="%,.0f",
                ),

                "Delivery_pct": st.column_config.NumberColumn(
                    "Delivery %",
                    format="%.1f%%",
                ),
            },

            hide_index=True,

            use_container_width=True,
        )


# ============================================================
# CHARACTERISTICS
# ============================================================

with tab_characteristics:

    st.subheader(
        "Aircraft characteristics"
    )


    if len(filtered) == 0:

        st.warning(
            "No aircraft match the current filters."
        )

    else:

        a1, a2 = st.columns(2)


        # ----------------------------------------------------
        # ENGINES / MTOW
        # ----------------------------------------------------

        with a1:

            fig = go.Figure()


            for company in selected_companies:

                d = filtered[
                    filtered["Company"] == company
                ]

                if len(d) == 0:
                    continue


                fig.add_trace(

                    go.Scatter(

                        x=d["MTOW_t"],

                        y=d["Number_of_Engines"],

                        mode="markers+text",

                        text=d["Variant"],

                        textposition="top center",

                        textfont=dict(
                            size=8,
                        ),

                        marker=dict(
                            size=np.maximum(
                                d["Seats"] / 10,
                                8,
                            ),

                            color=COMPANY_COLORS.get(
                                company,
                                "#666666",
                            ),
                        ),

                        customdata=make_hover_data(d),

                        hovertemplate=HOVER_TEMPLATE,

                        name=company,
                    )
                )


            fig.update_layout(
                title="Number of Engines vs MTOW",

                xaxis_title="MTOW (tonnes)",

                yaxis_title="Number of engines",

                height=600,

                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ----------------------------------------------------
        # EMPTY WEIGHT / MTOW
        # ----------------------------------------------------

        with a2:

            fig = go.Figure()


            for company in selected_companies:

                d = filtered[
                    filtered["Company"] == company
                ]

                d = d.dropna(
                    subset=[
                        "Empty_Weight_t",
                        "MTOW_t",
                    ]
                )


                if len(d) == 0:
                    continue


                fig.add_trace(

                    go.Scatter(

                        x=d["MTOW_t"],

                        y=d["Empty_Weight_t"],

                        mode="markers+text",

                        text=d["Variant"],

                        textposition="top center",

                        textfont=dict(
                            size=8,
                        ),

                        marker=dict(
                            size=np.maximum(
                                d["Seats"] / 10,
                                8,
                            ),

                            color=COMPANY_COLORS.get(
                                company,
                                "#666666",
                            ),
                        ),

                        customdata=make_hover_data(d),

                        hovertemplate=HOVER_TEMPLATE,

                        name=company,
                    )
                )


            fig.update_layout(
                title="Empty Weight vs MTOW",

                xaxis_title="MTOW (tonnes)",

                yaxis_title="Empty weight (tonnes)",

                height=600,

                template="plotly_white",
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


        # ----------------------------------------------------
        # LAUNCH YEAR / RANGE
        # ----------------------------------------------------

        st.subheader(
            "Aircraft development timeline"
        )


        timeline_data = filtered.dropna(
            subset=[
                "Launch_Year",
                "Range_km",
            ]
        )


        fig = go.Figure()


        for company in selected_companies:

            d = timeline_data[
                timeline_data["Company"] == company
            ]

            if len(d) == 0:
                continue


            fig.add_trace(

                go.Scatter(

                    x=d["Launch_Year"],

                    y=d["Range_km"],

                    mode="markers+text",

                    text=d["Variant"],

                    textposition="top center",

                    textfont=dict(
                        size=8,
                    ),

                    marker=dict(
                        size=np.maximum(
                            d["Seats"] / 10,
                            8,
                        ),

                        color=COMPANY_COLORS.get(
                            company,
                            "#666666",
                        ),
                    ),

                    customdata=make_hover_data(d),

                    hovertemplate=HOVER_TEMPLATE,

                    name=company,
                )
            )


        fig.update_layout(
            title="Launch Year vs Range",

            xaxis_title="Launch year",

            yaxis_title="Range (km)",

            height=650,

            template="plotly_white",
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ----------------------------------------------------
        # CHARACTERISTICS TABLE
        # ----------------------------------------------------

        st.subheader(
            "Aircraft characteristics table"
        )


        characteristics = filtered[
            [
                "Company",
                "Family",
                "Variant",
                "Generation",
                "Typical_Seat_Config",
                "Number_of_Engines",
                "Launch_Year",
                "Entry_Into_Service_Year",
                "List_Price_USD_M",
                "List_Price_Reference_Year",
                "Empty_Weight_t",
                "Max_Flight_Altitude_ft",
            ]
        ].sort_values(
            [
                "Company",
                "Launch_Year",
            ]
        )


        st.dataframe(

            characteristics,

            column_config={

                "Number_of_Engines":
                    st.column_config.NumberColumn(
                        "Engines",
                        format="%.0f",
                    ),

                "Launch_Year":
                    st.column_config.NumberColumn(
                        "Launch",
                        format="%.0f",
                    ),

                "Entry_Into_Service_Year":
                    st.column_config.NumberColumn(
                        "EIS",
                        format="%.0f",
                    ),

                "List_Price_USD_M":
                    st.column_config.NumberColumn(
                        "List Price (USD M)",
                        format="$%.1fM",
                    ),

                "List_Price_Reference_Year":
                    st.column_config.NumberColumn(
                        "Price Year",
                        format="%.0f",
                    ),

                "Empty_Weight_t":
                    st.column_config.NumberColumn(
                        "Empty Weight",
                        format="%.1f t",
                    ),

                "Max_Flight_Altitude_ft":
                    st.column_config.NumberColumn(
                        "Max Altitude",
                        format="%,.0f ft",
                    ),
            },

            hide_index=True,

            use_container_width=True,
        )


# ============================================================
# RANKINGS
# ============================================================

with tab_rankings:

    st.subheader(
        "Aircraft rankings"
    )


    if len(filtered) == 0:

        st.warning(
            "No aircraft match the current filters."
        )

    else:

        r1, r2 = st.columns(2)


        # ----------------------------------------------------
        # RANGE
        # ----------------------------------------------------

        with r1:

            st.markdown(
                "### 🥇 Range ranking"
            )


            rank = (
                filtered[
                    [
                        "Company",
                        "Family",
                        "Variant",
                        "Range_nm",
                        "Range_km",
                    ]
                ]
                .sort_values(
                    "Range_km",
                    ascending=False,
                )
                .reset_index(drop=True)
            )


            rank.insert(
                0,
                "Rank",
                range(1, len(rank) + 1),
            )


            st.dataframe(

                rank,

                column_config={

                    "Rank":
                        st.column_config.NumberColumn(
                            "Rank",
                            format="%d",
                        ),

                    "Range_nm":
                        st.column_config.NumberColumn(
                            "Range (nm)",
                            format="%,.0f",
                        ),

                    "Range_km":
                        st.column_config.NumberColumn(
                            "Range (km)",
                            format="%,.0f",
                        ),
                },

                hide_index=True,

                use_container_width=True,
            )


        # ----------------------------------------------------
        # SEATS
        # ----------------------------------------------------

        with r2:

            st.markdown(
                "### 🪑 Seating ranking"
            )


            rank = (
                filtered[
                    [
                        "Company",
                        "Family",
                        "Variant",
                        "Seats",
                    ]
                ]
                .sort_values(
                    "Seats",
                    ascending=False,
                )
                .reset_index(drop=True)
            )


            rank.insert(
                0,
                "Rank",
                range(1, len(rank) + 1),
            )


            st.dataframe(

                rank,

                column_config={

                    "Rank":
                        st.column_config.NumberColumn(
                            "Rank",
                            format="%d",
                        ),

                    "Seats":
                        st.column_config.NumberColumn(
                            "Seats",
                            format="%,.0f",
                        ),
                },

                hide_index=True,

                use_container_width=True,
            )


        r3, r4 = st.columns(2)


        # ----------------------------------------------------
        # MTOW
        # ----------------------------------------------------

        with r3:

            st.markdown(
                "### ⚖️ MTOW ranking"
            )


            rank = (
                filtered[
                    [
                        "Company",
                        "Family",
                        "Variant",
                        "MTOW_t",
                    ]
                ]
                .sort_values(
                    "MTOW_t",
                    ascending=False,
                )
                .reset_index(drop=True)
            )


            rank.insert(
                0,
                "Rank",
                range(1, len(rank) + 1),
            )


            st.dataframe(

                rank,

                column_config={

                    "Rank":
                        st.column_config.NumberColumn(
                            "Rank",
                            format="%d",
                        ),

                    "MTOW_t":
                        st.column_config.NumberColumn(
                            "MTOW",
                            format="%.1f t",
                        ),
                },

                hide_index=True,

                use_container_width=True,
            )


        # ----------------------------------------------------
        # FUEL EFFICIENCY
        # ----------------------------------------------------

        with r4:

            st.markdown(
                "### ⛽ Fuel efficiency ranking"
            )


            rank = (
                filtered[
                    [
                        "Company",
                        "Family",
                        "Variant",
                        "Seats",
                        "Fuel_kg_h",
                        "Fuel_L_h",
                        "Fuel_kg_seat_h",
                    ]
                ]
                .sort_values(
                    "Fuel_kg_seat_h",
                    ascending=True,
                )
                .reset_index(drop=True)
            )


            rank.insert(
                0,
                "Rank",
                range(1, len(rank) + 1),
            )


            st.dataframe(

                rank,

                column_config={

                    "Rank":
                        st.column_config.NumberColumn(
                            "Rank",
                            format="%d",
                        ),

                    "Seats":
                        st.column_config.NumberColumn(
                            "Seats",
                            format="%,.0f",
                        ),

                    "Fuel_kg_h":
                        st.column_config.NumberColumn(
                            "Fuel kg/hr",
                            format="%,.0f",
                        ),

                    "Fuel_L_h":
                        st.column_config.NumberColumn(
                            "Fuel L/hr",
                            format="%,.0f",
                        ),

                    "Fuel_kg_seat_h":
                        st.column_config.NumberColumn(
                            "Fuel kg/seat/hr",
                            format="%.2f",
                        ),
                },

                hide_index=True,

                use_container_width=True,
            )


        # ----------------------------------------------------
        # COMMERCIAL
        # ----------------------------------------------------

        st.markdown(
            "### 📦 Orders ranking"
        )


        rank = (
            filtered[
                [
                    "Company",
                    "Family",
                    "Variant",
                    "Orders",
                    "Deliveries",
                    "Backlog",
                    "Delivery_pct",
                ]
            ]
            .sort_values(
                "Orders",
                ascending=False,
            )
            .reset_index(drop=True)
        )


        rank.insert(
            0,
            "Rank",
            range(1, len(rank) + 1),
        )


        st.dataframe(

            rank,

            column_config={

                "Rank":
                    st.column_config.NumberColumn(
                        "Rank",
                        format="%d",
                    ),

                "Orders":
                    st.column_config.NumberColumn(
                        "Orders",
                        format="%,.0f",
                    ),

                "Deliveries":
                    st.column_config.NumberColumn(
                        "Deliveries",
                        format="%,.0f",
                    ),

                "Backlog":
                    st.column_config.NumberColumn(
                        "Backlog",
                        format="%,.0f",
                    ),

                "Delivery_pct":
                    st.column_config.NumberColumn(
                        "Delivery %",
                        format="%.1f%%",
                    ),
            },

            hide_index=True,

            use_container_width=True,
        )


# ============================================================
# DATA
# ============================================================

with tab_data:

    st.subheader(
        "Aircraft database"
    )


    st.write(
        f"Showing **{len(filtered)}** of "
        f"**{len(df)}** aircraft variants."
    )


    # --------------------------------------------------------
    # Column groups
    # --------------------------------------------------------

    column_groups = {

        "Identity": [
            "Company",
            "Family",
            "Variant",
            "Generation",
            "Typical_Seat_Config",
        ],

        "Performance": [
            "Seats",
            "Range_nm",
            "Range_km",
            "MTOW_t",
        ],

        "Fuel": [
            "Fuel_kg_h",
            "Fuel_L_h",
            "Fuel_kg_seat_h",
            "Fuel_L_seat_h",
        ],

        "Commercial": [
            "Orders",
            "Deliveries",
            "Backlog",
            "Delivery_pct",
        ],

        "Aircraft characteristics": [
            "Number_of_Engines",
            "Launch_Year",
            "Entry_Into_Service_Year",
            "List_Price_USD_M",
            "List_Price_Reference_Year",
            "Empty_Weight_t",
            "Max_Flight_Altitude_ft",
        ],

        "Sources": [
            "Technical_Data_Source",
            "Commercial_Data_Source",
            "Notes",
        ],
    }


    selected_columns = []


    for group_name, group_columns in column_groups.items():

        with st.expander(
            group_name,
            expanded=(
                group_name
                in [
                    "Identity",
                    "Performance",
                ]
            ),
        ):

            available = [
                col
                for col in group_columns
                if col in filtered.columns
            ]


            defaults = available.copy()


            chosen = st.multiselect(

                f"{group_name} columns",

                options=available,

                default=defaults,

                key=f"columns_{group_name}",
            )


            selected_columns.extend(chosen)


    # --------------------------------------------------------
    # Remove duplicates while preserving order
    # --------------------------------------------------------

    selected_columns = list(
        dict.fromkeys(
            selected_columns
        )
    )


    if not selected_columns:

        st.warning(
            "Select at least one column."
        )

    else:

        display_df = filtered[
            selected_columns
        ].copy()


        # ----------------------------------------------------
        # Column configuration
        # ----------------------------------------------------

        column_config = {

            "Seats":
                st.column_config.NumberColumn(
                    "Seats",
                    format="%,.0f",
                ),

            "Range_nm":
                st.column_config.NumberColumn(
                    "Range (nm)",
                    format="%,.0f",
                ),

            "Range_km":
                st.column_config.NumberColumn(
                    "Range (km)",
                    format="%,.0f",
                ),

            "MTOW_t":
                st.column_config.NumberColumn(
                    "MTOW",
                    format="%.1f t",
                ),

            "Fuel_kg_h":
                st.column_config.NumberColumn(
                    "Fuel (kg/hr)",
                    format="%,.0f",
                ),

            "Fuel_L_h":
                st.column_config.NumberColumn(
                    "Fuel (L/hr)",
                    format="%,.0f",
                ),

            "Fuel_kg_seat_h":
                st.column_config.NumberColumn(
                    "Fuel (kg/seat/hr)",
                    format="%.2f",
                ),

            "Fuel_L_seat_h":
                st.column_config.NumberColumn(
                    "Fuel (L/seat/hr)",
                    format="%.2f",
                ),

            "Orders":
                st.column_config.NumberColumn(
                    "Orders",
                    format="%,.0f",
                ),

            "Deliveries":
                st.column_config.NumberColumn(
                    "Deliveries",
                    format="%,.0f",
                ),

            "Backlog":
                st.column_config.NumberColumn(
                    "Backlog",
                    format="%,.0f",
                ),

            "Delivery_pct":
                st.column_config.NumberColumn(
                    "Delivery %",
                    format="%.1f%%",
                ),

            "Number_of_Engines":
                st.column_config.NumberColumn(
                    "Engines",
                    format="%.0f",
                ),

            "Launch_Year":
                st.column_config.NumberColumn(
                    "Launch Year",
                    format="%.0f",
                ),

            "Entry_Into_Service_Year":
                st.column_config.NumberColumn(
                    "Entry Into Service",
                    format="%.0f",
                ),

            "List_Price_USD_M":
                st.column_config.NumberColumn(
                    "List Price (USD M)",
                    format="$%.1fM",
                ),

            "List_Price_Reference_Year":
                st.column_config.NumberColumn(
                    "Price Reference Year",
                    format="%.0f",
                ),

            "Empty_Weight_t":
                st.column_config.NumberColumn(
                    "Empty Weight",
                    format="%.1f t",
                ),

            "Max_Flight_Altitude_ft":
                st.column_config.NumberColumn(
                    "Max Altitude",
                    format="%,.0f ft",
                ),
        }


        st.dataframe(

            display_df,

            column_config=column_config,

            hide_index=True,

            use_container_width=True,

            height=650,
        )


    # --------------------------------------------------------
    # Downloads
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Download data"
    )


    filtered_csv = (
        filtered
        .drop(
            columns=[
                "_Original_Order"
            ],
            errors="ignore",
        )
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    complete_csv = (
        df
        .drop(
            columns=[
                "_Original_Order"
            ],
            errors="ignore",
        )
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    d1, d2 = st.columns(2)


    with d1:

        st.download_button(

            label="⬇️ Download filtered CSV",

            data=filtered_csv,

            file_name=(
                "aircraft_data_filtered.csv"
            ),

            mime="text/csv",

            use_container_width=True,
        )


    with d2:

        st.download_button(

            label="⬇️ Download complete CSV",

            data=complete_csv,

            file_name=(
                "aircraft_data_complete.csv"
            ),

            mime="text/csv",

            use_container_width=True,
        )


# ============================================================
# SOURCES / METHODOLOGY
# ============================================================

with tab_sources:

    st.subheader(
        "Sources & methodology"
    )


    st.markdown(
        """
## Data architecture

The CSV is the **single source of truth** for the application.

No aircraft-specific technical or commercial data are
hard-coded in `app2.py`.

This means you can update `aircraft_data.csv` without changing
the application code.


---

## Range

Range is shown in both nautical miles and kilometres.

The conversion used is:

**1 nautical mile = 1.852 km**

The dataset's `Range_km` field is retained directly from the
CSV rather than being silently replaced by an app-side
calculation.


---

## Seating

The `Seats` field represents the typical configuration
specified in the CSV.

The associated `Typical_Seat_Config` field identifies the
configuration basis, e.g. 3-class or 2-class.

Actual airline configurations can vary substantially.


---

## MTOW

Maximum Take-Off Weight is expressed in metric tonnes.


---

## Fuel burn

Fuel-flow values are representative estimates.

They should **not** be interpreted as certified aircraft
performance.

Actual fuel burn varies according to:

- aircraft weight
- engine variant
- flight altitude
- cruise speed
- payload
- weather
- route
- flight phase
- airline operating procedures


### Fuel conversion

The dataset assumes:

**Jet-A density = 0.80 kg/L**

Therefore:

`Fuel_L_h = Fuel_kg_h / 0.80`


---

## Fuel efficiency

The principal efficiency metric is:

`Fuel_kg_seat_h = Fuel_kg_h / Seats`

A lower number indicates lower estimated fuel consumption
per available seat per flight hour.

This is a **comparative indicator**, not a measure of actual
trip fuel per passenger.


---

## Orders / deliveries

Orders, deliveries and backlog are carried through from the
CSV.

Backlog is:

`Orders - Deliveries`

Delivery percentage is:

`Deliveries / Orders × 100`


---

## List prices

List prices are historical nominal USD list prices where
documented.

They are **not actual transaction prices**.

Airline purchase prices can differ substantially because of:

- negotiated discounts
- financing
- options
- engine selection
- configuration
- escalation
- support packages
- other commercial terms

`List_Price_Reference_Year` identifies the year associated
with the list-price figure.


---

## Empty weight

Empty-weight definitions can differ between aircraft
manufacturers and sources.

Consequently, these values are best used for broad comparison
rather than as perfectly accounting-equivalent measures.


---

## Maximum altitude

Maximum flight altitude represents the maximum operating /
published altitude associated with the aircraft specification.

It should not be interpreted as a typical cruise altitude.


---

## Historical aircraft

For older aircraft such as the 747 and A340, some historical
fields are inherently less complete than for current aircraft.

Where a reliable public value could not be established, the
CSV intentionally contains a blank value rather than an
invented estimate.


---

## Passenger aircraft only

The dataset excludes freighter variants, consistent with the
scope of this comparison.
"""
    )


    st.divider()


    st.subheader(
        "Source fields in the dataset"
    )


    source_columns = [
        "Variant",
        "Technical_Data_Source",
        "Commercial_Data_Source",
        "Notes",
    ]


    source_table = filtered[
        source_columns
    ].copy()


    st.dataframe(

        source_table,

        hide_index=True,

        use_container_width=True,

        height=600,
    )


    st.divider()


    st.markdown(
        """
### Principal source families

**Airbus**

Airbus aircraft technical specifications and historical
aircraft information were used for Airbus variants.

**Boeing**

Boeing aircraft technical specifications and historical
aircraft information were used for Boeing variants.

**Historical commercial information**

Historical list-price information is identified separately
from technical specifications and should not be confused with
actual transaction prices.


### Important interpretation note

Aircraft specifications change depending on configuration,
engine option, MTOW variant and source publication date.

Where manufacturer documentation gives different values for
different configurations or revisions, the CSV should be
treated as a **comparative analytical dataset**, not as an
engineering certification database.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Range conversion: 1 nautical mile = 1.852 km. "
    "Fuel litres calculated using an assumed Jet-A density "
    "of 0.80 kg/L. Fuel-flow values are representative "
    "estimates and should not be interpreted as certified "
    "performance."
)

st.caption(
    f"Dataset: {DATA_FILE} | "
    f"{len(df)} passenger aircraft variants"
)
