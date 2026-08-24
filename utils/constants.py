# utils/constants.py


# ============================================================
# CONSTANTS
# ============================================================

DATA_FILE = "aircrafts_data.csv"
JET_A_DENSITY_KG_L = 0.80

COMPANY_COLORS = {
    "ATR": "#6C757D",
    "Airbus": "#2F8FC4",
    "Boeing": "#D62728",
    "Embraer": "#2CA02C",
    "de Havilland Canada": "#7B61A8",
    "COMAC": "#F39C12",
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
