import numpy as np
import pandas as pd

# Adjust to generate more data
N_SNAPSHOTS = 10     # e.g. 1440 for 24 h @1 m resolution
ROWS_PER_SNAPSHOT = 100  # e.g. 10000

def make_timestamps():
    base = pd.Timestamp("2025-01-01")
    return [base + pd.Timedelta(minutes=i) for i in range(N_SNAPSHOTS)]

def sample_bus_data():
    ts = make_timestamps()
    rows = []
    for snap_id, t in enumerate(ts):
        for _ in range(ROWS_PER_SNAPSHOT):
            rows.append({
                "BusID": np.random.randint(1, 500),
                "BusType": np.random.choice(["PQ","PV","Slack"]),
                "NominalVoltage_kV": np.random.choice([69,138,240]),
                "LoadMW": np.random.uniform(0, 100),
                "LoadMVAr": np.random.uniform(0, 50),
                "GenMW": np.random.uniform(0, 80),
                "GenMVAr": np.random.uniform(0, 40),
                "AreaID": np.random.randint(1,10),
                "ZoneID": np.random.randint(1,20),
                "Latitude": np.random.uniform(50, 55),
                "Longitude": np.random.uniform(-115, -110),
                "Timestamp": t
            })
    return pd.DataFrame(rows)

def sample_line_data():
    return pd.DataFrame([{
        "LineID": i,
        "FromBusID": np.random.randint(1,500),
        "ToBusID": np.random.randint(1,500),
        "R_pu": np.random.uniform(0.001,0.01),
        "X_pu": np.random.uniform(0.01,0.1),
        "B_shunt_pu": np.random.uniform(0,0.001),
        "ThermalLimit_MW": np.random.uniform(50,200),
        "Status": np.random.choice([0,1])
    } for i in range(1, 1001)])

def sample_switch_data():
    return pd.DataFrame([{
        "SwitchID": i,
        "ConnectedLineID": np.random.randint(1,1001),
        "Status": np.random.choice([0,1]),
        "Timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=np.random.randint(0,N_SNAPSHOTS))
    } for i in range(1,501)])

def sample_measurements():
    ts = make_timestamps()
    rows=[]
    for t in ts:
        for _ in range(ROWS_PER_SNAPSHOT//10):
            rows.append({
                "Timestamp": t,
                "BusID": np.random.randint(1,500),
                "Voltage_pu": np.random.uniform(0.95,1.05),
                "Angle_deg": np.random.uniform(-180,180),
                "LineID": np.random.randint(1,1001),
                "PowerFlow_MW": np.random.uniform(-100,100),
                "PowerFlow_MVAr": np.random.uniform(-50,50),
                "PMU_Flag": np.random.choice([0,1], p=[0.9,0.1])
            })
    return pd.DataFrame(rows)

def sample_contingencies():
    types = ["N-1_LineOutage","N-1_GenTrip","N-2"]
    return pd.DataFrame([{
        "EventID": i,
        "Timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=np.random.randint(0,N_SNAPSHOTS)),
        "Type": np.random.choice(types),
        "ElementID": np.random.randint(1,1001)
    } for i in range(1,1001)])

def sample_severity():
    return pd.DataFrame([{
        "EventID": i,
        "MaxOverload_pct": np.random.uniform(100,200),
        "VoltageViolation_MaxDev_pu": np.random.uniform(0,0.1),
        "EstimatedVoLL_CAD": np.random.uniform(10000,100000),
        "SeverityRank": np.random.randint(1,11)
    } for i in range(1,1001)])

def sample_topology_meta():
    return pd.DataFrame([{
        "SnapshotID": i,
        "NumIslands": np.random.randint(1,5),
        "JacobianRank": np.random.randint(400,450),
        "ObservableFlag": np.random.choice([0,1], p=[0.1,0.9]),
        "NodalObservability_pct": np.random.uniform(80,100)
    } for i in range(N_SNAPSHOTS)])

def sample_weather():
    ts = make_timestamps()
    return pd.DataFrame([{
        "Timestamp": t,
        "Temperature_C": np.random.uniform(-30,30),
        "WindSpeed_mps": np.random.uniform(0,15),
        "SolarIrradiance_W/m2": np.random.uniform(0,800)
    } for t in ts])

def sample_area_meta():
    return pd.DataFrame([{
        "AreaID": i,
        "HistoricalOutagesCount": np.random.randint(0,100),
        "RegulatoryWeight": np.random.uniform(0.5,2.0)
    } for i in range(1,11)])

def column_definitions():
    cols = {
        # sheet: [(col, desc, units, dtype, range), ...]
        "BusData": [
            ("BusID","Unique bus identifier","int","int","1–∞"),
            ("BusType","Type of bus","","category","PQ/PV/Slack"),
            # … add the rest similarly …
        ]
    }
    rows = []
    for sheet, metas in cols.items():
        for c,d,u,t,r in metas:
            rows.append({
                "Sheet": sheet,
                "ColumnName": c,
                "Description": d,
                "Units": u,
                "DataType": t,
                "ValidRange": r
            })
    return pd.DataFrame(rows)

# --- assemble and write ---
with pd.ExcelWriter("SyntheticGridContingencies.xlsx", engine="openpyxl") as w:
    sample_bus_data().to_excel(w, sheet_name="BusData", index=False)
    sample_line_data().to_excel(w, sheet_name="LineData", index=False)
    sample_switch_data().to_excel(w, sheet_name="SwitchData", index=False)
    sample_measurements().to_excel(w, sheet_name="Measurements", index=False)
    sample_contingencies().to_excel(w, sheet_name="ContingencyEvents", index=False)
    sample_severity().to_excel(w, sheet_name="SeverityLabels", index=False)
    sample_topology_meta().to_excel(w, sheet_name="TopologyMeta", index=False)
    sample_weather().to_excel(w, sheet_name="WeatherTime", index=False)
    sample_area_meta().to_excel(w, sheet_name="AreaMeta", index=False)
    column_definitions().to_excel(w, sheet_name="ColumnDefinitions", index=False)

print("Workbook written: SyntheticGridContingencies.xlsx") 