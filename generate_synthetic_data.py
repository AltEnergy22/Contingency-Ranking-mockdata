import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from tqdm import tqdm

# Constants for data generation
N_SNAPSHOTS = 10080  # 7 days * 24 hours * 60 minutes
ROWS_PER_SNAPSHOT = 100  # Number of buses
N_LINES = 150  # Number of transmission lines
N_SWITCHES = 50  # Number of switches
N_MEASUREMENTS = 500  # Number of measurements
N_EVENTS = 100  # Number of contingency events
N_AREAS = 10  # Number of areas
N_WEATHER = 10080  # 7 days * 24 hours * 60 minutes
N_BUSES = 100  # Number of buses

# Base timestamp for data generation
BASE_TIMESTAMP = pd.Timestamp("2025-01-01")

def make_timestamps():
    return [BASE_TIMESTAMP + pd.Timedelta(minutes=i) for i in range(N_SNAPSHOTS)]

def get_diurnal_load_factor(timestamp):
    """Calculate diurnal load factor based on time of day"""
    hour = timestamp.hour
    # Base load curve: higher during day, peak in evening, lowest at night
    base = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)  # 6am to 6pm cycle
    return base

def get_seasonal_factor(timestamp):
    """Calculate seasonal factor (higher in winter, lower in summer)"""
    day_of_year = timestamp.dayofyear
    # Seasonal cycle: peak in winter (day 1), trough in summer (day 182)
    return 1.0 + 0.2 * np.cos((day_of_year - 1) * 2 * np.pi / 365)

def get_solar_generation(timestamp, latitude):
    """Calculate solar generation based on time and location"""
    hour = timestamp.hour
    # Simple solar curve: zero at night, peak at noon
    solar_factor = max(0, np.sin((hour - 6) * np.pi / 12))
    # Adjust for latitude (higher latitude = lower solar potential)
    lat_factor = 1 - abs(latitude - 40) / 90  # 40° is reference latitude
    return solar_factor * lat_factor

def get_wind_generation(timestamp):
    """Calculate wind generation with diurnal pattern and gusts"""
    hour = timestamp.hour
    # Base wind pattern: higher at night, lower during day
    base = 0.5 + 0.3 * np.sin((hour - 18) * np.pi / 12)  # Peak at 6pm
    # Add random gusts
    gust = np.random.normal(0, 0.1)
    return max(0, min(1, base + gust))

def get_customer_mix(area_id):
    """Generate realistic customer mix based on area"""
    # Different areas have different customer compositions
    mixes = {
        1: {"residential": 0.7, "commercial": 0.2, "industrial": 0.1},  # Urban residential
        2: {"residential": 0.3, "commercial": 0.6, "industrial": 0.1},  # Commercial district
        3: {"residential": 0.2, "commercial": 0.3, "industrial": 0.5},  # Industrial area
        4: {"residential": 0.8, "commercial": 0.15, "industrial": 0.05},  # Suburban
        5: {"residential": 0.4, "commercial": 0.4, "industrial": 0.2},  # Mixed
    }
    return mixes.get(area_id, {"residential": 0.5, "commercial": 0.3, "industrial": 0.2})

def sample_bus_data():
    ts = make_timestamps()
    rows = []
    for snap_id, t in enumerate(ts):
        for bus_id in range(1, N_BUSES + 1):
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "BusID": bus_id,
                "Voltage_kV": np.random.uniform(115, 500),
                "Load_MW": np.random.uniform(0, 1000),
                "Generation_MW": np.random.uniform(0, 2000),
                "Status": np.random.choice([0, 1], p=[0.1, 0.9]),
                "CriticalityLevel": np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]),
                "Latitude": np.random.uniform(50, 55),
                "Longitude": np.random.uniform(-115, -110),
                "Zone": np.random.randint(1, 21)
            })
    return pd.DataFrame(rows)

def generate_line_name(from_bus, to_bus, voltage):
    """Generate realistic line names based on bus IDs and voltage"""
    cities = {
        1: "Alb", 2: "Edm", 3: "Cal", 4: "Van", 5: "Tor",
        6: "Mon", 7: "Ott", 8: "Hal", 9: "Win", 10: "Reg"
    }
    from_city = cities.get(from_bus % 10 + 1, f"Bus{from_bus}")
    to_city = cities.get(to_bus % 10 + 1, f"Bus{to_bus}")
    return f"{from_city}–{to_city} {voltage} kV Corridor"

def get_protection_scheme(criticality, length):
    """Determine protection scheme based on line criticality and length"""
    if criticality == 3:  # Transmission spine
        return "SPS-Backed"
    elif length > 100:  # Long lines
        return "AutoRecloser"
    else:
        return "Relays Only"

def get_line_failure_probability(length, age, criticality):
    """Calculate failure probability based on line characteristics"""
    # Base failure rate (per 100 km per year)
    base_rate = 0.5  # 0.5 failures per 100 km per year
    
    # Age factor (increases with age)
    age_factor = 1.0 + (age / 50)  # 50% increase over 50 years
    
    # Criticality factor (higher for more critical lines)
    criticality_factor = 1.0 + (criticality - 1) * 0.2  # 20% increase per criticality level
    
    # Calculate final probability
    probability = (base_rate * length / 100) * age_factor * criticality_factor
    
    return min(10.0, probability)  # Cap at 10% per year

def sample_line_data():
    ts = make_timestamps()
    rows = []
    for snap_id, t in enumerate(ts):
        for line_id in range(1, N_LINES + 1):
            from_bus = np.random.randint(1, N_BUSES + 1)
            to_bus = np.random.randint(1, N_BUSES + 1)
            while to_bus == from_bus:
                to_bus = np.random.randint(1, N_BUSES + 1)
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "LineID": line_id,
                "FromBusID": from_bus,
                "ToBusID": to_bus,
                "R_pu": round(np.random.uniform(0.01, 0.1), 4),
                "X_pu": round(np.random.uniform(0.1, 1.0), 4),
                "B_pu": round(np.random.uniform(0.1, 10.0), 4),
                "ThermalLimit_MW": round(np.random.uniform(100, 2000), 1),
                "CriticalityLevel": np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]),
                "Status": np.random.choice([0, 1], p=[0.1, 0.9]),
                "FailureRate_pct": round(np.random.uniform(0.1, 5.0), 2),
                "Length_km": np.random.uniform(10, 200),
                "Age_years": np.random.uniform(0, 50),
                "FromVoltage_pu": np.random.uniform(0.95, 1.05),
                "ToVoltage_pu": np.random.uniform(0.95, 1.05),
                "PowerFlow_MW": round(np.random.uniform(0, 1000), 1),
                "ReactivePower_MVAR": round(np.random.uniform(0, 500), 1)
            })
    return pd.DataFrame(rows)

def generate_switch_name(line_id, voltage, switch_type):
    """Generate realistic switch names based on line and type"""
    directions = ["North", "South", "East", "West"]
    cities = {
        1: "Edm", 2: "Cal", 3: "Van", 4: "Tor", 5: "Mon",
        6: "Ott", 7: "Hal", 8: "Win", 9: "Reg", 10: "Alb"
    }
    city = cities.get(line_id % 10 + 1, f"Bus{line_id}")
    direction = directions[line_id % 4]
    return f"{city}-{direction} {voltage} kV {switch_type}"

def get_switch_type(criticality, voltage):
    """Determine switch type based on criticality and voltage"""
    if voltage >= 240:
        return "Breakers"
    elif criticality == 3:  # Transmission spine
        return "Breakers"
    elif voltage >= 138:
        return np.random.choice(["Breakers", "Disconnect"], p=[0.7, 0.3])
    else:
        return np.random.choice(["Breakers", "Disconnect", "Recloser", "LoadBreaker"], 
                              p=[0.4, 0.3, 0.2, 0.1])

def get_failure_probability(switch_type, operation_count, age):
    """Calculate per-operation failure probability based on switch characteristics"""
    base_probs = {
        "Breakers": 0.01,      # 1% base failure rate
        "Disconnect": 0.005,   # 0.5% base failure rate
        "Recloser": 0.015,     # 1.5% base failure rate
        "LoadBreaker": 0.02    # 2% base failure rate
    }
    
    base_prob = base_probs[switch_type]
    age_factor = 1 + (age / 20)  # Older switches have higher probability
    operation_factor = 1 + (operation_count / 1000)  # More operations = higher probability
    
    return base_prob * age_factor * operation_factor

def get_next_test_date(base_time, switch_type, last_test=None):
    """Calculate next scheduled test date based on switch type and history"""
    if last_test is None:
        last_test = base_time - pd.Timedelta(days=np.random.randint(0, 365))
    
    test_intervals = {
        "Breakers": 365,      # Annual testing
        "Disconnect": 730,    # Bi-annual testing
        "Recloser": 180,      # Quarterly testing
        "LoadBreaker": 365    # Annual testing
    }
    
    interval = test_intervals[switch_type]
    next_test = last_test + pd.Timedelta(days=interval)
    
    # Add some randomness to the test date (±15 days)
    next_test += pd.Timedelta(days=np.random.randint(-15, 16))
    
    return next_test

def generate_bus_pairs():
    """Generate list of connected bus pairs from line data"""
    bus_pairs = []
    # Generate initial bus pairs for the first snapshot
    for i in range(1, ROWS_PER_SNAPSHOT + 1):
        # Each bus connects to 2-4 other buses
        n_connections = np.random.randint(2, 5)
        for _ in range(n_connections):
            to_bus = np.random.randint(1, ROWS_PER_SNAPSHOT + 1)
            if to_bus != i and (i, to_bus) not in bus_pairs and (to_bus, i) not in bus_pairs:
                bus_pairs.append((i, to_bus))
    return bus_pairs

def sample_switch_data():
    """Generate synthetic switch data"""
    ts = make_timestamps()
    rows = []
    
    # Generate bus pairs for network connectivity
    bus_pairs = generate_bus_pairs()
    
    # Pre-generate switch characteristics for consistency
    n_switches = N_SWITCHES
    switch_chars = {}
    
    for i in range(1, n_switches + 1):
        # Get line information
        line_id = np.random.randint(1, N_LINES + 1)
        
        # Generate switch characteristics
        voltage = np.random.choice([69, 138, 240])
        criticality = np.random.randint(1, 6)
        switch_type = get_switch_type(criticality, voltage)
        age = np.random.uniform(0, 30)  # Switch age in years
        operation_count = int(np.random.uniform(100, 5000))  # Lifetime operations
        
        switch_chars[i] = {
            "line_id": line_id,
            "voltage": voltage,
            "type": switch_type,
            "age": age,
            "operation_count": operation_count,
            "last_test": None
        }
    
    for snap_id, t in enumerate(ts):
        for switch_id in range(1, n_switches + 1):
            chars = switch_chars[switch_id]
            
            # Calculate failure probability
            failure_prob = get_failure_probability(
                chars["type"], 
                chars["operation_count"], 
                chars["age"]
            )
            
            # Determine next test date
            if chars["last_test"] is None:
                chars["last_test"] = t - pd.Timedelta(days=np.random.randint(0, 365))
            next_test = get_next_test_date(t, chars["type"], chars["last_test"])
            
            # Determine control source and operation reason
            control_source = np.random.choice(
                ["SCADA", "LocalProtection", "Manual"],
                p=[0.7, 0.2, 0.1]
            )
            
            last_operation_reason = np.random.choice(
                ["FaultIsolation", "LoadShedding", "Maintenance", "OperatorCommand"],
                p=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Determine status and overrides
            status = 1  # Default to closed
            manual_override = 0
            auto_reclose = 0
            
            if chars["type"] == "Recloser":
                auto_reclose = np.random.choice([0, 1], p=[0.2, 0.8])
            
            # Randomly set manual override (rare)
            if np.random.random() < 0.001:  # 0.1% chance
                manual_override = 1
                control_source = "Manual"
            
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "SwitchID": switch_id,
                "ConnectedLineID": chars["line_id"],
                "SwitchName": generate_switch_name(chars["line_id"], chars["voltage"], chars["type"]),
                "SwitchType": chars["type"],
                "Status": status,
                "OperationCount": chars["operation_count"],
                "NextScheduledTest": next_test,
                "ManualOverrideFlag": manual_override,
                "AutoRecloseEnabled": auto_reclose,
                "FailureProbability_pct": failure_prob * 100,  # Convert to percentage
                "ControlSource": control_source,
                "LastOperationReason": last_operation_reason
            })
    
    return pd.DataFrame(rows)

def generate_device_id(device_type, bus_id):
    """Generate realistic device IDs"""
    return f"{device_type}-Bus{bus_id:03d}"

def get_measurement_quality():
    """Generate measurement quality with realistic distribution"""
    return np.random.choice(
        ["Good", "Suspect", "Bad"],
        p=[0.989, 0.01, 0.001]  # 98.9% good, 1% suspect, 0.1% bad
    )

def get_data_latency(measurement_source):
    """Get realistic data latency based on measurement source"""
    if measurement_source == "PMU":
        return np.random.randint(10, 51)  # 10-50 ms for PMU
    else:
        return np.random.randint(500, 2001)  # 500-2000 ms for SCADA/AMI

def get_measurement_uncertainty(measurement_source, value):
    """Calculate measurement uncertainty based on source and value"""
    if measurement_source == "PMU":
        base_uncertainty = 0.1  # 0.1% for PMU
    else:
        base_uncertainty = 1.0  # 1.0% for SCADA/AMI
    
    # Add some randomness to the uncertainty
    uncertainty = base_uncertainty * (1 + np.random.normal(0, 0.1))
    return max(0.01, min(5.0, uncertainty))  # Cap between 0.01% and 5%

def get_phasor_confidence(voltage_pu, angle_deg):
    """Calculate PMU phasor confidence based on signal quality"""
    # Base confidence on how far the measurement is from nominal
    voltage_deviation = abs(voltage_pu - 1.0)
    angle_deviation = abs(angle_deg)
    
    # Calculate confidence (1.0 = perfect, 0.0 = poor)
    confidence = 1.0 - (voltage_deviation * 10 + angle_deviation / 180)
    return max(0.0, min(1.0, confidence))

def sample_measurements():
    ts = make_timestamps()
    rows = []
    measurement_id = 1
    
    # Pre-generate device locations for consistency
    device_locations = {}
    
    # Generate SCADA measurements (one per bus)
    for bus_id in range(1, ROWS_PER_SNAPSHOT + 1):
        device_locations[bus_id] = {
            "lat": np.random.uniform(50, 55),
            "lon": np.random.uniform(-115, -110)
        }
    
    # Identify PMU and AMI locations
    pmu_buses = set(np.random.choice(
        range(1, ROWS_PER_SNAPSHOT + 1),
        size=int(ROWS_PER_SNAPSHOT * 0.1),  # 10% of buses have PMUs
        replace=False
    ))
    
    ami_buses = set(np.random.choice(
        range(1, ROWS_PER_SNAPSHOT + 1),
        size=int(ROWS_PER_SNAPSHOT * 0.2),  # 20% of buses have AMI
        replace=False
    ))
    
    for snap_id, t in enumerate(ts):
        # Generate SCADA measurements
        for bus_id in range(1, ROWS_PER_SNAPSHOT + 1):
            # Base voltage with noise
            voltage_pu = 1.0 + np.random.normal(0, 0.01)  # SCADA noise
            angle_deg = np.random.normal(0, 1.0)  # SCADA angle noise
            
            # Calculate power flow (simplified DC power flow)
            power_mw = np.random.normal(0, 50)  # Base power flow
            power_mvar = power_mw * np.random.uniform(0.1, 0.3)  # Reactive power
            
            # Add measurement noise
            power_mw += np.random.normal(0, 0.5)  # SCADA power noise
            power_mvar += np.random.normal(0, 0.5)
            
            # Generate measurement
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "MeasurementID": measurement_id,
                "DeviceID": generate_device_id("RTU", bus_id),
                "MeasurementSource": "SCADA",
                "IntervalLength_s": 60,
                "BusID": bus_id,
                "Voltage_pu": voltage_pu,
                "Angle_deg": angle_deg,
                "PowerFlow_MW": power_mw,
                "PowerFlow_MVAr": power_mvar,
                "MeasurementQualityFlag": get_measurement_quality(),
                "MeasurementUncertainty_pct": get_measurement_uncertainty("SCADA", power_mw),
                "DataLatency_ms": get_data_latency("SCADA"),
                "PhasorConfidence": None,  # Not applicable for SCADA
                "GeoCoordinates": f"{device_locations[bus_id]['lat']:.6f},{device_locations[bus_id]['lon']:.6f}"
            })
            measurement_id += 1
        
        # Generate PMU measurements
        for bus_id in pmu_buses:
            # PMU measurements have higher precision
            voltage_pu = 1.0 + np.random.normal(0, 0.001)  # PMU noise
            angle_deg = np.random.normal(0, 0.01)  # PMU angle noise
            
            # Calculate power flow with PMU precision
            power_mw = np.random.normal(0, 50)
            power_mvar = power_mw * np.random.uniform(0.1, 0.3)
            
            # Add PMU measurement noise
            power_mw += np.random.normal(0, 0.05)  # PMU power noise
            power_mvar += np.random.normal(0, 0.05)
            
            # Generate PMU measurement
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t.replace(microsecond=0),  # Align to second
                "MeasurementID": measurement_id,
                "DeviceID": generate_device_id("PMU", bus_id),
                "MeasurementSource": "PMU",
                "IntervalLength_s": 1,
                "BusID": bus_id,
                "Voltage_pu": voltage_pu,
                "Angle_deg": angle_deg,
                "PowerFlow_MW": power_mw,
                "PowerFlow_MVAr": power_mvar,
                "MeasurementQualityFlag": get_measurement_quality(),
                "MeasurementUncertainty_pct": get_measurement_uncertainty("PMU", power_mw),
                "DataLatency_ms": get_data_latency("PMU"),
                "PhasorConfidence": get_phasor_confidence(voltage_pu, angle_deg),
                "GeoCoordinates": f"{device_locations[bus_id]['lat']:.6f},{device_locations[bus_id]['lon']:.6f}"
            })
            measurement_id += 1
        
        # Generate AMI measurements
        for bus_id in ami_buses:
            # AMI measurements have SCADA-like precision
            voltage_pu = 1.0 + np.random.normal(0, 0.01)  # AMI noise
            angle_deg = np.random.normal(0, 1.0)  # AMI angle noise
            
            # Calculate power flow
            power_mw = np.random.normal(0, 50)
            power_mvar = power_mw * np.random.uniform(0.1, 0.3)
            
            # Add AMI measurement noise
            power_mw += np.random.normal(0, 0.5)  # AMI power noise
            power_mvar += np.random.normal(0, 0.5)
            
            # Generate AMI measurement
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "MeasurementID": measurement_id,
                "DeviceID": generate_device_id("AMI", bus_id),
                "MeasurementSource": "AMI",
                "IntervalLength_s": 60,
                "BusID": bus_id,
                "Voltage_pu": voltage_pu,
                "Angle_deg": angle_deg,
                "PowerFlow_MW": power_mw,
                "PowerFlow_MVAr": power_mvar,
                "MeasurementQualityFlag": get_measurement_quality(),
                "MeasurementUncertainty_pct": get_measurement_uncertainty("AMI", power_mw),
                "DataLatency_ms": get_data_latency("AMI"),
                "PhasorConfidence": None,  # Not applicable for AMI
                "GeoCoordinates": f"{device_locations[bus_id]['lat']:.6f},{device_locations[bus_id]['lon']:.6f}"
            })
            measurement_id += 1
    
    return pd.DataFrame(rows)

def get_element_failure_rate(element_type, criticality, age=None):
    """Calculate annualized failure rate based on element characteristics"""
    base_rates = {
        "Line": 0.5,      # 0.5 failures per year
        "Generator": 2.0,  # 2.0 failures per year
        "Switch": 1.0     # 1.0 failures per year
    }
    
    base_rate = base_rates[element_type]
    criticality_factor = 1 + (criticality * 0.2)  # Higher criticality = higher rate
    
    if age is not None:
        age_factor = 1 + (age / 20)  # Older elements have higher rates
    else:
        age_factor = 1.0
    
    return base_rate * criticality_factor * age_factor

def get_contingency_type(element_type, is_sps=False):
    """Determine contingency type based on element type and SPS status"""
    if is_sps:
        return "SPS_Trigger"
    
    if element_type == "Line":
        return np.random.choice(["N-1_LineOutage", "N-2"], p=[0.9, 0.1])
    elif element_type == "Generator":
        return "N-1_GenTrip"
    else:
        return "N-1_LineOutage"  # Default for switches

def get_detection_method(element_type, criticality, has_pmu):
    """Determine how the contingency is detected"""
    if criticality == 3:
        return "PMU" if has_pmu else "SCADA"
    elif criticality == 2:
        return np.random.choice(["SCADA", "PMU", "OperatorReport"], p=[0.6, 0.3, 0.1])
    else:
        return np.random.choice(["SCADA", "OperatorReport"], p=[0.8, 0.2])

def get_simulated_duration(element_type, is_sps=False):
    """Calculate simulated duration until restoration"""
    if is_sps:
        return np.random.randint(1, 6)  # 1-5 seconds for SPS
    
    if element_type == "Generator":
        return np.random.randint(300, 3600)  # 5-60 minutes
    elif element_type == "Line":
        return np.random.randint(60, 300)  # 1-5 minutes
    else:  # Switch
        return np.random.randint(30, 180)  # 30 seconds to 3 minutes

def sample_contingencies():
    """Generate synthetic contingency events"""
    ts = make_timestamps()
    rows = []
    event_id = 1
    
    # Generate events for each snapshot
    for snap_id, t in enumerate(ts):
        # Determine number of events for this snapshot (0-3)
        n_events = np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.08, 0.02])
        
        for _ in range(n_events):
            # Select element type and specific element
            element_type = np.random.choice(["Line", "Generator", "Load", "Bus"])
            
            if element_type == "Line":
                # Select a random line
                line_idx = np.random.randint(0, len(line_data))
                element = {
                    "id": line_data.iloc[line_idx]["LineID"],
                    "criticality": line_data.iloc[line_idx]["CriticalityLevel"],
                    "from_bus": line_data.iloc[line_idx]["FromBusID"],
                    "to_bus": line_data.iloc[line_idx]["ToBusID"]
                }
            elif element_type == "Generator":
                # Select a random bus with generation
                gen_buses = bus_data[bus_data["GenMW"] > 0]  # Fixed column name
                if len(gen_buses) > 0:
                    bus_idx = np.random.randint(0, len(gen_buses))
                    element = {
                        "id": f"GEN-{gen_buses.iloc[bus_idx]['BusID']}",
                        "criticality": gen_buses.iloc[bus_idx]["CriticalityLevel"],
                        "bus": gen_buses.iloc[bus_idx]["BusID"]
                    }
            elif element_type == "Load":
                # Select a random bus with load
                load_buses = bus_data[bus_data["LoadMW"] > 0]  # Fixed column name
                if len(load_buses) > 0:
                    bus_idx = np.random.randint(0, len(load_buses))
                    element = {
                        "id": f"LOAD-{load_buses.iloc[bus_idx]['BusID']}",
                        "criticality": load_buses.iloc[bus_idx]["CriticalityLevel"],
                        "bus": load_buses.iloc[bus_idx]["BusID"]
                    }
            else:  # Bus
                # Select a random bus
                bus_idx = np.random.randint(0, len(bus_data))
                element = {
                    "id": bus_data.iloc[bus_idx]["BusID"],
                    "criticality": bus_data.iloc[bus_idx]["CriticalityLevel"]
                }
            
            # Determine if this is an SPS event
            is_sps = np.random.random() < 0.1  # 10% chance of SPS event
            
            # Get contingency type
            contingency_type = get_contingency_type(element_type, is_sps)
            
            # Get detection method
            has_pmu = np.random.random() < 0.3  # 30% chance of PMU presence
            detection_method = get_detection_method(element_type, element["criticality"], has_pmu)
            
            # Get simulated duration
            duration = get_simulated_duration(element_type, is_sps)
            
            # Get weather impact
            weather_id = np.random.randint(1, N_WEATHER + 1)
            
            # Determine SPS action
            sps_action = "None"
            if is_sps:
                sps_action = np.random.choice(
                    ["LoadShed", "GenerationTrip", "LineTrip"],
                    p=[0.5, 0.3, 0.2]
                )
            
            rows.append({
                "SnapshotID": snap_id,
                "EventID": event_id,
                "Type": element_type,
                "ElementType": "Transmission" if element_type == "Line" else "Distribution",
                "ElementID": element["id"],
                "StartTime": t,
                "EndTime": t + pd.Timedelta(minutes=duration),
                "DetectionMethod": detection_method,
                "WeatherID": weather_id,
                "SPSAction": sps_action
            })
            
            event_id += 1
    
    return pd.DataFrame(rows)

def get_severity_rank():
    """Generate severity rank with specified distribution"""
    return np.random.choice(
        range(1, 11),
        p=[0.1, 0.1, 0.1,  # Mild (1-3): 30%
           0.1, 0.1, 0.1, 0.1,  # Moderate (4-7): 40%
           0.1, 0.1, 0.1]  # Severe (8-10): 30%
    )

def get_severity_metrics(rank):
    """Generate severity metrics based on rank"""
    if rank == 1:  # Mild
        return {
            "overload_pct": np.random.uniform(0, 20),
            "voltage_dev": np.random.uniform(0, 0.05),
            "cascading_prob": np.random.uniform(0, 10),
            "restoration_time": np.random.uniform(5, 30)
        }
    elif rank == 2:  # Moderate
        return {
            "overload_pct": np.random.uniform(20, 50),
            "voltage_dev": np.random.uniform(0.05, 0.1),
            "cascading_prob": np.random.uniform(10, 30),
            "restoration_time": np.random.uniform(30, 120)
        }
    else:  # Severe
        return {
            "overload_pct": np.random.uniform(50, 100),
            "voltage_dev": np.random.uniform(0.1, 0.2),
            "cascading_prob": np.random.uniform(30, 50),
            "restoration_time": np.random.uniform(120, 240)
        }

def get_voll(rank):
    """Calculate Value of Lost Load with noise"""
    base_voll = 10000 * rank  # $10,000 per rank point
    noise = np.random.normal(0, 0.1)  # ±10% noise
    return base_voll * (1 + noise)

def get_economic_impact(voll, restoration_time):
    """Calculate total economic impact including restoration costs"""
    crew_cost_per_hour = 5000
    restoration_cost = crew_cost_per_hour * (restoration_time / 60)  # Convert minutes to hours
    return voll + restoration_cost

def get_weather_impact(weather_id, event_time):
    """Determine if weather conditions exacerbated the event"""
    global weather_data
    
    # Get weather conditions for this snapshot
    weather = weather_data[weather_data["SnapshotID"] == weather_id]
    if len(weather) == 0:
        return "No"
    
    # Check for severe weather conditions
    temp = weather.iloc[0]["Temperature_C"]
    wind_speed = weather.iloc[0]["WindSpeed_mps"]
    precip = weather.iloc[0]["Precipitation_mm"]
    
    # Determine if weather conditions are severe
    is_severe = (
        (temp < -20 or temp > 35) or  # Extreme temperatures
        (wind_speed > 20) or          # High winds
        (precip > 10)                 # Heavy precipitation
    )
    
    return "Yes" if is_severe else "No"

def get_load_shedding(rank, total_load):
    """Calculate load shedding amount if significant"""
    if rank >= 8:  # Severe events
        shedding_pct = np.random.uniform(0.1, 0.3)  # 10-30% of total load
        return total_load * shedding_pct
    elif rank >= 4:  # Moderate events
        if np.random.random() < 0.3:  # 30% chance of shedding
            shedding_pct = np.random.uniform(0.05, 0.15)  # 5-15% of total load
            return total_load * shedding_pct
    return 0

def sample_severity():
    """Sample severity labels for contingencies"""
    global contingency_events, line_data, bus_data, generator_data, transformer_data
    
    # Create severity labels DataFrame
    severity_labels = pd.DataFrame(columns=[
        "ContingencyID",
        "SeverityScore",
        "CriticalityLevel",
        "ImpactDescription"
    ])
    
    # Sample severity for each contingency
    for _, event in tqdm(contingency_events.iterrows(), total=len(contingency_events), desc="Severity Labels"):
        # Calculate base severity score
        base_score = 0
        
        # Add impact based on component type
        if event["ElementType"] == "Line":
            line = line_data[line_data["LineID"] == event["ElementID"]].iloc[0]
            base_score += line["ThermalLimit_MW"] / 100  # Normalize by 100 MW
            base_score += line["CriticalityLevel"] * 0.5  # Add criticality impact
        elif event["ElementType"] == "Bus":
            bus = bus_data[bus_data["BusID"] == event["ElementID"]].iloc[0]
            base_score += bus["Load_MW"] / 100  # Normalize by 100 MW
            base_score += 0.3  # Base impact for bus contingencies
        
        # Add random variation
        severity_score = base_score * np.random.uniform(0.8, 1.2)
        
        # Determine criticality level based on severity score
        if severity_score < 0.5:
            rank = 1  # Low
        elif severity_score < 1.0:
            rank = 2  # Medium
        else:
            rank = 3  # High
        
        # Generate impact description
        impact_desc = f"Impact level {rank} with severity score {severity_score:.2f}"
        
        # Add to DataFrame
        severity_labels = pd.concat([severity_labels, pd.DataFrame([{
            "ContingencyID": event["EventID"],
            "SeverityScore": severity_score,
            "CriticalityLevel": rank,
            "ImpactDescription": impact_desc
        }])], ignore_index=True)
    
    return severity_labels

def find_islands(bus_pairs, line_status):
    """Find all islands in the network using DFS"""
    def dfs(node, visited, island):
        visited.add(node)
        island.add(node)
        
        # Find all connected buses
        for pair in bus_pairs:
            if pair[0] == node:
                # Check if the line exists and is closed
                reverse_pair = (pair[1], pair[0])
                if (pair in line_status and line_status[pair] == 1) or \
                   (reverse_pair in line_status and line_status[reverse_pair] == 1):
                    if pair[1] not in visited:
                        dfs(pair[1], visited, island)
    
    # Find all unique buses
    all_buses = set()
    for pair in bus_pairs:
        all_buses.add(pair[0])
        all_buses.add(pair[1])
    
    # Find islands
    visited = set()
    islands = []
    for bus in all_buses:
        if bus not in visited:
            island = set()
            dfs(bus, visited, island)
            islands.append(island)
    
    return islands

def calculate_min_cut_size(bus_pairs, line_status, num_islands):
    """Calculate minimum cut size between islands"""
    if num_islands <= 1:
        return 0
    
    # Count lines that connect different islands
    cut_lines = 0
    for pair in bus_pairs:
        if pair in line_status and line_status[pair] == 1:
            # Check if this line connects different islands
            from_island = None
            to_island = None
            for i, island in enumerate(find_islands(bus_pairs, line_status)):
                if pair[0] in island:
                    from_island = i
                if pair[1] in island:
                    to_island = i
                if from_island is not None and to_island is not None:
                    break
            
            if from_island != to_island:
                cut_lines += 1
    
    return cut_lines

def get_line_loading(line_id, power_flow, thermal_limit):
    """Calculate line loading percentage"""
    if thermal_limit > 0:
        return abs(power_flow) / thermal_limit * 100
    return 0

def get_voltage_violations(bus_voltages):
    """Count buses with voltage violations"""
    return sum(1 for v in bus_voltages if abs(v - 1.0) > 0.05)

def get_overloads(line_loadings):
    """Count overloaded lines"""
    return sum(1 for loading in line_loadings if loading > 100)

def sample_topology_meta():
    """Generate synthetic topology metadata"""
    ts = make_timestamps()
    rows = []
    
    for snap_id, t in enumerate(ts):
        # Get line statuses
        line_status = {
            (row["FromBusID"], row["ToBusID"]): row["Status"]
            for _, row in line_data.iterrows()
        }
        
        # Find islands
        bus_pairs = generate_bus_pairs()
        islands = find_islands(bus_pairs, line_status)
        num_islands = len(islands)
        
        # Calculate minimum cut size
        min_cut = calculate_min_cut_size(bus_pairs, line_status, num_islands)
        
        # Get line loadings
        line_loadings = {
            row["LineID"]: get_line_loading(
                row["LineID"],
                np.random.uniform(0, row["ThermalLimit_MW"]),
                row["ThermalLimit_MW"]
            )
            for _, row in line_data.iterrows()
        }
        
        # Get voltage violations
        bus_voltages = {
            row["BusID"]: np.random.uniform(0.95, 1.05)
            for _, row in bus_data.iterrows()
        }
        voltage_violations = get_voltage_violations(bus_voltages)
        
        # Get overloads
        overloads = get_overloads(line_loadings)
        
        # Check for bad data
        bad_data = 1 if np.random.random() < 0.01 else 0  # 1% chance of bad data
        
        rows.append({
            "SnapshotID": snap_id,
            "Timestamp": t,
            "NumberOfIslands": num_islands,
            "MinimumCutSize": min_cut,
            "MaxLineLoading_pct": max(line_loadings.values()) if line_loadings else 0,
            "VoltageViolations": voltage_violations,
            "OverloadedLines": overloads,
            "BadDataFlag": bad_data
        })
    
    return pd.DataFrame(rows)

def calculate_daily_mean_temp(day_of_year):
    # Example: Simple sinusoidal model for daily mean temperature
    # Adjust parameters as needed for your use case
    base_temp = 15  # Base temperature in Celsius
    amplitude = 10  # Temperature variation amplitude
    phase_shift = 180  # Phase shift to align with summer/winter
    return base_temp + amplitude * np.sin(2 * np.pi * (day_of_year + phase_shift) / 365)

def get_solar_noon_offset(latitude=45):
    """Calculate solar noon offset based on latitude and day of year"""
    # Simplified solar position calculation
    day_of_year = 1  # Start with January 1st
    declination = -23.45 * np.cos(2 * np.pi * (day_of_year + 10) / 365)
    return 12 + (4 * latitude) / 60 + (declination * np.sin(2 * np.pi * latitude / 360))

def get_cloud_cover():
    """Generate realistic cloud cover percentage"""
    return np.random.uniform(10, 30)

def get_weather_condition(temp, wind_speed, precip, humidity, visibility):
    """Determine weather condition based on parameters"""
    if precip > 0.5:
        return "Rain" if temp > 0 else "Snow"
    elif visibility < 1:
        return "Fog"
    elif wind_speed > 15:
        return "Windy"
    elif humidity > 80:
        return "Humid"
    elif np.random.random() < 0.3:  # 30% chance of clouds
        return "Cloudy"
    else:
        return "Clear"

def calculate_wind_power(wind_speed, wind_direction, turbine_params):
    """Calculate wind power generation based on wind conditions"""
    # Extract turbine parameters
    cut_in = turbine_params["cut_in"]  # Minimum wind speed for generation
    rated = turbine_params["rated"]    # Wind speed at rated power
    cut_out = turbine_params["cut_out"]  # Maximum wind speed before shutdown
    capacity = turbine_params["capacity"]  # Maximum power output
    
    # Calculate power based on wind speed
    if wind_speed < cut_in:
        power = 0
    elif wind_speed > cut_out:
        power = 0
    elif wind_speed < rated:
        # Cubic relationship between wind speed and power up to rated speed
        power = capacity * ((wind_speed - cut_in) / (rated - cut_in)) ** 3
    else:
        power = capacity
    
    # Add some noise to the power output (±5%)
    power *= (1 + np.random.normal(0, 0.05))
    
    return max(0, min(capacity, power))

def calculate_solar_power(irradiance, temperature, panel_params):
    """Calculate solar power generation based on irradiance and temperature"""
    # Extract panel parameters
    efficiency = panel_params["efficiency"]  # Panel efficiency
    temp_coeff = panel_params["temp_coeff"]  # Temperature coefficient
    capacity = panel_params["capacity"]      # Maximum power output
    
    # Standard test conditions
    stc_irradiance = 1000  # W/m²
    stc_temp = 25         # °C
    
    # Calculate temperature factor (power decreases as temperature increases)
    temp_factor = 1 + temp_coeff * (temperature - stc_temp)
    
    # Calculate power output
    power = (irradiance / stc_irradiance) * capacity * efficiency * temp_factor
    
    # Add some noise to the power output (±3%)
    power *= (1 + np.random.normal(0, 0.03))
    
    return max(0, min(capacity, power))

def calculate_line_failure_probability(line, weather):
    """Calculate line failure probability based on line characteristics and weather"""
    # Get base failure rate from line data
    base_probability = line["FailureRate_pct"] / 100  # Convert from percentage to decimal
    
    # Weather impact factors
    temp_factor = 1.0
    if weather["temp"] < -20:  # Extreme cold
        temp_factor = 1.5
    elif weather["temp"] > 35:  # Extreme heat
        temp_factor = 1.3
    
    wind_factor = 1.0
    if weather["wind_speed"] > 20:  # High winds
        wind_factor = 2.0
    elif weather["wind_speed"] > 15:  # Moderate winds
        wind_factor = 1.5
    
    precip_factor = 1.0
    if weather["precip"] > 10:  # Heavy precipitation
        precip_factor = 1.4
    elif weather["precip"] > 5:  # Moderate precipitation
        precip_factor = 1.2
    
    # Calculate final probability
    final_probability = base_probability * temp_factor * wind_factor * precip_factor
    
    # Add some random variation (±20%)
    final_probability *= (1 + np.random.normal(0, 0.2))
    
    return min(0.1, max(0, final_probability))  # Cap between 0 and 10%

def sample_weather():
    """Generate synthetic weather data"""
    ts = make_timestamps()
    rows = []
    
    for snap_id, t in enumerate(ts):
        # Get daily mean temperature
        day_of_year = t.dayofyear
        daily_mean = calculate_daily_mean_temp(day_of_year)
        
        # Add diurnal variation
        hour = t.hour
        diurnal_variation = 5 * np.sin((hour - 6) * np.pi / 12)  # 5°C variation
        temp = daily_mean + diurnal_variation + np.random.normal(0, 1)
        
        # Generate other weather parameters
        wind_speed = np.random.uniform(0, 30)  # 0-30 m/s
        wind_direction = np.random.randint(0, 360)  # 0-359 degrees
        solar_noon = 12 + get_solar_noon_offset()
        solar_angle = abs(hour - solar_noon) * 15  # 15 degrees per hour
        cloud_cover = get_cloud_cover()
        irradiance = 1000 * (1 - cloud_cover) * np.cos(np.radians(solar_angle))
        precip = np.random.exponential(2) if np.random.random() < 0.1 else 0  # 10% chance of rain
        humidity = np.random.uniform(30, 90)  # Random humidity between 30% and 90%
        visibility = np.random.uniform(0, 10)  # 0-10 km
        
        # Get weather condition
        condition = get_weather_condition(temp, wind_speed, precip, humidity, visibility)
        
        # Calculate wind and solar power
        turbine_params = {
            "cut_in": 3,  # m/s
            "rated": 12,  # m/s
            "cut_out": 25,  # m/s
            "capacity": 2000  # MW
        }
        panel_params = {
            "efficiency": 0.15,
            "temp_coeff": -0.004,  # per °C
            "capacity": 1000  # MW
        }
        
        wind_power = calculate_wind_power(wind_speed, wind_direction, turbine_params)
        solar_power = calculate_solar_power(irradiance, temp, panel_params)
        
        # Calculate line failure probabilities
        line_failure_probs = {}
        for _, line in line_data.iterrows():
            line_failure_probs[line["LineID"]] = calculate_line_failure_probability(line, {
                "temp": temp,
                "wind_speed": wind_speed,
                "precip": precip,
                "humidity": humidity,
                "visibility": visibility
            })
        
        rows.append({
            "SnapshotID": snap_id,
            "Timestamp": t,
            "Temperature_C": round(temp, 1),
            "WindSpeed_mps": round(wind_speed, 1),
            "WindDirection_deg": wind_direction,
            "SolarIrradiance_W/m2": round(irradiance, 1),
            "Precipitation_mm": round(precip, 1),
            "Humidity_pct": round(humidity, 1),
            "Visibility_km": round(visibility, 1),
            "WeatherCondition": condition,
            "StormFlag": 1 if wind_speed > 20 or precip > 10 else 0,
            "WindPower_MW": round(wind_power, 1),
            "SolarPower_MW": round(solar_power, 1),
            "LineFailureProbabilities": json.dumps(line_failure_probs)
        })
    
    return pd.DataFrame(rows)

def sample_area_meta():
    return pd.DataFrame([{
        "AreaID": i,
        "HistoricalOutagesCount": np.random.randint(0,100),
        "RegulatoryWeight": np.random.uniform(0.5,2.0)
    } for i in range(1,11)])

def sample_column_definitions():
    """Generate comprehensive column definitions for all sheets"""
    definitions = []
    
    # BusData columns
    definitions.extend([
        {
            "Sheet": "BusData",
            "ColumnName": "SnapshotID",
            "Description": "Unique identifier for each system snapshot",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to N_SNAPSHOTS-1"
        },
        {
            "Sheet": "BusData",
            "ColumnName": "Timestamp",
            "Description": "Exact time of the snapshot",
            "Units": "",
            "DataType": "datetime",
            "ValidRange": "7 days of 1-minute intervals"
        },
        {
            "Sheet": "BusData",
            "ColumnName": "BusID",
            "Description": "Unique identifier for each bus",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "BusData",
            "ColumnName": "Voltage_pu",
            "Description": "Bus voltage magnitude in per unit",
            "Units": "pu",
            "DataType": "float",
            "ValidRange": "0.8 to 1.2"
        },
        {
            "Sheet": "BusData",
            "ColumnName": "Angle_deg",
            "Description": "Bus voltage angle in degrees",
            "Units": "degrees",
            "DataType": "float",
            "ValidRange": "-180 to 180"
        },
        {
            "Sheet": "BusData",
            "ColumnName": "Load_MW",
            "Description": "Total load at the bus",
            "Units": "MW",
            "DataType": "float",
            "ValidRange": "0 to 1000"
        },
        {
            "Sheet": "BusData",
            "ColumnName": "Gen_MW",
            "Description": "Total generation at the bus",
            "Units": "MW",
            "DataType": "float",
            "ValidRange": "0 to 2000"
        },
        {
            "Sheet": "BusData",
            "ColumnName": "CriticalityLevel",
            "Description": "Bus importance level (1-5)",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to 5"
        },
        {
            "Sheet": "BusData",
            "ColumnName": "Status",
            "Description": "Bus operational status",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 or 1"
        }
    ])
    
    # LineData columns
    definitions.extend([
        {
            "Sheet": "LineData",
            "ColumnName": "LineID",
            "Description": "Unique identifier for each line",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_LINES"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "FromBusID",
            "Description": "Source bus identifier",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "ToBusID",
            "Description": "Destination bus identifier",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "R_pu",
            "Description": "Line resistance in per unit",
            "Units": "pu",
            "DataType": "float",
            "ValidRange": "0 to 0.1"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "X_pu",
            "Description": "Line reactance in per unit",
            "Units": "pu",
            "DataType": "float",
            "ValidRange": "0.1 to 1.0"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "B_pu",
            "Description": "Line susceptance in per unit",
            "Units": "pu",
            "DataType": "float",
            "ValidRange": "0 to 10"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "ThermalLimit_MW",
            "Description": "Maximum power flow limit",
            "Units": "MW",
            "DataType": "float",
            "ValidRange": "100 to 2000"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "CriticalityLevel",
            "Description": "Line importance level (1-5)",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to 5"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "Status",
            "Description": "Line operational status",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 or 1"
        },
        {
            "Sheet": "LineData",
            "ColumnName": "FailureRate_pct",
            "Description": "Annual failure rate percentage",
            "Units": "%",
            "DataType": "float",
            "ValidRange": "0 to 10"
        }
    ])
    
    # SwitchData columns
    definitions.extend([
        {
            "Sheet": "SwitchData",
            "ColumnName": "SwitchID",
            "Description": "Unique identifier for each switch",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_SWITCHES"
        },
        {
            "Sheet": "SwitchData",
            "ColumnName": "FromBusID",
            "Description": "Source bus identifier",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "SwitchData",
            "ColumnName": "ToBusID",
            "Description": "Destination bus identifier",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "SwitchData",
            "ColumnName": "Type",
            "Description": "Switch type (Breaker, Isolator, etc.)",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "Breaker, Isolator, Recloser"
        },
        {
            "Sheet": "SwitchData",
            "ColumnName": "Status",
            "Description": "Switch operational status",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 or 1"
        },
        {
            "Sheet": "SwitchData",
            "ColumnName": "FailureRate_pct",
            "Description": "Annual failure rate percentage",
            "Units": "%",
            "DataType": "float",
            "ValidRange": "0 to 5"
        }
    ])
    
    # Measurements columns
    definitions.extend([
        {
            "Sheet": "Measurements",
            "ColumnName": "SnapshotID",
            "Description": "Unique identifier for each system snapshot",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to N_SNAPSHOTS-1"
        },
        {
            "Sheet": "Measurements",
            "ColumnName": "Timestamp",
            "Description": "Exact time of the measurement",
            "Units": "",
            "DataType": "datetime",
            "ValidRange": "7 days of 1-minute intervals"
        },
        {
            "Sheet": "Measurements",
            "ColumnName": "MeasurementID",
            "Description": "Unique identifier for each measurement",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_MEASUREMENTS"
        },
        {
            "Sheet": "Measurements",
            "ColumnName": "BusID",
            "Description": "Bus identifier for the measurement",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "Measurements",
            "ColumnName": "LineID",
            "Description": "Line identifier for the measurement",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_LINES"
        },
        {
            "Sheet": "Measurements",
            "ColumnName": "MeasurementType",
            "Description": "Type of measurement",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "Voltage, PowerFlow, Current, PowerInjection"
        },
        {
            "Sheet": "Measurements",
            "ColumnName": "Value",
            "Description": "Measured value",
            "Units": "varies",
            "DataType": "float",
            "ValidRange": "varies by type"
        },
        {
            "Sheet": "Measurements",
            "ColumnName": "MeasurementSource",
            "Description": "Source of the measurement",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "SCADA, PMU, DFR"
        },
        {
            "Sheet": "Measurements",
            "ColumnName": "MeasurementQualityFlag",
            "Description": "Quality indicator for the measurement",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "Good, Bad, Suspect"
        }
    ])
    
    # ContingencyEvents columns
    definitions.extend([
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "SnapshotID",
            "Description": "Unique identifier for each system snapshot",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to N_SNAPSHOTS-1"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "EventID",
            "Description": "Unique identifier for each event",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_EVENTS"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "Type",
            "Description": "Type of contingency event",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "Line, Generator, Load, Bus"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "ElementType",
            "Description": "Type of element involved",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "Transmission, Distribution, Generation, Load"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "ElementID",
            "Description": "Identifier of the affected element",
            "Units": "",
            "DataType": "int",
            "ValidRange": "varies by type"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "StartTime",
            "Description": "Event start time",
            "Units": "",
            "DataType": "datetime",
            "ValidRange": "7 days of 1-minute intervals"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "EndTime",
            "Description": "Event end time",
            "Units": "",
            "DataType": "datetime",
            "ValidRange": "7 days of 1-minute intervals"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "DetectionMethod",
            "Description": "Method used to detect the event",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "SCADA, PMU, DFR, Operator"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "WeatherID",
            "Description": "Associated weather condition",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_WEATHER"
        },
        {
            "Sheet": "ContingencyEvents",
            "ColumnName": "SPSAction",
            "Description": "Special Protection Scheme action taken",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "None, LoadShed, GenerationTrip, LineTrip"
        }
    ])
    
    # SeverityLabels columns
    definitions.extend([
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "EventID",
            "Description": "Unique identifier for each event",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_EVENTS"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "SeverityRank",
            "Description": "Severity ranking of the event",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to 5"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "MaxOverload_pct",
            "Description": "Maximum line overload percentage",
            "Units": "%",
            "DataType": "float",
            "ValidRange": "0 to 200"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "VoltageViolation_MaxDev_pu",
            "Description": "Maximum voltage deviation",
            "Units": "pu",
            "DataType": "float",
            "ValidRange": "0 to 0.5"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "EstimatedVoLL_CAD",
            "Description": "Estimated Value of Lost Load",
            "Units": "CAD",
            "DataType": "float",
            "ValidRange": "0 to 1000000"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "CascadingProbability_pct",
            "Description": "Probability of cascading failures",
            "Units": "%",
            "DataType": "float",
            "ValidRange": "0 to 100"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "RestorationTime_min",
            "Description": "Estimated restoration time",
            "Units": "minutes",
            "DataType": "float",
            "ValidRange": "0 to 1440"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "LoadSheddingAmount_MW",
            "Description": "Amount of load shed",
            "Units": "MW",
            "DataType": "float",
            "ValidRange": "0 to 1000"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "EconomicImpact_CAD",
            "Description": "Total economic impact",
            "Units": "CAD",
            "DataType": "float",
            "ValidRange": "0 to 1000000"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "CriticalityLevel",
            "Description": "Criticality level of the event",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to 5"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "WeatherImpactFlag",
            "Description": "Weather impact indicator",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 or 1"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "OperatorActionRequired",
            "Description": "Operator action requirement",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 or 1"
        },
        {
            "Sheet": "SeverityLabels",
            "ColumnName": "LabelTimestamp",
            "Description": "Time of severity assessment",
            "Units": "",
            "DataType": "datetime",
            "ValidRange": "7 days of 1-minute intervals"
        }
    ])
    
    # TopologyMeta columns
    definitions.extend([
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "SnapshotID",
            "Description": "Unique identifier for each system snapshot",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to N_SNAPSHOTS-1"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "Timestamp",
            "Description": "Exact time of the snapshot",
            "Units": "",
            "DataType": "datetime",
            "ValidRange": "7 days of 1-minute intervals"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "NumIslands",
            "Description": "Count of disconnected network segments",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to 10"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "JacobianRank",
            "Description": "Rank of the SE Jacobian matrix",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to 2*ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "ObservableFlag",
            "Description": "Observability indicator",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 or 1"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "NodalObservability_pct",
            "Description": "Percentage of observable buses",
            "Units": "%",
            "DataType": "float",
            "ValidRange": "0 to 100"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "CriticalPathsOutOfService",
            "Description": "Number of critical paths out of service",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to N_LINES"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "MaxIslandSize_buses",
            "Description": "Largest island's bus count",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "MinCutSize_lines",
            "Description": "Minimum cut size for islanding",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_LINES"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "AverageLineLoading_pct",
            "Description": "Average line loading percentage",
            "Units": "%",
            "DataType": "float",
            "ValidRange": "0 to 100"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "VoltageViolationCount",
            "Description": "Number of voltage violations",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to ROWS_PER_SNAPSHOT"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "OverloadCount",
            "Description": "Number of overloaded lines",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to N_LINES"
        },
        {
            "Sheet": "TopologyMeta",
            "ColumnName": "BadDataFlag",
            "Description": "Bad data indicator",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 or 1"
        }
    ])
    
    # WeatherTime columns
    definitions.extend([
        {
            "Sheet": "WeatherTime",
            "ColumnName": "Timestamp",
            "Description": "Exact time of the weather measurement",
            "Units": "",
            "DataType": "datetime",
            "ValidRange": "7 days of 1-minute intervals"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "Temperature_C",
            "Description": "Ambient temperature",
            "Units": "°C",
            "DataType": "float",
            "ValidRange": "-40 to 50"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "WindSpeed_mps",
            "Description": "Wind speed",
            "Units": "m/s",
            "DataType": "float",
            "ValidRange": "0 to 30"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "WindDirection_deg",
            "Description": "Wind direction",
            "Units": "degrees",
            "DataType": "int",
            "ValidRange": "0 to 359"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "SolarIrradiance_W/m2",
            "Description": "Solar irradiance",
            "Units": "W/m²",
            "DataType": "float",
            "ValidRange": "0 to 1000"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "Precipitation_mm",
            "Description": "Precipitation rate",
            "Units": "mm/h",
            "DataType": "float",
            "ValidRange": "0 to 100"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "Humidity_pct",
            "Description": "Relative humidity",
            "Units": "%",
            "DataType": "float",
            "ValidRange": "0 to 100"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "Visibility_km",
            "Description": "Visibility distance",
            "Units": "km",
            "DataType": "float",
            "ValidRange": "0 to 10"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "WeatherCondition",
            "Description": "Weather condition description",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "Clear, Cloudy, Rain, Snow, Fog, Windy, Humid"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "StormFlag",
            "Description": "Storm condition indicator",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 or 1"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "WindPower_MW",
            "Description": "Wind power generation",
            "Units": "MW",
            "DataType": "float",
            "ValidRange": "0 to 2000"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "SolarPower_MW",
            "Description": "Solar power generation",
            "Units": "MW",
            "DataType": "float",
            "ValidRange": "0 to 1000"
        },
        {
            "Sheet": "WeatherTime",
            "ColumnName": "LineFailureProbabilities",
            "Description": "Line failure probabilities",
            "Units": "",
            "DataType": "JSON",
            "ValidRange": "Dictionary of line IDs to probabilities (0-0.1)"
        }
    ])
    
    # AreaMeta columns
    definitions.extend([
        {
            "Sheet": "AreaMeta",
            "ColumnName": "AreaID",
            "Description": "Unique identifier for each area",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to N_AREAS"
        },
        {
            "Sheet": "AreaMeta",
            "ColumnName": "Name",
            "Description": "Area name",
            "Units": "",
            "DataType": "string",
            "ValidRange": "Any string"
        },
        {
            "Sheet": "AreaMeta",
            "ColumnName": "Type",
            "Description": "Area type",
            "Units": "",
            "DataType": "enum",
            "ValidRange": "Transmission, Distribution, Generation, Load"
        },
        {
            "Sheet": "AreaMeta",
            "ColumnName": "ParentAreaID",
            "Description": "Parent area identifier",
            "Units": "",
            "DataType": "int",
            "ValidRange": "0 to N_AREAS"
        },
        {
            "Sheet": "AreaMeta",
            "ColumnName": "CriticalityLevel",
            "Description": "Area importance level",
            "Units": "",
            "DataType": "int",
            "ValidRange": "1 to 5"
        }
    ])
    
    return pd.DataFrame(definitions)

def main():
    global bus_data, line_data, switch_data, measurement_data, contingency_events
    global severity_labels, topology_meta, weather_data, area_meta, column_definitions
    print("Generating synthetic grid contingency data...")
    
    # Phase 1: Generate base data
    print("\nGenerating base data...")
    bus_data = sample_bus_data()
    line_data = sample_line_data()
    switch_data = sample_switch_data()
    
    # Phase 2: Generate dependent data
    print("\nGenerating dependent data...")
    measurement_data = sample_measurement_data()
    weather_data = sample_weather()  # Generate weather data before contingencies
    contingency_events = sample_contingency_events()
    severity_labels = sample_severity()
    topology_meta = sample_topology_meta()
    area_meta = sample_area_meta()
    column_definitions = generate_column_definitions()
    
    # Save to Excel
    print("\nSaving data to Excel...")
    with pd.ExcelWriter("SyntheticGridContingencies.xlsx", engine='openpyxl') as writer:
        bus_data.to_excel(writer, sheet_name="BusData", index=False)
        line_data.to_excel(writer, sheet_name="LineData", index=False)
        switch_data.to_excel(writer, sheet_name="SwitchData", index=False)
        measurement_data.to_excel(writer, sheet_name="MeasurementData", index=False)
        contingency_events.to_excel(writer, sheet_name="ContingencyEvents", index=False)
        severity_labels.to_excel(writer, sheet_name="SeverityLabels", index=False)
        topology_meta.to_excel(writer, sheet_name="TopologyMeta", index=False)
        weather_data.to_excel(writer, sheet_name="WeatherData", index=False)
        area_meta.to_excel(writer, sheet_name="AreaMeta", index=False)
        column_definitions.to_excel(writer, sheet_name="ColumnDefinitions", index=False)
    
    print("\nData generation complete! File saved as 'SyntheticGridContingencies.xlsx'")

def sample_measurement_data():
    """Generate synthetic measurement data"""
    ts = make_timestamps()
    rows = []
    
    print("Generating measurement data...")
    for snap_id, t in tqdm(enumerate(ts), total=len(ts), desc="Measurement Data"):
        # Get current line and bus data
        current_lines = line_data[line_data["SnapshotID"] == snap_id]
        current_buses = bus_data[bus_data["SnapshotID"] == snap_id]
        
        # Generate measurements for each line
        for _, line in current_lines.iterrows():
            # Voltage measurements at both ends
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "MeasurementID": f"V_{line['FromBusID']}_{snap_id}",
                "Type": "Voltage",
                "ElementID": line["FromBusID"],
                "ElementType": "Bus",
                "Value": line["FromVoltage_pu"],
                "Unit": "pu",
                "Quality": 1 if np.random.random() > 0.01 else 0  # 1% bad data
            })
            
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "MeasurementID": f"V_{line['ToBusID']}_{snap_id}",
                "Type": "Voltage",
                "ElementID": line["ToBusID"],
                "ElementType": "Bus",
                "Value": line["ToVoltage_pu"],
                "Unit": "pu",
                "Quality": 1 if np.random.random() > 0.01 else 0
            })
            
            # Power flow measurements
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "MeasurementID": f"P_{line['LineID']}_{snap_id}",
                "Type": "PowerFlow",
                "ElementID": line["LineID"],
                "ElementType": "Line",
                "Value": line["PowerFlow_MW"],
                "Unit": "MW",
                "Quality": 1 if np.random.random() > 0.01 else 0
            })
            
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "MeasurementID": f"Q_{line['LineID']}_{snap_id}",
                "Type": "ReactivePower",
                "ElementID": line["LineID"],
                "ElementType": "Line",
                "Value": line["ReactivePower_MVAR"],
                "Unit": "MVAR",
                "Quality": 1 if np.random.random() > 0.01 else 0
            })
        
        # Generate measurements for each bus
        for _, bus in current_buses.iterrows():
            # Load measurements
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "MeasurementID": f"L_{bus['BusID']}_{snap_id}",
                "Type": "Load",
                "ElementID": bus["BusID"],
                "ElementType": "Bus",
                "Value": bus["Load_MW"],
                "Unit": "MW",
                "Quality": 1 if np.random.random() > 0.01 else 0
            })
            
            # Generation measurements
            rows.append({
                "SnapshotID": snap_id,
                "Timestamp": t,
                "MeasurementID": f"G_{bus['BusID']}_{snap_id}",
                "Type": "Generation",
                "ElementID": bus["BusID"],
                "ElementType": "Bus",
                "Value": bus["Generation_MW"],
                "Unit": "MW",
                "Quality": 1 if np.random.random() > 0.01 else 0
            })
    
    return pd.DataFrame(rows)

def sample_contingency_events():
    """Generate synthetic contingency events"""
    ts = make_timestamps()
    rows = []
    event_id = 1
    
    print("Generating contingency events...")
    for snap_id, t in tqdm(enumerate(ts), total=len(ts), desc="Contingency Events"):
        # Get current line and bus data
        current_lines = line_data[line_data["SnapshotID"] == snap_id]
        current_buses = bus_data[bus_data["SnapshotID"] == snap_id]
        
        # Generate line contingencies
        for _, line in current_lines.iterrows():
            if line["Status"] == 0:  # Only generate events for out-of-service lines
                # Calculate failure probability based on line characteristics
                base_prob = get_line_failure_probability(line["Length_km"], line["Age_years"], line["CriticalityLevel"])
                
                # Adjust probability based on weather
                weather_impact = 1.0
                if snap_id < len(ts):
                    weather = weather_data[weather_data["SnapshotID"] == snap_id].iloc[0]
                    weather_impact = calculate_weather_impact(weather, line)
                
                final_prob = base_prob * weather_impact
                
                if np.random.random() < final_prob:
                    # Determine contingency type
                    if line["CriticalityLevel"] >= 2:
                        cont_type = np.random.choice(
                            ["LineTrip", "CascadingTrip", "SPSAction"],
                            p=[0.6, 0.3, 0.1]
                        )
                    else:
                        cont_type = "LineTrip"
                    
                    # Generate event details
                    rows.append({
                        "SnapshotID": snap_id,
                        "EventID": event_id,
                        "Type": cont_type,
                        "ElementType": "Line",
                        "ElementID": line["LineID"],
                        "Timestamp": t,
                        "Duration_min": np.random.exponential(30),  # Mean 30 minutes
                        "WeatherImpact": 1 if weather_impact > 1.2 else 0,
                        "SPSTriggered": 1 if cont_type == "SPSAction" else 0,
                        "CascadingFlag": 1 if cont_type == "CascadingTrip" else 0
                    })
                    event_id += 1
        
        # Generate bus contingencies
        for _, bus in current_buses.iterrows():
            if bus["Status"] == 0:  # Only generate events for out-of-service buses
                base_prob = 0.01  # Base probability for bus failures
                
                # Adjust probability based on weather
                weather_impact = 1.0
                if snap_id < len(ts):
                    weather = weather_data[weather_data["SnapshotID"] == snap_id].iloc[0]
                    weather_impact = calculate_weather_impact(weather, bus)
                
                final_prob = base_prob * weather_impact
                
                if np.random.random() < final_prob:
                    rows.append({
                        "SnapshotID": snap_id,
                        "EventID": event_id,
                        "Type": "BusTrip",
                        "ElementType": "Bus",
                        "ElementID": bus["BusID"],
                        "Timestamp": t,
                        "Duration_min": np.random.exponential(45),  # Mean 45 minutes
                        "WeatherImpact": 1 if weather_impact > 1.2 else 0,
                        "SPSTriggered": 0,
                        "CascadingFlag": 0
                    })
                    event_id += 1
    
    return pd.DataFrame(rows)

def sample_severity():
    """Sample severity labels for contingencies"""
    global contingency_events, line_data, bus_data, generator_data, transformer_data
    
    # Create severity labels DataFrame
    severity_labels = pd.DataFrame(columns=[
        "ContingencyID",
        "SeverityScore",
        "CriticalityLevel",
        "ImpactDescription"
    ])
    
    # Sample severity for each contingency
    for _, event in tqdm(contingency_events.iterrows(), total=len(contingency_events), desc="Severity Labels"):
        # Calculate base severity score
        base_score = 0
        
        # Add impact based on component type
        if event["ElementType"] == "Line":
            line = line_data[line_data["LineID"] == event["ElementID"]].iloc[0]
            base_score += line["ThermalLimit_MW"] / 100  # Normalize by 100 MW
            base_score += line["CriticalityLevel"] * 0.5  # Add criticality impact
        elif event["ElementType"] == "Bus":
            bus = bus_data[bus_data["BusID"] == event["ElementID"]].iloc[0]
            base_score += bus["Load_MW"] / 100  # Normalize by 100 MW
            base_score += 0.3  # Base impact for bus contingencies
        
        # Add random variation
        severity_score = base_score * np.random.uniform(0.8, 1.2)
        
        # Determine criticality level based on severity score
        if severity_score < 0.5:
            rank = 1  # Low
        elif severity_score < 1.0:
            rank = 2  # Medium
        else:
            rank = 3  # High
        
        # Generate impact description
        impact_desc = f"Impact level {rank} with severity score {severity_score:.2f}"
        
        # Add to DataFrame
        severity_labels = pd.concat([severity_labels, pd.DataFrame([{
            "ContingencyID": event["EventID"],
            "SeverityScore": severity_score,
            "CriticalityLevel": rank,
            "ImpactDescription": impact_desc
        }])], ignore_index=True)
    
    return severity_labels

def sample_topology_meta():
    """Generate synthetic topology metadata"""
    ts = make_timestamps()
    rows = []
    
    print("Generating topology metadata...")
    for snap_id, t in tqdm(enumerate(ts), total=len(ts), desc="Topology Meta"):
        # Get line statuses
        line_status = {
            (row["FromBusID"], row["ToBusID"]): row["Status"]
            for _, row in line_data.iterrows()
        }
        
        # Find islands
        bus_pairs = generate_bus_pairs()
        islands = find_islands(bus_pairs, line_status)
        num_islands = len(islands)
        
        # Calculate minimum cut size
        min_cut = calculate_min_cut_size(bus_pairs, line_status, num_islands)
        
        # Get line loadings
        line_loadings = {
            row["LineID"]: get_line_loading(
                row["LineID"],
                np.random.uniform(0, row["ThermalLimit_MW"]),
                row["ThermalLimit_MW"]
            )
            for _, row in line_data.iterrows()
        }
        
        # Get voltage violations
        bus_voltages = {
            row["BusID"]: np.random.uniform(0.95, 1.05)
            for _, row in bus_data.iterrows()
        }
        voltage_violations = get_voltage_violations(bus_voltages)
        
        # Get overloads
        overloads = get_overloads(line_loadings)
        
        # Check for bad data
        bad_data = 1 if np.random.random() < 0.01 else 0  # 1% chance of bad data
        
        rows.append({
            "SnapshotID": snap_id,
            "Timestamp": t,
            "NumberOfIslands": num_islands,
            "MinimumCutSize": min_cut,
            "MaxLineLoading_pct": max(line_loadings.values()) if line_loadings else 0,
            "VoltageViolations": voltage_violations,
            "OverloadedLines": overloads,
            "BadDataFlag": bad_data
        })
    
    return pd.DataFrame(rows)

def sample_weather():
    """Generate synthetic weather data"""
    ts = make_timestamps()
    rows = []
    
    print("Generating weather data...")
    for snap_id, t in tqdm(enumerate(ts), total=len(ts), desc="Weather Data"):
        # Get daily mean temperature
        day_of_year = t.timetuple().tm_yday
        mean_temp = calculate_daily_mean_temp(day_of_year)
        
        # Add hourly variation
        hour = t.hour
        temp = mean_temp + calculate_hourly_temp_variation(hour)
        
        # Calculate other weather parameters
        wind_speed = calculate_wind_speed(temp, hour)
        solar_irradiance = calculate_solar_irradiance(day_of_year, hour)
        precipitation = calculate_precipitation(day_of_year, hour)
        humidity = np.random.uniform(30, 90)  # Random humidity between 30% and 90%
        visibility = np.random.uniform(0, 10)  # 0-10 km
        
        rows.append({
            "SnapshotID": snap_id,
            "Timestamp": t,
            "Temperature_C": temp,
            "WindSpeed_mps": wind_speed,
            "SolarIrradiance_Wpm2": solar_irradiance,
            "Precipitation_mm": precipitation,
            "WeatherCondition": get_weather_condition(temp, wind_speed, precipitation, humidity, visibility)
        })
    
    return pd.DataFrame(rows)

def calculate_hourly_temp_variation(hour):
    # Example: Simple sinusoidal model for hourly temperature variation
    # Adjust parameters as needed for your use case
    amplitude = 5  # Temperature variation amplitude
    phase_shift = 6  # Phase shift to align with peak temperature at noon
    return amplitude * np.sin(2 * np.pi * (hour + phase_shift) / 24)

def calculate_wind_speed(temp, hour):
    # Example: Simple model for wind speed
    # Adjust parameters as needed for your use case
    base_speed = 5  # Base wind speed in m/s
    temp_factor = 0.1 * (temp - 15)  # Wind increases with temperature deviation from 15°C
    hour_factor = 2 * np.sin(2 * np.pi * (hour - 6) / 24)  # Diurnal variation
    return max(0, base_speed + temp_factor + hour_factor)

def calculate_solar_irradiance(day_of_year, hour):
    # Example: Simple model for solar irradiance
    # Adjust parameters as needed for your use case
    base_irradiance = 800  # Base irradiance in W/m²
    seasonal_factor = 200 * np.sin(2 * np.pi * (day_of_year - 172) / 365)  # Seasonal variation
    diurnal_factor = 400 * np.sin(np.pi * (hour - 6) / 12)  # Diurnal variation
    return max(0, base_irradiance + seasonal_factor + diurnal_factor)

def calculate_precipitation(day_of_year, hour):
    # Example: Simple model for precipitation
    # Adjust parameters as needed for your use case
    base_precipitation = 0  # Base precipitation in mm
    seasonal_factor = 5 * np.sin(2 * np.pi * (day_of_year - 172) / 365)  # Seasonal variation
    diurnal_factor = 2 * np.sin(np.pi * (hour - 6) / 12)  # Diurnal variation
    return max(0, base_precipitation + seasonal_factor + diurnal_factor)

def calculate_weather_impact(weather_data, line):
    # Example: Simple model for weather impact
    # Adjust parameters as needed for your use case
    temp = weather_data["Temperature_C"]
    wind_speed = weather_data["WindSpeed_mps"]
    precip = weather_data["Precipitation_mm"]
    
    # Temperature impact: higher temperatures increase failure probability
    temp_factor = 1 + 0.1 * (temp - 20) / 20  # 10% increase per 20°C above 20°C
    
    # Wind impact: higher wind speeds increase failure probability
    wind_factor = 1 + 0.2 * (wind_speed - 10) / 10  # 20% increase per 10 m/s above 10 m/s
    
    # Precipitation impact: higher precipitation increases failure probability
    precip_factor = 1 + 0.3 * precip / 10  # 30% increase per 10 mm of precipitation
    
    return temp_factor * wind_factor * precip_factor

if __name__ == "__main__":
    main() 