import pandas as pd

# ==== Config ====
CSV_PATH = "/home/shahriar/Documents/bank_did_auth/checkpoints/run01/results.csv"

# Grid emission factor for Iran
# 0.494 kg CO2 per kWh = 494 g CO2/kWh
IRAN_GRID_KG_PER_KWH = 0.494

# Optional overhead multiplier to account for laptop/system losses
OVERHEAD_MULTIPLIER = 1.15

# ==== Load data ====
df = pd.read_csv(CSV_PATH)

# Basic checks
required_cols = ["epoch", "elapsed_s", "power_w"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ==== Energy per epoch ====
# energy_kwh = power_w * elapsed_s / 3.6e6
df["energy_kwh_raw"] = df["power_w"] * df["elapsed_s"] / 3_600_000

# Apply overhead if desired
df["energy_kwh_adj"] = df["energy_kwh_raw"] * OVERHEAD_MULTIPLIER

# CO2
df["co2_kg"] = df["energy_kwh_adj"] * IRAN_GRID_KG_PER_KWH
df["co2_g"] = df["co2_kg"] * 1000

# Totals
total_energy_raw_kwh = df["energy_kwh_raw"].sum()
total_energy_adj_kwh = df["energy_kwh_adj"].sum()
total_co2_kg = df["co2_kg"].sum()
total_co2_g = df["co2_g"].sum()

print("=== Per-epoch results ===")
print(df[["epoch", "power_w", "elapsed_s", "energy_kwh_raw", "energy_kwh_adj", "co2_g"]].to_string(index=False))

print("\n=== Totals ===")
print(f"Raw energy:       {total_energy_raw_kwh:.6f} kWh")
print(f"Adjusted energy:  {total_energy_adj_kwh:.6f} kWh")
print(f"CO2 emissions:    {total_co2_kg:.6f} kg ({total_co2_g:.2f} g)")
