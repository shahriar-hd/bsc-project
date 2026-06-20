import time
import os
import glob
import psutil
import pandas as pd
from datetime import datetime
from pynvml import *

SAMPLE_INTERVAL = 1
RAM_W_PER_GB = 0.35
OTHER_COMPONENTS_W = 25

LOG_FILE = "power_log.csv"

########################################
# GPU
########################################

nvmlInit()
GPU_COUNT = nvmlDeviceGetCount()

def get_gpu_power():

    total = 0

    for i in range(GPU_COUNT):
        handle = nvmlDeviceGetHandleByIndex(i)
        try:
            p = nvmlDeviceGetPowerUsage(handle) / 1000
            total += p
        except:
            pass

    return total

########################################
# CPU RAPL detection
########################################

def find_rapl_energy_files():

    paths = glob.glob("/sys/class/powercap/intel-rapl*/energy_uj")
    paths += glob.glob("/sys/class/powercap/*/energy_uj")

    return paths


rapl_files = find_rapl_energy_files()

last_cpu_energy = None
last_time = None


def get_cpu_power():

    global last_cpu_energy, last_time

    if not rapl_files:
        return 0

    total_energy = 0

    for f in rapl_files:
        try:
            with open(f) as file:
                total_energy += int(file.read())
        except:
            pass

    now = time.time()

    if last_cpu_energy is None:
        last_cpu_energy = total_energy
        last_time = now
        return 0

    energy_diff = (total_energy - last_cpu_energy) / 1e6
    time_diff = now - last_time

    last_cpu_energy = total_energy
    last_time = now

    if time_diff == 0:
        return 0

    power = energy_diff / time_diff

    return max(power, 0)

########################################
# RAM estimate
########################################

def get_ram_power():

    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)

    return used_gb * RAM_W_PER_GB

########################################
# Logging
########################################

records = []
total_energy_Wh = 0

print("\nPower monitor started...\n")

while True:

    timestamp = datetime.now()

    gpu = get_gpu_power()
    cpu = get_cpu_power()
    ram = get_ram_power()

    total_power = gpu + cpu + ram + OTHER_COMPONENTS_W

    energy = total_power * SAMPLE_INTERVAL / 3600
    total_energy_Wh += energy

    records.append(total_power)

    if len(records) >= 60:
        avg_1m = sum(records[-60:]) / 60
    else:
        avg_1m = sum(records) / len(records)

    print(
        f"{timestamp.strftime('%H:%M:%S')} | "
        f"GPU {gpu:.1f}W | "
        f"CPU {cpu:.1f}W | "
        f"RAM {ram:.1f}W | "
        f"TOTAL {total_power:.1f}W | "
        f"ENERGY {total_energy_Wh:.3f} Wh | "
        f"AVG1m {avg_1m:.1f}W"
    )

    records_df = pd.DataFrame([{
        "time": timestamp,
        "gpu_w": gpu,
        "cpu_w": cpu,
        "ram_w": ram,
        "total_w": total_power,
        "energy_wh": total_energy_Wh
    }])

    if not os.path.exists(LOG_FILE):
        records_df.to_csv(LOG_FILE, index=False)
    else:
        records_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    time.sleep(SAMPLE_INTERVAL)
