import pybrams
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import pandas as pd

# --- CONFIGURATION ---
# Target: Peak Geminids 2024 (Dec 14, 02:00 UTC)
# Format: YYYY-MM-DDThh:mm:ss
START_TIME = "2024-12-14T01:00:00"
END_TIME   = "2024-12-14T03:00:00"
INTERVAL_STR = f"{START_TIME}/{END_TIME}"

# Station: BEHUMA (Humain) is the best reference. 
# We search for it in the available systems.
print("Searching for Humain station (BEHUMA)...")
all_systems = pybrams.system.all()
target_system = []

for code in all_systems.keys():
    if "BEHUMA" in code:
        target_system.append(code)
        break # Just grab the first BEHUMA system found

if not target_system:
    print("Warning: BEHUMA not found, falling back to all systems (this might be slow).")
    target_system = None # None fetches everything (careful!)

print(f"Targeting System: {target_system}")

# --- 1. FETCH DATA ---
print("Initializing download via pybrams...")
interval = pybrams.utils.interval.Interval.from_string(INTERVAL_STR)

# clean=True ensures valid files
files_dict = pybrams.brams.file.get(interval, target_system, clean=True)

# --- 2. PROCESS DATA (Generate SNR Series) ---
print("Processing raw signals...")

all_snr_values = []
timestamps = []
global_time_counter = 0

for sys_code, file_list in files_dict.items():
    for file in file_list:
        
        # Extract Raw Audio from the BRAMS object
        # The notebook shows data is stored in file.signal.series.data
        raw_audio = file.signal.series.data
        
        # BRAMS usually stores sample rate in the signal object
        # If file.signal.fs doesn't exist, hardcode 5512 (standard BRAMS rate)
        try:
            fs = file.signal.fs
        except AttributeError:
            fs = 5512 

        # --- SNR CALCULATION (Same logic as before) ---
        nfft = 4096
        step = int(fs * 0.1) # 100ms steps
        
        for i in range(0, len(raw_audio) - nfft, step):
            chunk = raw_audio[i : i + nfft]
            
            # Power Spectral Density
            f, pxx = welch(chunk, fs=fs, nperseg=nfft)
            pxx_db = 10 * np.log10(pxx + 1e-12)
            
            # Masks for BRAMS (Beacon is ~1000Hz in mixed audio)
            # Signal: 1050-1500 Hz | Noise: 1600-2000 Hz
            sig_mask = (f > 1050) & (f < 1500)
            noise_mask = (f > 1600) & (f < 2000)
            
            if np.sum(sig_mask) > 0 and np.sum(noise_mask) > 0:
                signal_power = np.max(pxx_db[sig_mask])
                noise_floor = np.median(pxx_db[noise_mask])
                snr = signal_power - noise_floor
                
                all_snr_values.append(snr)
                timestamps.append(global_time_counter)
                global_time_counter += 0.1 # Increment by 0.1s

# --- 3. SAVE AND PLOT ---
print(f"Extracted {len(all_snr_values)} data points.")

# Save to CSV for your Arduino Training
df = pd.DataFrame({"time": timestamps, "snr": all_snr_values})
df.to_csv("data/brams_geminids_2024.csv", index=False)
print("Saved to data/brams_geminids_2024.csv")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['snr'], color='#458588', linewidth=0.8)
plt.title(f"Geminids 2024 SNR Profile ({target_system[0]})")
plt.xlabel("Time (s)")
plt.ylabel("SNR (dB)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()