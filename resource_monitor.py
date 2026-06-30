import psutil
import subprocess
import time
import os

# Interval updated to 30 seconds
INTERVAL = 30 

log_file = "system_log.txt"
print(f"Logging to '{log_file}' every {INTERVAL} seconds. Press Ctrl+C to stop.")

# Open in append mode
with open(log_file, "a") as f:
    # Only write header if the file is new/empty
    if os.stat(log_file).st_size == 0:
        f.write("Timestamp, RAM_Available_MB, GPU_Free_MB\n")
    print("Starting monitoring...")
    print("Timestamp, RAM_Available_MB, GPU_Free_MB")
    try:
        while True:
            # Get RAM info (Available MB)
            ram = psutil.virtual_memory().available / (1024 * 1024)

            # Get GPU info (requires NVIDIA drivers/nvidia-smi)
            try:
                gpu_data = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                    stderr=subprocess.DEVNULL # Suppress error messages if command fails
                ).decode().strip()
            except:
                gpu_data = "N/A"

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp}, {ram:.2f}, {gpu_data}"
            print(log_line)

            f.write(log_line + "\n")
            f.flush()  # Forces write to disk immediately
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
