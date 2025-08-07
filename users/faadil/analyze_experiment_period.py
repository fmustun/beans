import os
import re
from datetime import datetime
import matplotlib.pyplot as plt

# Path to the dataset folder or recordings folder
# Change this path to analyze a different folder
PATH = "/media/DOLPHIN2/Recordings"  # or "/media/DOLPHIN1/new_extraction_dataset/start_2021"

# Regex for experiment folders like Exp_01_Apr_2021_0045am
FOLDER_REGEX = re.compile(r"Exp_(\d{2})_(\w{3})_(\d{4})_(\d{4})(am|pm)?$")
# Regex for wav files like Exp_01_Dec_2021_1545_channel_0.wav (more robust)
FILE_REGEX = re.compile(r"Exp_(\d{2})_(\w{3})_(\d{4})_(\d{4})_channel_\d+\.wav", re.IGNORECASE)

# Month abbreviation to number
MONTHS = {month: idx for idx, month in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}

def parse_folder_datetime(folder_name):
    match = FOLDER_REGEX.match(folder_name)
    if not match:
        return None
    day_str, month_str, year, time_str, ampm = match.groups()
    month = MONTHS.get(month_str)
    if not month:
        return None
    day = int(day_str)
    hour = int(time_str[:2])
    minute = int(time_str[2:])
    if ampm:
        if ampm == 'pm' and hour != 12:
            hour += 12
        if ampm == 'am' and hour == 12:
            hour = 0
    try:
        dt = datetime(int(year), month, day, hour, minute)
        return dt
    except Exception:
        return None

def parse_file_datetime(filename):
    match = FILE_REGEX.search(filename)
    if not match:
        return None
    day_str, month_str, year, time_str = match.groups()
    month = MONTHS.get(month_str)
    if not month:
        return None
    day = int(day_str)
    hour = int(time_str[:2])
    minute = int(time_str[2:])
    try:
        dt = datetime(int(year), month, day, hour, minute)
        return dt
    except Exception:
        return None

def main():
    entries = os.listdir(PATH)
    folders = [f for f in entries if os.path.isdir(os.path.join(PATH, f))]
    wav_files = [f for f in entries if f.lower().endswith('.wav')]
    print(f"Found {len(folders)} folders and {len(wav_files)} wav files in {PATH}")
    datetimes = []
    mode = None
    # Prioritize wav files if any are present
    if wav_files:
        mode = 'wav files'
        for filename in wav_files:
            dt = parse_file_datetime(filename)
            if dt:
                datetimes.append(dt)
    elif folders:
        mode = 'folders'
        for folder in folders:
            dt = parse_folder_datetime(folder)
            if dt:
                datetimes.append(dt)
    else:
        print("No valid experiment folders or .wav files found.")
        return
    print(f"Parsed {len(datetimes)} datetimes from {mode}.")
    if not datetimes:
        print(f"No valid datetimes extracted from {mode}.")
        return
    datetimes.sort()
    print(f"Mode: {mode}")
    print(f"Earliest: {datetimes[0]}")
    print(f"Latest:   {datetimes[-1]}")
    # Plot
    plt.figure(figsize=(12, 2))
    color = 'blue' if mode == 'folders' else 'green'
    plt.scatter(datetimes, [1]*len(datetimes), marker='|', color=color)
    plt.yticks([])
    plt.xlabel('Date and Time')
    plt.title(f'Experiment Timeline ({mode})')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 