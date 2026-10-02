#!/usr/bin/env python3
"""
bno085_data_analyzer.py – Parse BNO085 telemetry files, calculate true ODR
vs software loop speed, evaluate min/max metrics, and auto-export artifacts.
"""

import sys
import argparse
import subprocess  # <-- Add this to handle system OS launches
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze BNO085 logged telemetry data.")
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        default=None,
        help="Path to the logged CSV or TXT file. If omitted, the newest log in the logs/ directory is used."
    )
    return parser.parse_args()

def get_target_file(requested_file):
    """Determines the target file path, falling back to the newest file if none requested."""
    logs_dir = Path.cwd() / "logs"

    if requested_file:
        return Path(requested_file)
        
    # If no file was specified, automatically scan the logs directory
    if not logs_dir.exists():
        print(f"[ERROR] No file specified and default directory does not exist: {logs_dir}")
        sys.exit(1)
        
    # Gather all csv, txt, and tsv files in logs/
    log_files = [
        p for p in logs_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in [".csv", ".txt", ".tsv"]
    ]
    
    if not log_files:
        print(f"[ERROR] No log files (.csv, .txt, .tsv) found inside: {logs_dir}")
        sys.exit(1)
        
    # Sort files by modification time to find the newest one
    newest_file = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Auto-detected newest log file: {newest_file.name}")
    return newest_file

def analyze_data():
    args = parse_args()
    
    # Process argument vs auto-detect logic
    file_path = get_target_file(args.file)
    
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
        
    print(f"[INFO] Reading file: {file_path.name}")
    
    # Handle CSV vs Tab-Delimited TXT files
    delimiter = "\t" if file_path.suffix.lower() in [".txt", ".tsv"] else ","
    df = pd.read_csv(file_path, sep=delimiter)
    
    # Ensure standard timestamp parsing
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df['time_seconds'] = (df['timestamp_dt'] - df['timestamp_dt'].iloc[0]).dt.total_seconds()
    
    total_duration = df['time_seconds'].iloc[-1]
    total_rows = len(df)
    
    # 1. Calculate Software Logging Speed (Loop Rate)
    software_loop_rate = total_rows / total_duration if total_duration > 0 else 0.0
    
    # Identify sensor groups present in log headers
    sensor_configs = {
        "accel": {"fields": ["accel_x", "accel_y", "accel_z"], "unit": "m/s²", "title": "Accelerometer"},
        "gyro":  {"fields": ["gyro_x", "gyro_y", "gyro_z"],   "unit": "rad/s", "title": "Gyroscope"},
        "mag":   {"fields": ["mag_x", "mag_y", "mag_z"],      "unit": "μT",    "title": "Magnetometer"}
    }
    
    active_sensors = [s for s, conf in sensor_configs.items() if all(f in df.columns for f in conf['fields'])]
    
    if not active_sensors:
        print("[ERROR] No valid BNO085 sensor headers recognized in file.")
        sys.exit(1)
        
    # Set up output directories
    output_dir = Path.cwd() / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    base_name = file_path.stem
    plot_out_path = output_dir / f"{base_name}_plot.png"
    info_out_path = output_dir / f"{base_name}_info.txt"
    
    info_lines = []
    info_lines.append("=" * 60)
    info_lines.append(f" BNO085 TELEMETRY ANALYSIS REPORT: {file_path.name}")
    info_lines.append("=" * 60)
    info_lines.append(f"File Target:       {file_path.resolve()}")
    info_lines.append(f"Total Log Records: {total_rows} frames")
    info_lines.append(f"Total Duration:    {total_duration:.3f} seconds")
    info_lines.append(f"Software Logging Speed (Loop Rate): {software_loop_rate:.1f} Hz\n")
    
    info_lines.append("-" * 60)
    info_lines.append(" SENSOR FREQUENCY & METRICS BREAKDOWN")
    info_lines.append("-" * 60)
    
    # 2. Determine True Output Data Rate (ODR) & Range Metrics
    for s_name in active_sensors:
        conf = sensor_configs[s_name]
        fields = conf['fields']
        unit = conf['unit']
        
        # Calculate true data updates by dropping consecutive identical samples
        df_unique = df.drop_duplicates(subset=fields)
        unique_count = len(df_unique)
        
        # Deduce sample rate from unique changes over target window
        true_sample_rate = unique_count / total_duration if total_duration > 0 else 0.0
        
        info_lines.append(f"\n● {conf['title']} Performance Profile:")
        info_lines.append(f"  True Hardware Sample Rate (ODR): {true_sample_rate:.1f} Hz (Based on {unique_count} data flips)")
        
        for f in fields:
            f_min = df[f].min()
            f_max = df[f].max()
            axis_lbl = f.split('_')[-1].upper()
            info_lines.append(f"  Range [{axis_lbl}-Axis]: Min = {f_min:11.4f} {unit} | Max = {f_max:11.4f} {unit}")
            
    info_lines.append("\n" + "=" * 60)
    
    # Save the Info Text File Artifact
    with open(info_out_path, "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(info_lines))
        
    # Automatically spawn a desktop window showing the text report
    try:
        if sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', str(info_out_path)])
        elif sys.platform == 'darkwin' or sys.platform == 'darwin':  # macOS fallback
            subprocess.Popen(['open', str(info_out_path)])
        elif sys.platform == 'win32':  # Windows fallback
            os.startfile(info_out_path)
        print(f"[INFO] Launched system viewer for: {info_out_path.name}")
    except Exception as e:
        print(f"[WARNING] Could not automatically open text file: {e}")

    # 3. Handle Matplotlib Visualization Setup
    num_plots = len(active_sensors)

        
    # 3. Handle Matplotlib Visualization Setup
    num_plots = len(active_sensors)
    fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 3 * num_plots + 1))
    if num_plots == 1:
        axes = [axes]
        
    fig.suptitle(f"BNO085 Dynamic Telemetry Data Plot: {file_path.name}", y=0.96, fontweight="bold")
    colors = ["#ff5555", "#55ff55", "#5555ff"]
    
    for idx, s_name in enumerate(active_sensors):
        ax = axes[idx]
        conf = sensor_configs[s_name]
        
        for i, f in enumerate(conf['fields']):
            axis_lbl = f.split('_')[-1].upper()
            ax.plot(df['time_seconds'], df[f], color=colors[i], label=axis_lbl if idx == 0 else "")
            
        ax.set_ylabel(f"{conf['title']} ({conf['unit']})")
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True)
            
    axes[-1].set_xlabel("Time elapsed (seconds)")
    plt.tight_layout()
    
    # Save Image Plot Artifact
    plt.savefig(plot_out_path, dpi=200)
    print(f"[SUCCESS] Artifact saved: {info_out_path.name}")
    print(f"[SUCCESS] Artifact saved: {plot_out_path.name}")
    
    # Display view window to human monitor
    plt.show()

if __name__ == "__main__":
    analyze_data()
