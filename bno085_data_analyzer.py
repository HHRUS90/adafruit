#!/usr/bin/env python3
"""
bno085_data_analyzer.py – Parse BNO085 telemetry files, calculate true ODR
vs software loop speed, evaluate min/max metrics, and auto-export artifacts.
Supports rolling window transient window slicing.
"""

import sys
import argparse
import subprocess
import os
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
    parser.add_argument(
        "--transient",
        action="store_true",
        help="Crop data analysis and plot layout to the specific transient window frame."
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Standard deviation multiplier threshold used to detect transient bounds (default: 3.0)."
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=3.0,
        help="Consecutive seconds data must rest within baseline limits to conclude transient (default: 3.0)."
    )
    return parser.parse_args()

def get_target_file(requested_file):
    """Determines the target file path, falling back to the newest file if none requested."""
    logs_dir = Path.cwd() / "logs"

    if requested_file:
        return Path(requested_file)
        
    if not logs_dir.exists():
        print(f"[ERROR] No file specified and default directory does not exist: {logs_dir}")
        sys.exit(1)
        
    log_files = [
        p for p in logs_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in [".csv", ".txt", ".tsv"]
    ]
    
    if not log_files:
        print(f"[ERROR] No log files (.csv, .txt, .tsv) found inside: {logs_dir}")
        sys.exit(1)
        
    newest_file = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Auto-detected newest log file: {newest_file.name}")
    return newest_file

def find_transient_window(df, active_fields, sigma_thresh, cooldown_seconds, software_rate):
    """
    Finds the transient bounds across all active sensor fields.
    Uses the initial 1.5 seconds as a stable baseline window.
    """
    # Use the first 1.5 seconds of data to compute baseline mean and std
    baseline_rows = int(1.5 * software_rate)
    if baseline_rows >= len(df) or baseline_rows < 5:
        baseline_rows = min(50, len(df) // 5)
        
    df_base = df.iloc[:baseline_rows]
    
    means = {f: df_base[f].mean() for f in active_fields}
    stds = {f: df_base[f].std() for f in active_fields}
    
    # Avoid zero division errors for stationary datasets
    for f in active_fields:
        if stds[f] < 1e-6:
            stds[f] = 1e-4

    # Calculate absolute deviations out of baseline bounds
    transient_active = pd.Series(False, index=df.index)
    for f in active_fields:
        deviation = (df[f] - means[f]).abs()
        transient_active |= (deviation >= (sigma_thresh * stds[f]))

    # Find the very first row where deviation breaches limits
    breach_indices = df.index[transient_active]
    if breach_indices.empty:
        return None, None

    start_idx = breach_indices[0]
    start_time = df.loc[start_idx, 'time_seconds']

    # Cooldown verification loop
    cooldown_frames = int(cooldown_seconds * software_rate)
    end_time = df['time_seconds'].iloc[-1]
    
    for idx in range(start_idx, len(df)):
        # Inspect a prospective sliding window of frames forward
        window = transient_active.iloc[idx : idx + cooldown_frames]
        
        # If the window is full length and contains ZERO deviations,
        # the transient ended exactly at the START of this calm window (the current idx)
        if len(window) >= cooldown_frames and not window.any():
            end_time = df['time_seconds'].iloc[idx]  # <-- FIX: Capture the drop moment, not the end of the wait
            break

    return start_time, end_time

def analyze_data():
    args = parse_args()
    file_path = get_target_file(args.file)
    
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
        
    print(f"[INFO] Reading file: {file_path.name}")
    delimiter = "\t" if file_path.suffix.lower() in [".txt", ".tsv"] else ","
    df = pd.read_csv(file_path, sep=delimiter)
    
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df['time_seconds'] = (df['timestamp_dt'] - df['timestamp_dt'].iloc[0]).dt.total_seconds()
    
    total_duration = df['time_seconds'].iloc[-1]
    total_rows = len(df)
    software_loop_rate = total_rows / total_duration if total_duration > 0 else 0.0
    
    sensor_configs = {
        "accel": {"fields": ["accel_x", "accel_y", "accel_z"], "unit": "m/s²", "title": "Accelerometer"},
        "gyro":  {"fields": ["gyro_x", "gyro_y", "gyro_z"],   "unit": "rad/s", "title": "Gyroscope"},
        "mag":   {"fields": ["mag_x", "mag_y", "mag_z"],      "unit": "μT",    "title": "Magnetometer"}
    }
    
    active_sensors = [s for s, conf in sensor_configs.items() if all(f in df.columns for f in conf['fields'])]
    if not active_sensors:
        print("[ERROR] No valid BNO085 sensor headers recognized in file.")
        sys.exit(1)
        
    # Collate active column names, excluding magnetometer from the trigger list
    all_active_fields = []
    for s_name in active_sensors:
        if s_name != "mag":  # <-- EXCLUDE MAGNETOMETER OVERLAY FROM TRIGGER MATH
            all_active_fields.extend(sensor_configs[s_name]['fields'])

    # Fallback to prevent crash if running ONLY on magnetometer data
    if not all_active_fields:
        for s_name in active_sensors:
            all_active_fields.extend(sensor_configs[s_name]['fields'])

    transient_detected = False
    t_start, t_end = None, None

    # Execute Slicing Window Calculations if requested
    if args.transient:
        t_start, t_end = find_transient_window(
            df, all_active_fields, args.sigma, args.cooldown, software_loop_rate
        )
        if t_start is not None and t_end is not None:
            transient_detected = True
            transient_duration = t_end - t_start
            # Expand analytical bounds (1 second before, 2 seconds after)
            plot_start = max(0.0, t_start - 1.0)
            plot_end = min(total_duration, t_end + 2.0)
            # Crop data framework down to slice bounds
            df_cropped = df[(df['time_seconds'] >= plot_start) & (df['time_seconds'] <= plot_end)].copy()
        else:
            print("[WARNING] Transient window not found with current settings. Defaulting to full duration.")
            df_cropped = df.copy()
    else:
        df_cropped = df.copy()

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
    info_lines.append(f"Software Logging Speed (Loop Rate): {software_loop_rate:.1f} Hz")
    
    if args.transient:
        info_lines.append(f"Transient Mode:    ENABLED (Sigma={args.sigma}, Cooldown={args.cooldown}s)")
        if transient_detected:
            info_lines.append(f"Transient Start:   {t_start:.3f} seconds")
            info_lines.append(f"Transient End:     {t_end:.3f} seconds")
            info_lines.append(f"Transient Length:  {transient_duration:.3f} seconds")
            info_lines.append(f"Cropped Plot Range: {plot_start:.3f}s to {plot_end:.3f}s")
        else:
            info_lines.append("Transient Length:  NOT DETECTED (No breach found Outside Baseline)")
    else:
        info_lines.append("Transient Mode:    DISABLED (Full file window analyzed)")
        
    info_lines.append("\n" + "-" * 60)
    info_lines.append(" SENSOR FREQUENCY & METRICS BREAKDOWN (CROP SCOPE)")
    info_lines.append("-" * 60)
    
    # 2. Evaluate Performance and Metrics across cropped frame
    for s_name in active_sensors:
        conf = sensor_configs[s_name]
        fields = conf['fields']
        unit = conf['unit']
        
        # Calculate true rate metrics over full duration, but ranges inside crop frame
        df_unique = df.drop_duplicates(subset=fields)
        true_sample_rate = len(df_unique) / total_duration if total_duration > 0 else 0.0
        
        info_lines.append(f"\n● {conf['title']} Performance Profile:")
        info_lines.append(f"  True Hardware Sample Rate (ODR): {true_sample_rate:.1f} Hz")
        
        for f in fields:
            f_min = df_cropped[f].min()
            f_max = df_cropped[f].max()
            axis_lbl = f.split('_')[-1].upper()
            info_lines.append(f"  Range [{axis_lbl}-Axis]: Min = {f_min:11.4f} {unit} | Max = {f_max:11.4f} {unit}")
            
    info_lines.append("\n" + "=" * 60)
    report_text = "\n".join(info_lines)
    
    with open(info_out_path, "w", encoding="utf-8") as text_file:
        text_file.write(report_text)
        
    # 3. Handle Matplotlib Visualization Setup
    num_plots = len(active_sensors)
    fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 3 * num_plots + 1))
    if num_plots == 1:
        axes = [axes]
        
    fig.suptitle(f"BNO085 Telemetry Plot: {file_path.name}", y=0.96, fontweight="bold")
    colors = ["#ff5555", "#55ff55", "#5555ff"]
    
    for idx, s_name in enumerate(active_sensors):
        ax = axes[idx]
        conf = sensor_configs[s_name]
        
        for i, f in enumerate(conf['fields']):
            axis_lbl = f.split('_')[-1].upper()
            ax.plot(df_cropped['time_seconds'], df_cropped[f], color=colors[i], label=axis_lbl if idx == 0 else "")
            
        ax.set_ylabel(f"{conf['title']} ({conf['unit']})")
        ax.set_xlim(plot_start, plot_end)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Draw lines AND apply a shaded background patch for the active transient window
        if transient_detected:
            # Shaded transparent background block spanning the transient runtime
            ax.axvspan(t_start, t_end, color="#9b59b6", alpha=0.15, label="Transient Active" if idx == 0 else "")
            
            # Boundary markers
            ax.axvline(t_start, color="purple", linestyle=":", alpha=0.8, label="Transient Start" if idx == 0 else "")
            ax.axvline(t_end, color="darkorange", linestyle=":", alpha=0.8, label="Transient End" if idx == 0 else "")
            
        if idx == 0:
            ax.legend(loc="upper right", frameon=True)

            
    axes[-1].set_xlabel("Time elapsed (seconds)")
    plt.tight_layout()
    
    # Save Image Plot Artifact
    plt.savefig(plot_out_path, dpi=200)
    print(f"[SUCCESS] Artifact saved: {info_out_path.name}")
    print(f"[SUCCESS] Artifact saved: {plot_out_path.name}\n")
    
    # Automatically spawn a desktop window showing the text report
    try:
        if sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', str(info_out_path)])
        elif sys.platform in ['darkwin', 'darwin']:
            subprocess.Popen(['open', str(info_out_path)])
        elif sys.platform == 'win32':
            os.startfile(info_out_path)
    except Exception as e:
        print(f"[WARNING] Could not automatically open text file: {e}")

    # Display view window to human monitor
    plt.show()
    
if __name__ == "__main__":
    analyze_data()
