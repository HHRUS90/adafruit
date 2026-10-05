#!/usr/bin/env python3
"""
bno085_data_analyzer.py – Parse BNO085 telemetry files, calculate true ODR
vs software loop speed, evaluate min/max metrics, and auto‑export artifacts.
Now reports **per‑sensor effective sample‑rate** for each measurement type.
"""

import sys
import argparse
import subprocess
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# --------------------------------------------------------------
# 1️⃣ USER SETTINGS
# --------------------------------------------------------------
TRANSIENT_PADDING_SEC = 0.200   # 200 ms padding before/after transient window


# ----------------------------------------------------------------------
# 2️⃣ Detect sensors + measurement groups from column names
# ----------------------------------------------------------------------
def detect_sensors(df: pd.DataFrame):
    """
    Returns:
        sensors           – dict{sensor_id: {measurement_type: [col,…]}}
        measurement_types – list of measurement types that appear (ordered accel‑gyro‑mag)
    Accepted patterns:
        accel_x , gyro_y , mag_z                     (no prefix)
        S1_accel_x , S2_gyro_z , …                  (optional prefix)
    """
    pattern = re.compile(
        r'(?:(?P<sensor>[^_]+)_)?'      # optional prefix (e.g. S1)
        r'(?P<type>accel|gyro|mag)_'    # measurement type
        r'(?P<axis>[xyz])$'             # axis letter
    )

    sensors = defaultdict(lambda: defaultdict(list))
    measurement_set = set()

    for col in df.columns:
        m = pattern.match(col)
        if not m:
            continue
        sensor_id = m.group('sensor') or "S1"   # default when no prefix
        meas_type = m.group('type')
        sensors[sensor_id][meas_type].append(col)
        measurement_set.add(meas_type)

    # deterministic order
    ordered = ["accel", "gyro", "mag"]
    measurement_types = [t for t in ordered if t in measurement_set]

    return dict(sensors), measurement_types


# ----------------------------------------------------------------------
# 3️⃣ Compute effective sample‑rate (distinct rows) for a list of fields
# ----------------------------------------------------------------------
def compute_sensor_rate(df: pd.DataFrame, fields: list, time_col: str = "time_seconds"):
    """Effective ODR (Hz) = distinct rows that contain data for the fields / total time."""
    if not fields:
        return 0.0
    have_data = df[fields].notna().any(axis=1)
    distinct = df.loc[have_data, fields].drop_duplicates()
    total_time = df[time_col].iloc[-1] - df[time_col].iloc[0]
    if total_time <= 0:
        return 0.0
    return len(distinct) / total_time


# ----------------------------------------------------------------------
# 4️⃣ Argument parsing (unchanged)
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze BNO085 logged telemetry data (supports up to 2 sensors)."
    )
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="Path to CSV/TXT log. If omitted, newest file in logs/ is used.")
    parser.add_argument("--transient", action="store_true",
                        help="Crop analysis to automatically detected transient window.")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Open the generated plot image (default: OFF).")
    parser.add_argument("--info", action="store_true", default=False,
                        help="Open the generated text report (default: OFF).")
    parser.add_argument("--print", dest="print_info", action="store_true",
                        default=False, help="Print the report to the terminal (default: OFF).")
    return parser.parse_args()


# ----------------------------------------------------------------------
# 5️⃣ File discovery (unchanged)
# ----------------------------------------------------------------------
def get_target_file(requested_file):
    logs_dir = Path.cwd() / "logs"

    if requested_file:
        return Path(requested_file)

    if not logs_dir.exists():
        print(f"[ERROR] No file specified and default directory does not exist: {logs_dir}")
        sys.exit(1)

    log_files = [p for p in logs_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in [".csv", ".txt", ".tsv"]]

    if not log_files:
        print(f"[ERROR] No log files (.csv, .txt, .tsv) found inside: {logs_dir}")
        sys.exit(1)

    newest_file = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Auto‑detected newest log file: {newest_file.name}")
    return newest_file


# ----------------------------------------------------------------------
# 6️⃣ Transient detection (unchanged – uses accel+gyro fields only)
# ----------------------------------------------------------------------
def find_transient_window_automated(df, active_fields, software_rate):
    baseline_rows = int(1.5 * software_rate)
    if baseline_rows >= len(df) or baseline_rows < 5:
        baseline_rows = min(50, len(df) // 5)

    df_base = df.iloc[:baseline_rows]
    means = {f: df_base[f].mean() for f in active_fields}

    thresholds = {}
    for f in active_fields:
        if "accel" in f:
            thresholds[f] = 0.15   # m/s² (~15 mg)
        elif "gyro" in f:
            thresholds[f] = 0.05   # rad/s
        else:
            thresholds[f] = 1.0    # magnetometer fallback

    transient_active = pd.Series(False, index=df.index)
    for f in active_fields:
        deviation = (df[f] - means[f]).abs()
        transient_active |= (deviation >= thresholds[f])

    breach_indices = df.index[transient_active]
    if breach_indices.empty:
        return None, None

    start_idx = breach_indices[0]
    start_time = df.loc[start_idx, 'time_seconds']

    cooldown_frames = int(0.75 * software_rate)
    end_time = df['time_seconds'].iloc[-1]

    for idx in range(start_idx, len(df) - cooldown_frames):
        window = transient_active.iloc[idx : idx + cooldown_frames]
        if not window.any():
            end_time = df['time_seconds'].iloc[idx]
            break

    return start_time, end_time


# ----------------------------------------------------------------------
# 7️⃣ Main analysis routine – now includes per‑sensor ODR for each measurement
# ----------------------------------------------------------------------
def analyze_data():
    args = parse_args()
    file_path = get_target_file(args.file)

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print(f"[INFO] Reading file: {file_path.name}")
    delimiter = "\t" if file_path.suffix.lower() in [".txt", ".tsv"] else ","
    df = pd.read_csv(file_path, sep=delimiter)

    # --------------------------------------------------------------
    # 7.1 Timestamp handling
    # --------------------------------------------------------------
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df['time_seconds'] = (df['timestamp_dt'] - df['timestamp_dt'].iloc[0]).dt.total_seconds()

    total_duration = df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]
    total_rows = len(df)
    software_loop_rate = total_rows / total_duration if total_duration > 0 else 0.0

    # --------------------------------------------------------------
    # 7.2 Detect sensors + measurement types
    # --------------------------------------------------------------
    sensors, measurement_types = detect_sensors(df)
    if not measurement_types:
        print("[ERROR] No recognizable BNO085 measurement columns found.")
        sys.exit(1)

    num_sensors = len(sensors)
    sensor_id_list = sorted(sensors.keys())   # deterministic order, e.g. ['S1', 'S2']

    # Quick lookup column → sensor (used for the range‑line suffixes)
    col_to_sensor = {}
    for sid, meas_dict in sensors.items():
        for mtype, cols in meas_dict.items():
            for c in cols:
                col_to_sensor[c] = sid

    # --------------------------------------------------------------
    # 7.3 Fields for transient detection (accel + gyro from every sensor)
    # --------------------------------------------------------------
    active_fields = []
    for sid in sensor_id_list:
        for mtype in ("accel", "gyro"):
            active_fields.extend(sensors[sid].get(mtype, []))

    # --------------------------------------------------------------
    # 7.4 Transient window handling (unchanged)
    # --------------------------------------------------------------
    transient_detected = False
    t_start, t_end = None, None
    plot_start, plot_end = 0.0, total_duration

    if args.transient:
        t_start, t_end = find_transient_window_automated(df, active_fields, software_loop_rate)
        if t_start is not None and t_end is not None:
            transient_detected = True
            transient_duration = t_end - t_start
            plot_start = max(0.0, t_start - TRANSIENT_PADDING_SEC)
            plot_end = min(total_duration, t_end + TRANSIENT_PADDING_SEC)

            df_cropped = df[(df['time_seconds'] >= plot_start) &
                            (df['time_seconds'] <= plot_end)].copy()
        else:
            print("[WARNING] No transient burst found – analysing full file.")
            df_cropped = df.copy()
    else:
        df_cropped = df.copy()

    # --------------------------------------------------------------
    # 7.5 Output directory & filenames (always written)
    # --------------------------------------------------------------
    output_dir = Path.cwd() / "analysis_output"
    output_dir.mkdir(exist_ok=True)

    base_name = file_path.stem
    plot_out_path = output_dir / f"{base_name}_plot.png"
    info_out_path = output_dir / f"{base_name}_info.txt"

    # --------------------------------------------------------------
    # 7.6 Build the textual report
    # --------------------------------------------------------------
    info_lines = []
    info_lines.append("=" * 60)
    info_lines.append(f" BNO085 TELEMETRY ANALYSIS REPORT: {file_path.name}")
    info_lines.append("=" * 60)

    # ----- Header -------------------------------------------------
    info_lines.append(f"File Target:       {file_path.resolve()}")
    info_lines.append(f"Sensors detected:  {num_sensors} ({', '.join(sensor_id_list)})")
    info_lines.append(
        "Measurements present: "
        f"{', '.join([mt.title() for mt in measurement_types])}"
    )
    info_lines.append(f"Total Log Records: {total_rows} frames")
    info_lines.append(f"Total Duration:    {total_duration:.3f} seconds")
    info_lines.append(f"Software Logging Speed (Loop Rate): {software_loop_rate:.1f} Hz")

    # ----- Transient info -----------------------------------------
    if args.transient:
        info_lines.append("Transient Mode:    ENABLED (Physics‑Driven Noise Floor Automation)")
        if transient_detected:
            info_lines.append(f"Transient Start:   {t_start:.3f} seconds")
            info_lines.append(f"Transient End:     {t_end:.3f} seconds")
            info_lines.append(f"Transient Length:  {transient_duration:.3f} seconds")
            info_lines.append(f"Cropped Plot Range: {plot_start:.3f}s to {plot_end:.3f}s")
        else:
            info_lines.append("Transient Length:  NOT DETECTED (Signal stayed completely inside noise floor)")
    else:
        info_lines.append("Transient Mode:    DISABLED (Full file window analyzed)")

    info_lines.append("\n" + "-" * 60)
    info_lines.append(" SENSOR FREQUENCY & METRICS BREAKDOWN (CROP SCOPE)")
    info_lines.append("-" * 60)

    # ----- Per‑measurement block (Accel / Gyro / Mag) ---------------
    for mtype in measurement_types:                     # ordered accel‑gyro‑mag
        # Collect all columns of this measurement across every sensor
        all_fields = []
        for sid in sensor_id_list:
            all_fields.extend(sensors[sid].get(mtype, []))

        info_lines.append(f"\n● {mtype.title()} Performance Profile:")

        # ---- Per‑sensor effective ODR for THIS measurement ----
        for sid in sensor_id_list:
            sensor_fields = sensors[sid].get(mtype, [])
            rate = compute_sensor_rate(df, sensor_fields, time_col="time_seconds")
            info_lines.append(f"  Effective Sample Rate {sid} (ODR): {rate:.1f} Hz")

        # ---- Min / Max for each axis, per sensor ---------------
        unit = {"accel": "m/s²", "gyro": "rad/s", "mag": "μT"}[mtype]
        for f in all_fields:
            axis_lbl = f.split('_')[-1].upper()
            sensor_tag = col_to_sensor.get(f, "S?")   # safety fallback
            f_min = df_cropped[f].min()
            f_max = df_cropped[f].max()
            info_lines.append(
                f"  Range [{axis_lbl}-Axis] ( {sensor_tag} ): "
                f"Min = {f_min:11.4f} {unit} | Max = {f_max:11.4f} {unit}"
            )

    info_lines.append("\n" + "=" * 60)
    report_text = "\n".join(info_lines)

    # --------------------------------------------------------------
    # 7.7 Write artifacts (always written)
    # --------------------------------------------------------------
    with open(info_out_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(report_text)
    print(f"[SUCCESS] Artifact saved: {info_out_path.name}")

    # --------------------------------------------------------------
    # 7.8 Plot generation – one subplot per measurement type,
    #      multiple sensor lines per subplot
    # --------------------------------------------------------------
    num_plots = len(measurement_types)
    fig, axes = plt.subplots(num_plots, 1, sharex=True,
                             figsize=(10, 3 * num_plots + 1))
    if num_plots == 1:
        axes = [axes]

    fig.suptitle(f"BNO085 Telemetry Plot: {file_path.name}",
                 y=0.96, fontweight="bold")
    sensor_colors = ["#ff5555", "#55ff55", "#5555ff", "#ffaa00"]  # extend if needed

    for plot_idx, mtype in enumerate(measurement_types):
        ax = axes[plot_idx]
        ax.set_ylabel(f"{mtype.title()} ({{'accel':'m/s²','gyro':'rad/s','mag':'μT'}[mtype]})")
        ax.set_xlim(plot_start, plot_end)
        ax.grid(True, linestyle="--", alpha=0.6)

        for s_idx, sid in enumerate(sensor_id_list):
            fields = sensors[sid].get(mtype, [])
            if not fields:
                continue
            for f in fields:
                axis_lbl = f.split('_')[-1].upper()
                label = f"{sid} {axis_lbl}" if len(sensor_id_list) > 1 else axis_lbl
                ax.plot(df_cropped['time_seconds'], df_cropped[f],
                        color=sensor_colors[s_idx % len(sensor_colors)],
                        label=label)

        if transient_detected:
            ax.axvspan(t_start, t_end, color="#9b59b6", alpha=0.15,
                       label="Transient Active")
            ax.axvline(t_start, color="purple", linestyle=":", alpha=0.8,
                       label="Transient Start")
            ax.axvline(t_end, color="darkorange", linestyle=":", alpha=0.8,
                       label="Transient End")

        if plot_idx == 0:           # legend only on the first subplot
            ax.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("Time elapsed (seconds)")
    plt.tight_layout()
    plt.savefig(plot_out_path, dpi=200)
    print(f"[SUCCESS] Artifact saved: {plot_out_path.name}")

    # --------------------------------------------------------------
    # 7.9 Conditional opening of artifacts (flags)
    # --------------------------------------------------------------
    if args.info:
        try:
            if sys.platform.startswith('linux'):
                subprocess.Popen(['xdg-open', str(info_out_path)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(info_out_path)])
            elif sys.platform == 'win32':
                os.startfile(info_out_path)
        except Exception as e:
            print(f"[WARNING] Could not automatically open text file: {e}")

    if args.plot:
        plt.show()
        try:
            if sys.platform.startswith('linux'):
                subprocess.Popen(['xdg-open', str(plot_out_path)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(plot_out_path)])
            elif sys.platform == 'win32':
                os.startfile(plot_out_path)
        except Exception as e:
            print(f"[WARNING] Could not automatically open PNG file: {e}")

    # --------------------------------------------------------------
    # 7.10 Print report to terminal if requested
    # --------------------------------------------------------------
    if args.print_info:
        print("\n=== BNO085 ANALYSIS REPORT ===\n")
        print(report_text)


if __name__ == "__main__":
    analyze_data()
