"""
Ring Spinning Frame Digital Twin — bearing condition monitoring prototype
---------------------------------------------------------------------------
GUI: Tkinter + Matplotlib
Simulation: synthetic, physics-inspired signals for algorithm development.
It is NOT a validated model of any particular manufacturer's ring frame or bearing.
Install dependencies: pip install numpy pandas matplotlib
Run: python ring_spinning_digital_twin.py
"""

import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


APP_TITLE = "Ring Spinning Frame | Bearing-Wear Digital Twin"


class RingSpinningDigitalTwin:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1440x900")
        self.root.minsize(1120, 720)

        self.last_df = None
        self.last_summary = None
        self.last_result = None
        self.vars = {}
        self._build_ui()
        self._set_defaults()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 9))

        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="RING SPINNING FRAME DIGITAL TWIN",
                  style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Synthetic bearing-condition simulation • local processing",
                  style="Small.TLabel").pack(side="right")

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left_outer = ttk.Frame(main, width=350)
        main.add(left_outer, weight=0)
        right = ttk.Frame(main)
        main.add(right, weight=1)

        # Scrollable controls panel
        self.control_canvas = tk.Canvas(left_outer, highlightthickness=0, width=350)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical",
                                  command=self.control_canvas.yview)
        self.control_frame = ttk.Frame(self.control_canvas, padding=8)
        self.control_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all"))
        )
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=scrollbar.set)
        self.control_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.control_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._add_section("Machine operating conditions")
        self._add_entry("rpm", "Spindle speed (RPM)", "18000")
        self._add_entry("load_pct", "Estimated load (%)", "65")
        self._add_entry("duration_s", "Vibration capture duration (s)", "2.0")
        self._add_combo("sample_rate", "Vibration sampling rate (Hz)",
                        ["2560", "6400", "12800", "25600"], "12800")
        self._add_entry("ambient_c", "Ambient temperature (°C)", "30")
        self._add_entry("initial_temp_c", "Initial bearing-housing temp (°C)", "32")
        self._add_entry("thermal_duration_s", "Thermal simulation horizon (s)", "600")

        self._add_section("Bearing geometry (approximate)")
        self._add_entry("rolling_elements", "Number of rolling elements", "8")
        self._add_entry("ball_d_mm", "Rolling-element diameter (mm)", "8.0")
        self._add_entry("pitch_d_mm", "Bearing pitch diameter (mm)", "40.0")
        self._add_entry("contact_angle_deg", "Contact angle (degrees)", "0")
        self._add_entry("resonance_hz", "Impact resonance frequency (Hz)", "2500")

        self._add_section("Fault and mechanical condition")
        self._add_combo("fault_type", "Fault type",
                        ["Healthy", "Outer-race defect", "Inner-race defect",
                         "Rolling-element defect", "Cage defect", "Imbalance",
                         "Misalignment", "Lubrication degradation"], "Outer-race defect")
        self._add_entry("fault_severity", "Fault severity (0–100%)", "35")
        self._add_entry("imbalance_mg", "Equivalent unbalance mass (mg)", "8")
        self._add_entry("eccentricity_mm", "Unbalance eccentricity (mm)", "0.05")
        self._add_entry("misalignment_pct", "Misalignment level (0–100%)", "15")
        self._add_entry("lubrication_pct", "Lubrication condition (100=good)", "85")
        self._add_entry("sensor_noise", "Sensor noise standard deviation (m/s²)", "0.08")
        self._add_entry("baseline_vibration", "Baseline 1× vibration amplitude (m/s²)", "0.35")

        self._add_section("Thermal and electrical proxy model")
        self._add_entry("thermal_tau_s", "Thermal time constant (s)", "180")
        self._add_entry("base_heat_w", "Baseline heat input (W, proxy)", "5")
        self._add_entry("heat_loss_w_k", "Heat loss coefficient (W/K)", "0.8")
        self._add_entry("fault_heat_w", "Extra fault heat at 100% (W)", "10")
        self._add_entry("motor_base_a", "Baseline motor current proxy (A)", "12")
        self._add_entry("current_load_gain", "Current increase at 100% load (A)", "8")
        self._add_entry("random_seed", "Random seed (repeatable)", "42")

        buttons = ttk.Frame(self.control_frame, padding=(0, 10))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="RUN DIGITAL TWIN", command=self.run_simulation).pack(
            fill="x", pady=3)
        ttk.Button(buttons, text="Reset defaults", command=self._set_defaults).pack(
            fill="x", pady=3)
        ttk.Button(buttons, text="Export sensor data CSV", command=self.export_data).pack(
            fill="x", pady=3)
        ttk.Button(buttons, text="Export summary CSV", command=self.export_summary).pack(
            fill="x", pady=3)

        note = ("Model note: all sensor signals are simulated. Values are not measured "
                "from a real mill. Bearing geometry and fault amplitudes must be calibrated "
                "against the actual machine before making engineering decisions.")
        ttk.Label(self.control_frame, text=note, wraplength=300,
                  foreground="#8a4b08", justify="left").pack(fill="x", pady=(4, 12))

        # Right-side tabs
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        self.dashboard_tab = ttk.Frame(self.notebook, padding=8)
        self.features_tab = ttk.Frame(self.notebook, padding=8)
        self.data_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.features_tab, text="Features & Diagnosis")
        self.notebook.add(self.data_tab, text="Data Preview")

        self.metric_frame = ttk.Frame(self.dashboard_tab)
        self.metric_frame.pack(fill="x", pady=(0, 8))
        self.metric_labels = {}
        for i, (key, title) in enumerate([
            ("condition", "Estimated condition"),
            ("rms", "Vibration RMS"),
            ("temp", "Final housing temp"),
            ("rpm", "Rotational frequency"),
            ("current", "Current proxy"),
        ]):
            card = ttk.LabelFrame(self.metric_frame, text=title, padding=8)
            card.grid(row=0, column=i, sticky="nsew", padx=3)
            self.metric_frame.columnconfigure(i, weight=1)
            lab = ttk.Label(card, text="—", style="Metric.TLabel", wraplength=150)
            lab.pack()
            self.metric_labels[key] = lab

        self.fig = Figure(figsize=(10, 7), dpi=90, constrained_layout=True)
        self.ax_time = self.fig.add_subplot(221)
        self.ax_fft = self.fig.add_subplot(222)
        self.ax_temp = self.fig.add_subplot(223)
        self.ax_features = self.fig.add_subplot(224)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.dashboard_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = ttk.Frame(self.dashboard_tab)
        toolbar_frame.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, toolbar_frame).update()

        self.features_text = tk.Text(self.features_tab, wrap="word", height=25,
                                      font=("Consolas", 10))
        self.features_text.pack(fill="both", expand=True)
        self.features_text.insert("1.0", "Run the digital twin to view calculated features and diagnostic notes.")
        self.features_text.configure(state="disabled")

        data_toolbar = ttk.Frame(self.data_tab)
        data_toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(data_toolbar, text="Preview of simulated sensor samples (first 500 rows)").pack(side="left")
        self.data_text = tk.Text(self.data_tab, wrap="none", font=("Consolas", 9))
        data_scroll_y = ttk.Scrollbar(self.data_tab, orient="vertical", command=self.data_text.yview)
        data_scroll_x = ttk.Scrollbar(self.data_tab, orient="horizontal", command=self.data_text.xview)
        self.data_text.configure(yscrollcommand=data_scroll_y.set, xscrollcommand=data_scroll_x.set)
        self.data_text.pack(side="top", fill="both", expand=True)
        data_scroll_y.pack(side="right", fill="y")
        data_scroll_x.pack(side="bottom", fill="x")

        self.status = tk.StringVar(value="Ready. Adjust inputs and run the simulation.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w",
                  padding=(8, 4)).pack(fill="x", side="bottom")

    def _on_mousewheel(self, event):
        try:
            self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _add_section(self, title):
        ttk.Label(self.control_frame, text=title, style="Section.TLabel").pack(
            anchor="w", pady=(10, 4))

    def _add_entry(self, key, label, default):
        row = ttk.Frame(self.control_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, wraplength=190).pack(side="left", fill="x", expand=True)
        var = tk.StringVar(value=default)
        self.vars[key] = var
        ttk.Entry(row, textvariable=var, width=10).pack(side="right")

    def _add_combo(self, key, label, values, default):
        row = ttk.Frame(self.control_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, wraplength=190).pack(side="left", fill="x", expand=True)
        var = tk.StringVar(value=default)
        self.vars[key] = var
        ttk.Combobox(row, textvariable=var, values=values, width=12,
                     state="readonly").pack(side="right")

    def _set_defaults(self):
        defaults = {
            "rpm": "18000", "load_pct": "65", "duration_s": "2.0",
            "sample_rate": "12800", "ambient_c": "30", "initial_temp_c": "32",
            "thermal_duration_s": "600", "rolling_elements": "8", "ball_d_mm": "8.0",
            "pitch_d_mm": "40.0", "contact_angle_deg": "0", "resonance_hz": "2500",
            "fault_type": "Outer-race defect", "fault_severity": "35",
            "imbalance_mg": "8", "eccentricity_mm": "0.05", "misalignment_pct": "15",
            "lubrication_pct": "85", "sensor_noise": "0.08",
            "baseline_vibration": "0.35", "thermal_tau_s": "180", "base_heat_w": "5",
            "heat_loss_w_k": "0.8", "fault_heat_w": "10", "motor_base_a": "12",
            "current_load_gain": "8", "random_seed": "42"
        }
        for key, value in defaults.items():
            self.vars[key].set(value)

    def _read_inputs(self):
        p = {}
        try:
            for key, var in self.vars.items():
                if key in ("fault_type", "sample_rate"):
                    p[key] = var.get()
                else:
                    p[key] = float(var.get())
        except ValueError as exc:
            raise ValueError("One or more input fields are not valid numbers.") from exc

        limits = {
            "rpm": (100, 40000), "load_pct": (0, 100),
            "duration_s": (0.25, 10), "ambient_c": (-10, 60),
            "initial_temp_c": (-10, 150), "thermal_duration_s": (10, 3600),
            "rolling_elements": (3, 30), "ball_d_mm": (1, 30),
            "pitch_d_mm": (5, 200), "contact_angle_deg": (0, 45),
            "resonance_hz": (100, 10000), "fault_severity": (0, 100),
            "imbalance_mg": (0, 1000), "eccentricity_mm": (0, 2),
            "misalignment_pct": (0, 100), "lubrication_pct": (0, 100),
            "sensor_noise": (0, 10), "baseline_vibration": (0.001, 20),
            "thermal_tau_s": (1, 10000), "base_heat_w": (0, 500),
            "heat_loss_w_k": (0.01, 100), "fault_heat_w": (0, 500),
            "motor_base_a": (0, 1000), "current_load_gain": (0, 1000),
            "random_seed": (0, 2147483647)
        }
        for key, (low, high) in limits.items():
            if not low <= p[key] <= high:
                raise ValueError(f"{key}: enter a value from {low} to {high}.")
        if p["pitch_d_mm"] <= p["ball_d_mm"]:
            raise ValueError("Pitch diameter must be greater than rolling-element diameter.")
        if p["rolling_elements"] != int(p["rolling_elements"]):
            raise ValueError("Number of rolling elements must be a whole number.")
        p["rolling_elements"] = int(p["rolling_elements"])
        p["sample_rate"] = int(p["sample_rate"])
        if p["sample_rate"] * p["duration_s"] > 200000:
            raise ValueError("Capture exceeds 200,000 samples. Reduce duration or sample rate.")
        return p

    @staticmethod
    def _kurtosis(x):
        x = np.asarray(x, dtype=float)
        centered = x - np.mean(x)
        variance = np.mean(centered ** 2)
        if variance < 1e-18:
            return 0.0
        return float(np.mean(centered ** 4) / (variance ** 2))

    @staticmethod
    def _rms(x):
        return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

    def _simulate(self, p):
        rng = np.random.default_rng(int(p["random_seed"]))
        fs = p["sample_rate"]
        duration = p["duration_s"]
        n_samples = max(256, int(fs * duration))
        t = np.arange(n_samples) / fs
        rpm = p["rpm"]
        fr = rpm / 60.0
        omega = 2 * np.pi * fr
        severity = p["fault_severity"] / 100.0
        load = p["load_pct"] / 100.0
        lube_bad = 1.0 - p["lubrication_pct"] / 100.0
        misalign = p["misalignment_pct"] / 100.0
        fault_type = p["fault_type"]

        # Approximate rolling-element bearing characteristic frequencies.
        d = p["ball_d_mm"] / p["pitch_d_mm"]
        beta = math.radians(p["contact_angle_deg"])
        term = max(0.0, 1.0 - d * math.cos(beta))
        ftf = 0.5 * fr * (1.0 - d * math.cos(beta))
        bpfo = 0.5 * p["rolling_elements"] * fr * (1.0 - d * math.cos(beta))
        bpfi = 0.5 * p["rolling_elements"] * fr * (1.0 + d * math.cos(beta))
        bsf = (p["pitch_d_mm"] / (2 * p["ball_d_mm"])) * fr * (
            1.0 - (d * math.cos(beta)) ** 2)
        defect_freq = {
            "Outer-race defect": bpfo,
            "Inner-race defect": bpfi,
            "Rolling-element defect": bsf,
            "Cage defect": ftf
        }.get(fault_type, 0.0)

        # Base rotational vibration and harmonics; amplitudes are tunable proxies.
        base_amp = p["baseline_vibration"]
        x = base_amp * np.sin(2 * np.pi * fr * t)
        x += 0.18 * base_amp * np.sin(2 * np.pi * 2 * fr * t + 0.3)
        x += 0.08 * base_amp * np.sin(2 * np.pi * 3 * fr * t + 0.7)

        # Imbalance force proxy scales with speed squared and user-specified unbalance.
        # It is normalized to keep the parameter useful for a synthetic signal.
        imbalance_amp = (p["imbalance_mg"] / 8.0) * (p["eccentricity_mm"] / 0.05) \
                        * (rpm / 18000.0) ** 2 * 0.18
        if fault_type == "Imbalance":
            imbalance_amp += severity * 0.9 * (rpm / 18000.0) ** 2
        x += imbalance_amp * np.sin(2 * np.pi * fr * t + 0.8)

        # Misalignment proxy: increased 2x component.
        misalignment_amp = misalign * 0.8
        if fault_type == "Misalignment":
            misalignment_amp += severity * 0.8
        x += misalignment_amp * np.sin(2 * np.pi * 2 * fr * t + 1.1)

        # Fault impacts: periodic decaying resonant bursts, a simplified envelope model.
        if defect_freq > 0 and (fault_type in (
                "Outer-race defect", "Inner-race defect",
                "Rolling-element defect", "Cage defect")) and severity > 0:
            impact_interval = 1.0 / max(defect_freq, 1e-6)
            event_times = np.arange(0.0, duration, impact_interval)
            resonance = min(p["resonance_hz"], 0.42 * fs)
            burst_len = max(8, int(0.018 * fs))
            burst_t = np.arange(burst_len) / fs
            burst = np.exp(-burst_t * 230) * np.sin(2 * np.pi * resonance * burst_t)
            impact_amp = severity * (0.5 + 1.2 * load) * (1.0 + 1.5 * lube_bad)
            for event_time in event_times:
                start = int(event_time * fs)
                end = min(n_samples, start + burst_len)
                if start < n_samples:
                    x[start:end] += impact_amp * burst[:end - start]

        # Lubrication degradation: adds broadband/high-frequency content and impacts.
        if fault_type == "Lubrication degradation":
            x += severity * (0.25 + 0.8 * lube_bad) * rng.normal(size=n_samples)
        # A poor lubrication state can influence all fault cases.
        x += lube_bad * 0.12 * rng.normal(size=n_samples)
        x += rng.normal(0.0, p["sensor_noise"], size=n_samples)

        # Three-axis synthetic channels: correlated but not identical.
        y = 0.72 * x + 0.12 * base_amp * np.sin(2 * np.pi * fr * t + 1.4)
        y += rng.normal(0, p["sensor_noise"], size=n_samples)
        z = 0.48 * x + 0.10 * base_amp * np.sin(2 * np.pi * 2 * fr * t + 0.4)
        z += rng.normal(0, p["sensor_noise"], size=n_samples)

        # Temperature: first-order lumped thermal model, run at 1-second steps.
        # The chosen thermal coefficients are model proxies, not a manufacturer's data.
        thermal_dt = 1.0
        thermal_n = int(max(10, p["thermal_duration_s"] / thermal_dt)) + 1
        tt = np.arange(thermal_n) * thermal_dt
        temp = np.zeros(thermal_n)
        temp[0] = p["initial_temp_c"]
        fault_heat = p["fault_heat_w"] * severity
        if fault_type == "Lubrication degradation":
            fault_heat += 8.0 * lube_bad
        fault_heat += 4.0 * lube_bad
        speed_factor = max(0.1, rpm / 18000.0)
        heat_in = p["base_heat_w"] + 4.0 * load + fault_heat * speed_factor
        # Effective cooling coefficient; temperature rise approaches heat_in / heat_loss.
        k_loss = p["heat_loss_w_k"]
        for i in range(1, thermal_n):
            dTdt = (heat_in - k_loss * (temp[i - 1] - p["ambient_c"])) / max(
                p["thermal_tau_s"], 1.0)
            temp[i] = temp[i - 1] + dTdt * thermal_dt

        # Current proxy is not a measured electrical model.
        current = p["motor_base_a"] + p["current_load_gain"] * load
        current += 0.04 * rpm / 1000.0
        current += 1.8 * severity + 1.2 * lube_bad
        current += rng.normal(0, 0.03, size=n_samples)

        # Simple features.
        rms = self._rms(x)
        peak = float(np.max(np.abs(x)))
        peak_to_peak = float(np.ptp(x))
        crest = peak / max(rms, 1e-12)
        kurt = self._kurtosis(x)
        fft_freq = np.fft.rfftfreq(n_samples, d=1 / fs)
        fft_mag = np.abs(np.fft.rfft(x - np.mean(x))) * (2.0 / n_samples)
        # Limit plotted spectrum to 0-5 kHz or Nyquist, whichever is lower.
        fmax = min(5000.0, fs / 2)
        fft_mask = fft_freq <= fmax

        # A transparent heuristic, not a trained or safety-certified classifier.
        score = 0.0
        if fault_type != "Healthy":
            score += 35.0 * severity
        score += min(25.0, max(0.0, (rms - 0.35) * 12.0))
        score += min(15.0, max(0.0, (kurt - 3.0) * 2.0))
        score += 12.0 * lube_bad
        score += 8.0 * misalign
        score = min(100.0, score)
        if score < 18:
            diagnosis = "Healthy-like"
        elif score < 45:
            diagnosis = "Watch / possible degradation"
        elif score < 70:
            diagnosis = "Abnormal — inspect"
        else:
            diagnosis = "Severe synthetic fault indication"

        df = pd.DataFrame({
            "time_s": t,
            "rpm": np.full(n_samples, rpm),
            "accel_x_m_s2": x,
            "accel_y_m_s2": y,
            "accel_z_m_s2": z,
            "bearing_temp_proxy_c": np.interp(t, tt, temp),
            "ambient_temp_c": np.full(n_samples, p["ambient_c"]),
            "motor_current_proxy_a": current,
            "fault_type_input": np.full(n_samples, fault_type),
            "fault_severity_input_pct": np.full(n_samples, p["fault_severity"]),
        })

        features = {
            "Vibration RMS X (m/s²)": rms,
            "Peak absolute acceleration X (m/s²)": peak,
            "Peak-to-peak X (m/s²)": peak_to_peak,
            "Crest factor X": crest,
            "Kurtosis X (raw, Gaussian≈3)": kurt,
            "Rotational frequency (Hz)": fr,
            "FTF estimate (Hz)": ftf,
            "BPFO estimate (Hz)": bpfo,
            "BPFI estimate (Hz)": bpfi,
            "BSF estimate (Hz)": bsf,
            "Selected defect frequency (Hz)": defect_freq,
            "Final housing temperature proxy (°C)": float(temp[-1]),
            "Maximum thermal-model temperature (°C)": float(np.max(temp)),
            "Temperature rise from initial (°C)": float(temp[-1] - temp[0]),
            "Mean motor-current proxy (A)": float(np.mean(current)),
            "Heuristic condition score (0–100)": score,
            "Fault input": fault_type,
            "Fault severity input (%)": p["fault_severity"],
            "Diagnostic label": diagnosis,
        }
        result = {
            "t": t, "x": x, "y": y, "z": z, "tt": tt, "temp": temp,
            "fft_freq": fft_freq[fft_mask], "fft_mag": fft_mag[fft_mask],
            "features": features, "df": df, "params": p,
            "fault_freqs": {"FTF": ftf, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf},
            "diagnosis": diagnosis, "score": score,
        }
        return result

    def run_simulation(self):
        try:
            p = self._read_inputs()
            self.status.set("Running simulation…")
            self.root.update_idletasks()
            result = self._simulate(p)
            self.last_result = result
            self.last_df = result["df"]
            self.last_summary = result["features"]
            self._update_dashboard(result)
            self._update_features(result)
            self._update_data_preview(result["df"])
            self.status.set(
                f"Simulation complete • {len(result['df']):,} samples • "
                f"{result['diagnosis']} • synthetic data only")
            self.notebook.select(self.dashboard_tab)
        except Exception as exc:
            self.status.set("Simulation failed. Check the input values.")
            messagebox.showerror("Digital twin input / simulation error", str(exc))

    def _update_dashboard(self, r):
        f = r["features"]
        self.metric_labels["condition"].configure(text=f["Diagnostic label"])
        self.metric_labels["rms"].configure(text=f'{f["Vibration RMS X (m/s²)"]:.3f} m/s²')
        self.metric_labels["temp"].configure(
            text=f'{f["Final housing temperature proxy (°C)"]:.1f} °C')
        self.metric_labels["rpm"].configure(text=f'{f["Rotational frequency (Hz)"]:.1f} Hz')
        self.metric_labels["current"].configure(text=f'{f["Mean motor-current proxy (A)"]:.2f} A')

        self.ax_time.clear()
        self.ax_fft.clear()
        self.ax_temp.clear()
        self.ax_features.clear()

        t, x = r["t"], r["x"]
        display_n = min(len(t), 3000)
        self.ax_time.plot(t[:display_n], x[:display_n], linewidth=0.7)
        self.ax_time.set_title("Synthetic vibration — X axis")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Acceleration (m/s²)")
        self.ax_time.grid(True, alpha=0.25)

        self.ax_fft.plot(r["fft_freq"], r["fft_mag"], linewidth=0.8)
        self.ax_fft.set_title("Single-sided FFT magnitude")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Amplitude (m/s²)")
        self.ax_fft.grid(True, alpha=0.25)
        for name, freq in r["fault_freqs"].items():
            if freq > 0 and freq <= min(5000, r["params"]["sample_rate"] / 2):
                self.ax_fft.axvline(freq, linestyle="--", linewidth=0.8, alpha=0.6)
        self.ax_fft.set_xlim(0, min(5000, r["params"]["sample_rate"] / 2))

        self.ax_temp.plot(r["tt"], r["temp"], linewidth=1.5)
        self.ax_temp.axhline(r["params"]["ambient_c"], linestyle="--", linewidth=0.8)
        self.ax_temp.set_title("Lumped thermal-model proxy")
        self.ax_temp.set_xlabel("Thermal simulation time (s)")
        self.ax_temp.set_ylabel("Temperature (°C)")
        self.ax_temp.grid(True, alpha=0.25)

        names = ["RMS", "Peak", "Crest", "Kurtosis", "Temp rise"]
        vals = [
            r["features"]["Vibration RMS X (m/s²)"],
            r["features"]["Peak absolute acceleration X (m/s²)"],
            r["features"]["Crest factor X"],
            r["features"]["Kurtosis X (raw, Gaussian≈3)"],
            r["features"]["Temperature rise from initial (°C)"],
        ]
        self.ax_features.bar(names, vals)
        self.ax_features.set_title("Selected condition features (different units)")
        self.ax_features.set_ylabel("Value (units differ)")
        self.ax_features.tick_params(axis="x", labelrotation=15)
        self.ax_features.grid(True, axis="y", alpha=0.25)
        self.fig.suptitle(
            f"Fault input: {r['params']['fault_type']} | "
            f"Severity: {r['params']['fault_severity']:.0f}% | "
            f"Condition score: {r['score']:.1f}/100",
            fontsize=11)
        self.canvas.draw_idle()

    def _update_features(self, r):
        p = r["params"]
        lines = [
            "RING SPINNING FRAME — SIMULATION REPORT",
            "=" * 68,
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "IMPORTANT: Synthetic, physics-inspired model; not a calibrated machine model.",
            "",
            "OPERATING INPUTS",
            "-" * 68,
            f"Spindle speed: {p['rpm']:.1f} RPM",
            f"Rotational frequency: {r['features']['Rotational frequency (Hz)']:.3f} Hz",
            f"Load: {p['load_pct']:.1f}%",
            f"Vibration capture: {p['duration_s']:.2f} s at {p['sample_rate']} Hz",
            f"Ambient temperature: {p['ambient_c']:.1f} °C",
            f"Fault input: {p['fault_type']}",
            f"Fault severity input: {p['fault_severity']:.1f}%",
            f"Lubrication condition: {p['lubrication_pct']:.1f}% (100 = good)",
            "",
            "VIBRATION FEATURES",
            "-" * 68,
        ]
        for key in [
            "Vibration RMS X (m/s²)", "Peak absolute acceleration X (m/s²)",
            "Peak-to-peak X (m/s²)", "Crest factor X",
            "Kurtosis X (raw, Gaussian≈3)"
        ]:
            lines.append(f"{key}: {r['features'][key]:.5f}")
        lines += [
            "",
            "APPROXIMATE BEARING KINEMATIC FREQUENCIES",
            "-" * 68,
            f"FTF:  {r['fault_freqs']['FTF']:.3f} Hz",
            f"BPFO: {r['fault_freqs']['BPFO']:.3f} Hz",
            f"BPFI: {r['fault_freqs']['BPFI']:.3f} Hz",
            f"BSF:  {r['fault_freqs']['BSF']:.3f} Hz",
            f"Selected fault frequency: {r['features']['Selected defect frequency (Hz)']:.3f} Hz",
            "These are geometry-based estimates; real peaks can differ due to slip and resonances.",
            "",
            "THERMAL / CURRENT PROXIES",
            "-" * 68,
            f"Initial housing temperature: {p['initial_temp_c']:.2f} °C",
            f"Final housing temperature proxy: {r['features']['Final housing temperature proxy (°C)']:.2f} °C",
            f"Maximum thermal-model temperature: {r['features']['Maximum thermal-model temperature (°C)']:.2f} °C",
            f"Temperature rise: {r['features']['Temperature rise from initial (°C)']:.2f} °C",
            f"Mean motor-current proxy: {r['features']['Mean motor-current proxy (A)']:.3f} A",
            "",
            "HEURISTIC DIAGNOSTIC (NOT A TRAINED CLASSIFIER)",
            "-" * 68,
            f"Condition score: {r['score']:.1f}/100",
            f"Label: {r['diagnosis']}",
            "",
            "Interpretation:",
            "• The condition score is a transparent demonstration heuristic, not a validated probability.",
            "• A high score does not prove bearing damage; speed, load, mounting and noise affect features.",
            "• Calibrate with healthy machine data and verified fault examples before drawing conclusions.",
            "• Do not use this GUI as a safety-rated shutdown or personnel-protection system.",
        ]
        self.features_text.configure(state="normal")
        self.features_text.delete("1.0", "end")
        self.features_text.insert("1.0", "\n".join(lines))
        self.features_text.configure(state="disabled")

    def _update_data_preview(self, df):
        self.data_text.delete("1.0", "end")
        self.data_text.insert("1.0", df.head(500).to_string(index=False))

    def export_data(self):
        if self.last_df is None:
            messagebox.showinfo("No data", "Run the digital twin before exporting sensor data.")
            return
        path = filedialog.asksaveasfilename(
            title="Export simulated sensor data",
            defaultextension=".csv",
            initialfile="ring_frame_synthetic_sensor_data.csv",
            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.last_df.to_csv(path, index=False)
            messagebox.showinfo("Export complete", f"Sensor data exported to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_summary(self):
        if self.last_summary is None:
            messagebox.showinfo("No summary", "Run the digital twin before exporting the summary.")
            return
        path = filedialog.asksaveasfilename(
            title="Export simulation summary",
            defaultextension=".csv",
            initialfile="ring_frame_run_summary.csv",
            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            summary = pd.DataFrame(
                [{"parameter": k, "value": v} for k, v in self.last_summary.items()])
            summary.to_csv(path, index=False)
            messagebox.showinfo("Export complete", f"Summary exported to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))


def main():
    root = tk.Tk()
    app = RingSpinningDigitalTwin(root)
    root.mainloop()


if __name__ == "__main__":
    main()
