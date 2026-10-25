import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import joblib
import pandas as pd

FEATURES = [
    "rpm", "load_pct", "ambient_temp_c", "sample_rate_hz", "sensor_noise",
    "vibration_rms_x", "vibration_peak_x", "vibration_peak_to_peak_x",
    "crest_factor_x", "kurtosis_x", "final_temperature_c",
    "max_temperature_c", "temperature_rise_c", "mean_current_proxy_a"
]
DEFAULTS = {
    "rpm": 18000, "load_pct": 70, "ambient_temp_c": 30,
    "sample_rate_hz": 1000, "sensor_noise": 0.02, "vibration_rms_x": 0.5,
    "vibration_peak_x": 1.5, "vibration_peak_to_peak_x": 2.5,
    "crest_factor_x": 3.0, "kurtosis_x": 3.2, "final_temperature_c": 45,
    "max_temperature_c": 47, "temperature_rise_c": 15, "mean_current_proxy_a": 1.8
}
LABELS = {
    "healthy": "Healthy", "imbalance": "Imbalance",
    "misalignment": "Misalignment", "early_bearing_fault": "Early bearing fault",
    "severe_bearing_fault": "Severe bearing fault"
}

class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("Ring Spinning Frame — AI Condition Monitor")
        root.geometry("900x760")
        root.minsize(760, 650)
        self.model = None
        self.model_features = FEATURES.copy()
        self.model_medians = {}
        self.model_path = tk.StringVar()
        self.status = tk.StringVar(value="Load your trained .joblib model to begin.")
        self.prediction = tk.StringVar(value="No prediction yet")
        self.confidence = tk.StringVar()
        self.vars = {}
        self.build()

    def build(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Ring Spinning Frame — AI Condition Monitor",
                  font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Enter new digital-twin measurements or load a CSV row.").pack(anchor="w", pady=(4, 12))

        mf = ttk.LabelFrame(outer, text="1. Trained AI model", padding=10)
        mf.pack(fill="x", pady=(0, 10))
        ttk.Entry(mf, textvariable=self.model_path, state="readonly").pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(mf, text="Browse model…", command=self.load_model).pack(side="left")
        ttk.Label(mf, textvariable=self.status, wraplength=820).pack(anchor="w", pady=(8, 0))

        ff = ttk.LabelFrame(outer, text="2. New measurements", padding=10)
        ff.pack(fill="both", expand=True, pady=(0, 10))
        canvas = tk.Canvas(ff, highlightthickness=0)
        scroll = ttk.Scrollbar(ff, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for i, name in enumerate(FEATURES):
            r, c = divmod(i, 2)
            cell = ttk.Frame(form, padding=4)
            cell.grid(row=r, column=c, sticky="ew")
            ttk.Label(cell, text=name, width=26).pack(side="left")
            var = tk.StringVar(value=str(DEFAULTS[name]))
            ttk.Entry(cell, textvariable=var, width=14).pack(side="left")
            self.vars[name] = var
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(0, 10))
        ttk.Button(buttons, text="Predict condition", command=self.predict).pack(side="left")
        ttk.Button(buttons, text="Load one CSV row…", command=self.load_csv).pack(side="left", padx=8)
        ttk.Button(buttons, text="Reset values", command=self.reset).pack(side="left")

        result = ttk.LabelFrame(outer, text="3. Prediction and alert", padding=12)
        result.pack(fill="x")
        ttk.Label(result, textvariable=self.prediction, font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(result, textvariable=self.confidence, wraplength=820).pack(anchor="w", pady=(4, 0))
        self.alert = ttk.Label(result, text="Alert status: waiting for prediction",
                               font=("Segoe UI", 11, "bold"), wraplength=820)
        self.alert.pack(anchor="w", pady=(8, 0))
        ttk.Label(outer, text="Prototype only: trained on synthetic data; not validated for real industrial safety alerts.",
                  wraplength=820).pack(anchor="w", pady=(10, 0))

    def load_model(self):
        path = filedialog.askopenfilename(title="Select trained AI model",
                                          filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")])
        if not path:
            return
        try:
            loaded = joblib.load(path)

            # The training script saves a bundle dictionary, not the estimator alone.
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
                model_features = list(loaded.get("features", FEATURES))
                medians = loaded.get("medians", {})
            else:
                # Also support a plain estimator saved directly with joblib.
                model = loaded
                model_features = list(getattr(model, "feature_names_in_", FEATURES))
                medians = {}

            missing = [f for f in model_features if f not in FEATURES]
            if missing:
                raise ValueError("Model expects features not supported by this dashboard: " + ", ".join(missing))
            if not hasattr(model, "predict"):
                raise ValueError(
                    "The selected file does not contain a trained estimator. "
                    "Choose ring_spinning_bearing_ai_model.joblib created by train_bearing_ai.py."
                )

            self.model = model
            self.model_features = model_features
            self.model_medians = medians
            self.model_path.set(path)
            self.status.set(f"Model loaded: {Path(path).name} ({len(model_features)} input features)")
        except Exception as e:
            messagebox.showerror("Model loading failed", str(e))

    def make_row(self):
        values = {}
        for name, var in self.vars.items():
            try:
                values[name] = float(var.get().strip())
            except ValueError:
                raise ValueError(f"Enter a valid number for {name}.")
        return pd.DataFrame([values], columns=FEATURES)

    def predict(self):
        if self.model is None:
            messagebox.showwarning("Load model first", "Click Browse model and select ring_spinning_bearing_ai_model.joblib.")
            return
        try:
            row = self.make_row()
            expected = self.model_features
            # Use the training script's saved median values for any unavailable/invalid values.
            row = row[expected].copy()
            for feature in expected:
                row[feature] = pd.to_numeric(row[feature], errors="coerce")
                if row[feature].isna().any():
                    if feature in self.model_medians:
                        row[feature] = row[feature].fillna(self.model_medians[feature])
                    else:
                        raise ValueError(f"Missing or invalid numeric value for '{feature}'.")
            pred = str(self.model.predict(row)[0])
            name = LABELS.get(pred, pred.replace("_", " ").title())
            self.prediction.set("Predicted condition: " + name)
            self.confidence.set("")
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(row)[0]
                classes = list(self.model.classes_)
                ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
                self.confidence.set("Model confidence estimate: " + f"{max(probs):.1%}" +
                                    " | " + ", ".join(f"{LABELS.get(str(c), str(c))}: {p:.0%}" for c, p in ranked))
            if pred == "healthy":
                self.alert.configure(text="STATUS: No fault class predicted. Continue monitoring.", foreground="green")
            else:
                self.alert.configure(text=f"ABNORMAL CONDITION ALERT: {name}. Inspect the simulated condition.", foreground="red")
        except Exception as e:
            messagebox.showerror("Prediction failed", str(e))

    def load_csv(self):
        path = filedialog.askopenfilename(title="Select digital-twin CSV",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            if df.empty:
                raise ValueError("CSV has no rows.")
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                raise ValueError("CSV is missing required columns: " + ", ".join(missing))
            row = df.iloc[-1]
            for name in FEATURES:
                self.vars[name].set(str(row[name]))
            self.status.set(f"Loaded last row from {Path(path).name}; click Predict condition.")
        except Exception as e:
            messagebox.showerror("CSV loading failed", str(e))

    def reset(self):
        for name, value in DEFAULTS.items():
            self.vars[name].set(str(value))
        self.prediction.set("No prediction yet")
        self.confidence.set("")
        self.alert.configure(text="Alert status: waiting for prediction", foreground="")

def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
