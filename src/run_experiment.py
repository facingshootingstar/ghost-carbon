#!/usr/bin/env python3
"""Ghost Carbon: meter train+infer CO2 for small models vs 'green' claims."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PROXY_CPU_WATTS = 65.0
PROXY_GCO2_PER_KWH = 475.0
CLAIM_KG = 0.0


def proxy_kgco2e(duration_s: float) -> dict:
    kwh = PROXY_CPU_WATTS * duration_s / 3600.0 / 1000.0
    kg = kwh * PROXY_GCO2_PER_KWH / 1000.0
    return {
        "energy_kwh": kwh,
        "emissions_kg": kg,
        "meter": "PROXY_labeled",
        "proxy_cpu_w": PROXY_CPU_WATTS,
        "proxy_gco2_per_kwh": PROXY_GCO2_PER_KWH,
    }


def try_codecarbon_tracker():
    try:
        from codecarbon import EmissionsTracker

        return EmissionsTracker(
            project_name="ghost-carbon",
            output_dir=str(RESULTS),
            log_level="error",
            save_to_file=False,
            measure_power_secs=1,
        )
    except Exception as e:
        print(f"[info] CodeCarbon unavailable ({e}); using PROXY estimator.")
        return None


def measure(fn, label: str) -> dict:
    tracker = try_codecarbon_tracker()
    t0 = time.perf_counter()
    if tracker is not None:
        try:
            tracker.start()
            fn()
            emissions = tracker.stop()
            dur = time.perf_counter() - t0
            energy = getattr(tracker, "_total_energy", None)
            kwh = float(energy.kWh) if energy is not None else float("nan")
            return {
                "phase_model": label,
                "duration_s": dur,
                "energy_kwh": kwh,
                "emissions_kg": float(emissions) if emissions is not None else float("nan"),
                "meter": "codecarbon",
                "proxy_cpu_w": np.nan,
                "proxy_gco2_per_kwh": np.nan,
            }
        except Exception as e:
            print(f"[warn] CodeCarbon failed ({e}); PROXY for {label}")
            try:
                tracker.stop()
            except Exception:
                pass
    fn()
    dur = time.perf_counter() - t0
    row = proxy_kgco2e(dur)
    row["phase_model"] = label
    row["duration_s"] = dur
    return row


def try_torch_mlp(n_features, n_classes, X_train_s, y_train, X_test_s):
    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(42)

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_classes),
                )

            def forward(self, x):
                return self.net(x)

        model = TinyMLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        xt = torch.tensor(X_train_s, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        xv = torch.tensor(X_test_s, dtype=torch.float32)

        def train_mlp():
            model.train()
            for _ in range(40):
                opt.zero_grad()
                loss = loss_fn(model(xt), yt)
                loss.backward()
                opt.step()

        def infer_mlp():
            model.eval()
            with torch.no_grad():
                for _ in range(50):
                    model(xv)

        def accuracy():
            model.eval()
            with torch.no_grad():
                pred = model(xv).argmax(1).numpy()
            return float((pred == y_train[:0]).mean()) if False else float(
                (model(xv).argmax(1).numpy() == np.asarray(
                    __import__("sklearn.model_selection", fromlist=["x"]).train_test_split
                ))
            )

        return "torch", train_mlp, infer_mlp, model, xv
    except Exception as e:
        print(f"[info] torch unavailable ({e}); sklearn MLPClassifier")
        return None


def main():
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rows = []

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rows.append(measure(lambda: rf.fit(X_train, y_train), "RF_train"))
    rows.append(measure(lambda: [rf.predict(X_test) for _ in range(50)], "RF_infer"))

    lr = LogisticRegression(max_iter=500, random_state=42)
    rows.append(measure(lambda: lr.fit(X_train_s, y_train), "LR_train"))
    rows.append(measure(lambda: [lr.predict(X_test_s) for _ in range(50)], "LR_infer"))

    # MLP: torch if present else sklearn
    torch_bundle = None
    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(42)

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(X_train_s.shape[1], 64),
                    nn.ReLU(),
                    nn.Linear(64, len(np.unique(y))),
                )

            def forward(self, x):
                return self.net(x)

        model = TinyMLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        xt = torch.tensor(X_train_s, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        xv = torch.tensor(X_test_s, dtype=torch.float32)

        def train_mlp():
            model.train()
            for _ in range(40):
                opt.zero_grad()
                loss = loss_fn(model(xt), yt)
                loss.backward()
                opt.step()

        def infer_mlp():
            model.eval()
            with torch.no_grad():
                for _ in range(50):
                    model(xv)

        rows.append(measure(train_mlp, "MLP_train"))
        rows.append(measure(infer_mlp, "MLP_infer"))
        model.eval()
        with torch.no_grad():
            mlp_acc = float((model(xv).argmax(1).numpy() == y_test).mean())
        mlp_backend = "torch"
    except Exception as e:
        print(f"[info] torch unavailable ({e}); using sklearn MLPClassifier")
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
        rows.append(measure(lambda: mlp.fit(X_train_s, y_train), "MLP_train"))
        rows.append(measure(lambda: [mlp.predict(X_test_s) for _ in range(50)], "MLP_infer"))
        mlp_acc = float(mlp.score(X_test_s, y_test))
        mlp_backend = "sklearn"

    df = pd.DataFrame(rows)
    df["claim_kg"] = CLAIM_KG
    df["ghost_gap_kg"] = df["emissions_kg"] - df["claim_kg"]
    df.to_csv(RESULTS / "emissions.csv", index=False)
    print(df.to_string(index=False))

    acc = {
        "RF": float(rf.score(X_test, y_test)),
        "LR": float(lr.score(X_test_s, y_test)),
        "MLP": mlp_acc,
        "mlp_backend": mlp_backend,
    }
    (RESULTS / "accuracies.json").write_text(json.dumps(acc, indent=2))

    families = ["RF", "LR", "MLP"]
    metered = [df[df["phase_model"].str.startswith(f)]["emissions_kg"].sum() for f in families]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(families))
    ax.bar(x - 0.2, [CLAIM_KG] * len(families), width=0.4, label="Narrative claim (negligible)")
    ax.bar(x + 0.2, metered, width=0.4, label="Metered / proxy kgCO2e")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("kgCO₂e (train+infer)")
    ax.set_title("Ghost Carbon: claim vs metered")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "claim_vs_metered.png", dpi=120)
    plt.close()
    print(f"Wrote {RESULTS}")
    print("accuracies:", acc)


if __name__ == "__main__":
    main()
