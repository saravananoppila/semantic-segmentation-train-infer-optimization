"""Plot GPU utilization + memory over time from nvidia-smi telemetry CSVs.

Usage:
    python plot_gpu_metrics.py gpu_metrics_bf16_fusedsgd.csv [more.csv ...]

Produces a 2-row figure per file (util% and memory MiB vs elapsed seconds),
plus a combined overlay if multiple files are given.
"""
import sys
import csv
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # headless: write PNG, no display needed
import matplotlib.pyplot as plt


def load(path):
    ts, util, mem = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                t = datetime.strptime(row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f")
                u = float(row[" utilization.gpu [%]"].strip().split()[0])
                m = float(row[" memory.used [MiB]"].strip().split()[0])
            except (ValueError, KeyError, IndexError):
                continue
            ts.append(t); util.append(u); mem.append(m)
    t0 = ts[0]
    elapsed = [(t - t0).total_seconds() for t in ts]
    return elapsed, util, mem


def plot_single(path):
    elapsed, util, mem = load(path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax1.plot(elapsed, util, lw=0.9, color="tab:blue")
    ax1.fill_between(elapsed, util, alpha=0.15, color="tab:blue")
    ax1.axhline(sum(util) / len(util), ls="--", lw=1, color="navy",
                label=f"avg {sum(util)/len(util):.1f}%")
    ax1.set_ylabel("GPU utilization [%]")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)
    ax1.set_title(f"GPU telemetry — {path}")

    ax2.plot(elapsed, mem, lw=0.9, color="tab:red")
    ax2.fill_between(elapsed, mem, alpha=0.15, color="tab:red")
    ax2.axhline(max(mem), ls="--", lw=1, color="darkred",
                label=f"peak {max(mem):.0f} MiB")
    ax2.axhline(23034, ls=":", lw=1, color="black", label="L4 ceiling 23034 MiB")
    ax2.set_ylabel("Memory used [MiB]")
    ax2.set_xlabel("Elapsed time [s]")
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + "_plot.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_overlay(paths):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for p in paths:
        elapsed, util, mem = load(p)
        label = p.replace("gpu_metrics_", "").rsplit(".", 1)[0]
        ax1.plot(elapsed, util, lw=0.8, alpha=0.8, label=label)
        ax2.plot(elapsed, mem, lw=0.8, alpha=0.8, label=label)
    ax1.set_ylabel("GPU utilization [%]"); ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title("GPU telemetry overlay")
    ax2.axhline(23034, ls=":", lw=1, color="black", label="L4 ceiling")
    ax2.set_ylabel("Memory used [MiB]"); ax2.set_xlabel("Elapsed time [s]")
    ax2.legend(loc="lower right", fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("gpu_metrics_overlay_plot.png", dpi=120)
    plt.close(fig)
    print("wrote gpu_metrics_overlay_plot.png")


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        sys.exit("usage: python plot_gpu_metrics.py <csv> [csv ...]")
    for f in files:
        plot_single(f)
    if len(files) > 1:
        plot_overlay(files)
