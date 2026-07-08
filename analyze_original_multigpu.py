"""Summarize the ORIGINAL multi-GPU run: per-GPU telemetry + training throughput.

- Telemetry CSV has an `index` column (0-3) since it was captured with --query-gpu=index,...
  so we summarize util/mem/power PER GPU and aggregated.
- Throughput parsed from the tee'd training log (`Time:` per disp_iter line). Effective batch
  = num_gpus * batch_size_per_gpu = 4 * 2 = 8 images per iter.
"""
import csv, re, statistics, sys

TELEM = "gpu_metrics_original_multigpu.csv"
LOG = "train_original_multigpu.log"
NGPU = 4
BATCH_PER_GPU = 2
EFF_BATCH = NGPU * BATCH_PER_GPU


def fnum(s):
    return float(s.strip().split()[0])


def summarize_telemetry():
    per = {i: {"util": [], "mem": [], "pow": []} for i in range(NGPU)}
    with open(TELEM) as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row["index"].strip())
                per[idx]["util"].append(fnum(row[" utilization.gpu [%]"]))
                per[idx]["mem"].append(fnum(row[" memory.used [MiB]"]))
                per[idx]["pow"].append(fnum(row[" power.draw [W]"]))
            except (KeyError, ValueError, IndexError):
                continue
    print("\n=== PER-GPU TELEMETRY (1 Hz nvidia-smi, whole training run) ===")
    print(f"{'GPU':>3} {'mean util%':>10} {'max util%':>9} {'mean mem':>9} {'peak mem':>9} {'mean pow':>9}")
    aggu, aggm, aggp = [], [], []
    for i in range(NGPU):
        u, m, p = per[i]["util"], per[i]["mem"], per[i]["pow"]
        if not u:
            continue
        print(f"{i:>3} {statistics.mean(u):>10.1f} {max(u):>9.0f} "
              f"{statistics.mean(m):>8.0f}M {max(m):>8.0f}M {statistics.mean(p):>8.1f}W")
        aggu += u; aggm.append(max(m)); aggp += p
    if aggu:
        print(f"{'ALL':>3} {statistics.mean(aggu):>10.1f} {max(aggu):>9.0f} "
              f"{'-':>8} {sum(aggm):>8.0f}M {statistics.mean(aggp)*NGPU:>8.1f}W (sum)")
    return per


def summarize_throughput():
    times = []
    pat = re.compile(r"Epoch: \[(\d+)\]\[(\d+)/\d+\].*Time: ([\d.]+).*Loss: ([\d.]+)")
    last_loss = None
    with open(LOG) as f:
        for line in f:
            m = pat.search(line)
            if m:
                ep, it, t, loss = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
                # skip iter 0 of each epoch (warmup/checkpoint spike)
                if it > 0:
                    times.append(t)
                last_loss = loss
    if not times:
        print("\n(no steady-state iter timings yet)")
        return
    # steady state: drop the warmest 10% as outliers
    times_sorted = sorted(times)
    steady = times_sorted[: int(len(times_sorted) * 0.9)] or times_sorted
    mean_t = statistics.mean(steady)
    print("\n=== TRAINING THROUGHPUT (original train.py, 4xL4 DataParallel) ===")
    print(f"iters timed (it>0):     {len(times)}")
    print(f"mean steady iter time:  {mean_t:.3f} s  (median {statistics.median(times):.3f}s)")
    print(f"effective batch:        {EFF_BATCH} img/iter  ({NGPU} gpus x {BATCH_PER_GPU})")
    print(f"throughput:             {EFF_BATCH / mean_t:.2f} img/s")
    print(f"last loss:              {last_loss}")
    print(f"~time per full epoch:   {20210 / (EFF_BATCH / mean_t) / 60:.1f} min (20,210 imgs)")


if __name__ == "__main__":
    summarize_telemetry()
    summarize_throughput()
