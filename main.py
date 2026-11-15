import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from anfis import pretrain, repair, QL_MAX, V_MAX
from simulator import Scenario, simulate, BASE_DEMAND, MIXES
from controllers import FixedTime, WebsterController, AnfisController, webster_greens
from optimize import run_opt, pick_compromise

QUICK = False                          # True = fast test run (a few minutes less)
POP, GEN = (20, 10) if QUICK else (40, 30)
N_TEST_SEEDS = 5 if QUICK else 10
os.makedirs("results", exist_ok=True)

# ---------- 1. Pretrain ANFIS (replicates Base Paper 2) ----------
model, info = pretrain()
base = model.p.copy()
print(f"ANFIS RMSE  train={info['train']:.4f}  check={info['check']:.4f}  test={info['test']:.4f}")

# ---------- 2. Optimise (GA baseline vs NSGA-II) ----------
train_sc = [Scenario("tr-med-2W", 0.8, "2W-dominant"), Scenario("tr-peak-2W", 1.0, "2W-dominant"),
            Scenario("tr-med-car", 0.8, "car-dominant"), Scenario("tr-peak-car", 1.0, "car-dominant")]

t0 = time.time()
print("\n=== Single-objective GA-ANFIS (delay only) ===")
res_ga = run_opt(base, train_sc, 1, POP, GEN)
print("\n=== MOGA-ANFIS (NSGA-II: delay, queue, stops) ===")
res_mo = run_opt(base, train_sc, 3, POP, GEN)
print(f"\nOptimisation time: {(time.time() - t0) / 60:.1f} min")

ga_params = repair(np.atleast_2d(res_ga.X)[0])
F, X = np.atleast_2d(res_mo.F), np.atleast_2d(res_mo.X)
idx = pick_compromise(F)
mo_params = repair(X[idx])
np.save("results/ga_params.npy", ga_params); np.save("results/mo_params.npy", mo_params)
np.save("results/pareto_F.npy", F);          np.save("results/pareto_X.npy", X)

# ---------- 3. Test on unseen seeds and scenarios ----------
test_sc = [Scenario(f"{lvl}|{mix}", sc, mix)
           for lvl, sc in [("Low", 0.6), ("Medium", 0.85), ("Peak", 1.1)] for mix in MIXES]
rows = []
for sc in test_sc:
    ws, _ = webster_greens(BASE_DEMAND * sc.scale)
    ctrls = [FixedTime(), WebsterController(ws), AnfisController(base, name="ANFIS (pretrained)"),
             AnfisController(ga_params, name="GA-ANFIS"), AnfisController(mo_params, name="MOGA-ANFIS")]
    for c in ctrls:
        for seed in range(2000, 2000 + N_TEST_SEEDS):
            r = simulate(c, sc, seed)
            rows.append({"scenario": sc.name, "controller": c.name, "seed": seed, **r})
df = pd.DataFrame(rows)
df.to_csv("results/all_runs.csv", index=False)

order = ["Fixed-time", "Webster", "ANFIS (pretrained)", "GA-ANFIS", "MOGA-ANFIS"]
summary = df.groupby(["scenario", "controller"])[["delay", "queue", "stops", "throughput"]].agg(["mean", "std"]).round(3)
summary.to_csv("results/summary.csv")
print("\n", df.groupby(["scenario", "controller"])[["delay", "queue", "stops", "throughput"]].mean().round(2))

# ---------- 4. Plots ----------
# (a) Membership functions before / after
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
xq, xv = np.linspace(0, QL_MAX, 400), np.linspace(0, V_MAX, 400)
for p, ls in [(base, "--"), (mo_params, "-")]:
    for i in range(5):
        ax[0].plot(xq, np.exp(-(xq - p[i]) ** 2 / (2 * p[5 + i] ** 2)), ls, color=f"C{i}")
        ax[1].plot(xv, np.exp(-(xv - p[10 + i]) ** 2 / (2 * p[15 + i] ** 2)), ls, color=f"C{i}")
ax[0].set_title("Queue length MFs (dashed = pretrained, solid = MOGA-tuned)"); ax[0].set_xlabel("m")
ax[1].set_title("Arrival MFs"); ax[1].set_xlabel("PCU/min/lane")
plt.tight_layout(); plt.savefig("results/mfs.png", dpi=150); plt.close()

# (b) Convergence
plt.figure(figsize=(6, 4))
plt.plot(res_ga.conv, label="GA-ANFIS (single-objective)")
plt.plot(res_mo.conv, label="MOGA-ANFIS (best delay on front)")
plt.xlabel("Generation"); plt.ylabel("Training avg delay (s/veh)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/convergence.png", dpi=150); plt.close()

# (c) Pareto front (pairwise)
names = ["Avg delay (s/veh)", "Avg queue (m)", "Stops/veh"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, (i, j) in zip(ax, [(0, 1), (0, 2), (1, 2)]):
    a.scatter(F[:, i], F[:, j], s=25, label="Pareto front")
    a.scatter(F[idx, i], F[idx, j], c="r", marker="*", s=200, label="Compromise")
    a.set_xlabel(names[i]); a.set_ylabel(names[j]); a.grid(alpha=.3)
ax[0].legend(); plt.tight_layout(); plt.savefig("results/pareto.png", dpi=150); plt.close()

# (d) Delay improvement over fixed-time
piv = df.groupby(["scenario", "controller"])["delay"].mean().unstack()[order]
imp = (1 - piv.div(piv["Fixed-time"], axis=0)) * 100
imp.drop(columns="Fixed-time").plot(kind="bar", figsize=(11, 4.5))
plt.ylabel("Delay reduction vs fixed-time (%)"); plt.xticks(rotation=30, ha="right"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/improvement.png", dpi=150); plt.close()

print("\nDone. Check the results/ folder.")