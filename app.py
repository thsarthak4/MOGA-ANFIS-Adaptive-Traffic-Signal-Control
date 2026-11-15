import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pymoo.core.callback import Callback

from anfis import pretrain, repair, ANFIS, QL_MAX, V_MAX, RULES
from simulator import Scenario, simulate, BASE_DEMAND, MIXES, LANES, SPACE
from controllers import (FixedTime, WebsterController, AnfisController,
                         webster_greens, GMIN, GMAX)
from optimize import run_opt, pick_compromise

st.set_page_config(page_title="MOGA-ANFIS Traffic Controller", layout="wide")
os.makedirs("results", exist_ok=True)
PH = ["Phase 1 (E-W through)", "Phase 2 (N-S)", "Phase 3 (E-W left)"]
ORDER = ["Fixed-time", "Webster", "ANFIS (pretrained)", "GA-ANFIS", "MOGA-ANFIS"]

TRAIN_SC = [Scenario("tr-med-2W", 0.8, "2W-dominant"), Scenario("tr-peak-2W", 1.0, "2W-dominant"),
            Scenario("tr-med-car", 0.8, "car-dominant"), Scenario("tr-peak-car", 1.0, "car-dominant")]
TEST_SC = [Scenario(f"{lvl}|{mix}", sc, mix)
           for lvl, sc in [("Low", 0.6), ("Medium", 0.85), ("Peak", 1.1)] for mix in MIXES]


# ---------------- helpers ----------------
def load(path):
    return np.load(path) if os.path.exists(path) else None


if "init" not in st.session_state:
    st.session_state.ga = load("results/ga_params.npy")
    st.session_state.mo = load("results/mo_params.npy")
    st.session_state.pF = load("results/pareto_F.npy")
    st.session_state.pX = load("results/pareto_X.npy")
    st.session_state.init = True


@st.cache_resource(show_spinner="Pretraining ANFIS (hybrid learning)...")
def get_pretrained():
    m, info = pretrain()
    return m.p.copy(), info


base, info = get_pretrained()


def build_controllers(scale):
    ws, _ = webster_greens(BASE_DEMAND * scale)
    d = {"Fixed-time": FixedTime(), "Webster": WebsterController(ws),
         "ANFIS (pretrained)": AnfisController(base, name="ANFIS (pretrained)")}
    if st.session_state.ga is not None:
        d["GA-ANFIS"] = AnfisController(st.session_state.ga, name="GA-ANFIS")
    if st.session_state.mo is not None:
        d["MOGA-ANFIS"] = AnfisController(st.session_state.mo, name="MOGA-ANFIS")
    return d


def draw_state(sec, i, ymax):
    t, p, is_green, q0, q1, q2 = sec[i]
    qs = np.array([q0, q1, q2]) / LANES * SPACE
    cols = [("#2ecc71" if is_green else "#f1c40f") if k == p else "#e74c3c" for k in range(3)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(["Ph 1\nE-W thru", "Ph 2\nN-S", "Ph 3\nE-W left"], qs, color=cols)
    for k, v in enumerate(qs):
        ax.text(k, v + ymax * 0.02, f"{v:.0f} m", ha="center")
    state = "GREEN" if is_green else "YELLOW / ALL-RED"
    ax.set_title(f"t = {int(t)} s   |   Phase {int(p) + 1}: {state}")
    ax.set_ylim(0, ymax * 1.15); ax.set_ylabel("Queue length (m)")
    plt.tight_layout()
    return fig


class Prog(Callback):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def notify(self, algorithm):
        self.fn(algorithm.n_gen, algorithm.opt.get("F"))


# ---------------- sidebar ----------------
st.sidebar.title("Scenario (Live tab)")
scale = st.sidebar.slider("Demand scale (1.0 = Hanoi peak ~3100 PCU/h)", 0.4, 1.4, 1.0, 0.05)
mix = st.sidebar.selectbox("Traffic mix", list(MIXES))
amp = st.sidebar.slider("Demand fluctuation", 0.0, 0.8, 0.4, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 99999, 7)
T = st.sidebar.select_slider("Simulated seconds", [600, 900, 1200, 1800], 900)
st.sidebar.markdown("---")
st.sidebar.write("**Saved models**")
st.sidebar.write(f"GA-ANFIS: {'✅' if st.session_state.ga is not None else '❌ not trained'}")
st.sidebar.write(f"MOGA-ANFIS: {'✅' if st.session_state.mo is not None else '❌ not trained'}")

st.title("🚦 MOGA-ANFIS Adaptive Traffic Signal Control")
st.caption("Heterogeneous mixed traffic | NSGA-II tuned ANFIS | 3-phase isolated intersection")

tabs = st.tabs(["1. Overview", "2. Live simulation", "3. ANFIS", "4. Optimisation", "5. Pareto front", "6. Comparison"])

# ================= 1. OVERVIEW =================
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Pipeline")
        st.markdown("""
1. **Simulator**: queue-based model of a 3-phase intersection, Poisson arrivals, mixed vehicles converted to PCU.
2. **ANFIS**: inputs = queue length and arrival rate, 25 Sugeno rules, output = urgency U in [0,1].
3. **Green time**: `g = g_min + U x (g_max - g_min)` with g_min = 15 s, g_max from Webster.
4. **GA baseline** (Base Paper 1): tunes ANFIS for **delay only**.
5. **NSGA-II** (our contribution): tunes ANFIS for **delay, queue length and stops** together.
6. **Compare** against fixed-time, Webster, pretrained ANFIS and GA-ANFIS.
""")
    with c2:
        st.subheader("Vehicle mix & PCU")
        st.dataframe(pd.DataFrame({"Type": ["2-wheeler", "Car", "Auto", "Bus"],
                                   "PCU": [0.15, 1.0, 0.75, 2.5],
                                   "2W-dominant %": [70, 20, 7, 3],
                                   "Car-dominant %": [15, 65, 15, 5]}), hide_index=True)
        st.write("Base demand (PCU/h per phase):", BASE_DEMAND.tolist())
        st.write("Lanes per phase:", LANES.tolist())

# ================= 2. LIVE =================
with tabs[1]:
    ctrls = build_controllers(scale)
    c1, c2, c3 = st.columns([2, 1, 1])
    cname = c1.selectbox("Controller", list(ctrls))
    step = c2.selectbox("Play step (s/frame)", [5, 10, 20], index=1)
    run = c3.button("Run simulation", type="primary")
    if run:
        trace = {"sec": [], "dec": []}
        r = simulate(ctrls[cname], Scenario("live", scale, mix, amp), int(seed), T, trace)
        st.session_state.live = (cname, r, trace)

    if "live" in st.session_state:
        lname, r, trace = st.session_state.live
        st.markdown(f"**Last run: {lname}**")
        m = st.columns(4)
        m[0].metric("Avg delay (s/veh)", f"{r['delay']:.1f}")
        m[1].metric("Avg queue (m)", f"{r['queue']:.1f}")
        m[2].metric("Stops / veh", f"{r['stops']:.2f}")
        m[3].metric("Throughput (veh/h)", f"{r['throughput']:.0f}")

        sec = trace["sec"]
        arr = np.array(sec, dtype=float)
        meters = arr[:, 3:6] / LANES * SPACE
        ymax = float(meters.max()) + 1

        left, right = st.columns([1, 1])
        with left:
            i = st.slider("Time (s)", 0, len(sec) - 1, 0)
            holder = st.empty()
            fig = draw_state(sec, i, ymax); holder.pyplot(fig); plt.close(fig)
            if st.button("Play animation"):
                for k in range(0, len(sec), step):
                    fig = draw_state(sec, k, ymax)
                    holder.pyplot(fig); plt.close(fig)
                    time.sleep(0.04)
        with right:
            st.write("**Queue length per phase over time (m)**")
            st.line_chart(pd.DataFrame(meters, columns=PH, index=arr[:, 0].astype(int)))

        dec = pd.DataFrame(trace["dec"])
        a, b = st.columns([1, 1])
        with a:
            st.write("**Green time chosen by the controller at each phase start**")
            st.line_chart(dec.pivot_table(index="t", columns="phase", values="green_s"))
        with b:
            st.write("**Decision log (what ANFIS saw, what it decided)**")
            st.dataframe(dec, height=300, hide_index=True)
    else:
        st.info("Pick a controller and press **Run simulation**.")

# ================= 3. ANFIS =================
with tabs[2]:
    st.subheader("ANFIS structure and learning")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE train", f"{info['train']:.4f}")
    c2.metric("RMSE check", f"{info['check']:.4f}")
    c3.metric("RMSE test", f"{info['test']:.4f}")
    st.caption("Paper reports approx 0.025 / 0.023 / 0.029 on its own dataset (3750 samples, 70/20/10 split).")

    tuned = st.session_state.mo if st.session_state.mo is not None else None
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    xq, xv = np.linspace(0, QL_MAX, 400), np.linspace(0, V_MAX, 400)
    sets = [(base, "--")] + ([(tuned, "-")] if tuned is not None else [])
    for p, ls in sets:
        for k in range(5):
            ax[0].plot(xq, np.exp(-(xq - p[k]) ** 2 / (2 * p[5 + k] ** 2)), ls, color=f"C{k}")
            ax[1].plot(xv, np.exp(-(xv - p[10 + k]) ** 2 / (2 * p[15 + k] ** 2)), ls, color=f"C{k}")
    ax[0].set_title("Queue length MFs" + (" (dashed=pretrained, solid=MOGA)" if tuned is not None else ""))
    ax[0].set_xlabel("m"); ax[1].set_title("Arrival MFs"); ax[1].set_xlabel("PCU/min/lane")
    st.pyplot(fig); plt.close(fig)

    opts = {"Pretrained": base}
    if st.session_state.ga is not None: opts["GA-ANFIS"] = st.session_state.ga
    if st.session_state.mo is not None: opts["MOGA-ANFIS"] = st.session_state.mo

    a, b = st.columns(2)
    with a:
        st.write("**Rule base (rows = arrival V, cols = queue QL)**")
        st.dataframe(pd.DataFrame(RULES, index=["V: VS", "V: S", "V: M", "V: L", "V: VL"],
                                  columns=["QL: VS", "QL: S", "QL: M", "QL: L", "QL: VL"]))
        st.write("**Hybrid-learning curve (RMSE vs epoch)**")
        st.line_chart(pd.DataFrame(info["hist"], columns=["train", "check"]))
    with b:
        st.write("**Controller response surface: green time (s)**")
        which = st.selectbox("Model", list(opts))
        ph = st.selectbox("Phase", [1, 2, 3]) - 1
        gq, gv = np.meshgrid(np.linspace(0, QL_MAX, 50), np.linspace(0, V_MAX, 50))
        u = ANFIS(opts[which]).predict(gq.ravel(), gv.ravel()).reshape(gq.shape)
        g = GMIN + u * (GMAX[ph] - GMIN)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        cs = ax.contourf(gq, gv, g, 20, cmap="viridis")
        plt.colorbar(cs, label="green time (s)")
        ax.set_xlabel("Queue length (m)"); ax.set_ylabel("Arrival (PCU/min/lane)")
        st.pyplot(fig); plt.close(fig)
    st.write("**Try the ANFIS by hand**")
    c1, c2 = st.columns(2)
    qi = c1.slider("Queue length (m)", 0.0, QL_MAX, 60.0)
    vi = c2.slider("Arrival (PCU/min/lane)", 0.0, V_MAX, 8.0)
    uu = float(ANFIS(opts[which]).predict(np.array([qi]), np.array([vi]))[0])
    st.success(f"Urgency U = {uu:.3f}  ->  green = {GMIN + uu * (GMAX[ph] - GMIN):.1f} s for {PH[ph]}")

# ================= 4. OPTIMISATION =================
with tabs[3]:
    st.subheader("Train GA-ANFIS / MOGA-ANFIS")
    c1, c2, c3 = st.columns(3)
    pop = c1.slider("Population", 10, 60, 20, 5)
    gen = c2.slider("Generations", 5, 50, 10, 5)
    mode = c3.radio("Algorithm", ["NSGA-II (delay, queue, stops)", "GA (delay only)"])
    st.caption("Quick demo: pop 20 / gen 10. Full run: pop 40 / gen 30 (about 10-15 min).")
    if st.button("Run optimisation", type="primary"):
        n_obj = 3 if mode.startswith("NSGA") else 1
        bar = st.progress(0.0); status = st.empty(); chart = st.empty(); best = []

        def cb(n_gen, F):
            best.append(float(np.min(F[:, 0])))
            bar.progress(min(n_gen / gen, 1.0))
            status.write(f"Generation {n_gen}/{gen} - best delay so far: {best[-1]:.2f} s/veh")
            chart.line_chart(pd.DataFrame({"best delay": best}))

        res = run_opt(base, TRAIN_SC, n_obj, pop, gen, callback=Prog(cb))
        if n_obj == 1:
            st.session_state.ga = repair(np.atleast_2d(res.X)[0])
            np.save("results/ga_params.npy", st.session_state.ga)
            st.success("GA-ANFIS trained and saved.")
        else:
            F, X = np.atleast_2d(res.F), np.atleast_2d(res.X)
            st.session_state.pF, st.session_state.pX = F, X
            st.session_state.mo = repair(X[pick_compromise(F)])
            np.save("results/pareto_F.npy", F); np.save("results/pareto_X.npy", X)
            np.save("results/mo_params.npy", st.session_state.mo)
            st.success(f"MOGA-ANFIS trained: {len(F)} Pareto solutions found. See the Pareto tab.")

# ================= 5. PARETO =================
with tabs[4]:
    F, X = st.session_state.pF, st.session_state.pX
    if F is None:
        st.info("Run NSGA-II in the Optimisation tab first.")
    else:
        st.subheader(f"Pareto front ({len(F)} solutions)")
        st.write("Choose how much you care about each objective. The best-matching solution is highlighted.")
        c1, c2, c3 = st.columns(3)
        w = np.array([c1.slider("Weight: delay", 0.0, 1.0, 1.0),
                      c2.slider("Weight: queue", 0.0, 1.0, 1.0),
                      c3.slider("Weight: stops", 0.0, 1.0, 1.0)])
        Fn = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
        idx = int(np.argmin(Fn @ w)) if w.sum() > 0 else pick_compromise(F)
        names = ["Avg delay (s/veh)", "Avg queue (m)", "Stops/veh"]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for a, (i, j) in zip(ax, [(0, 1), (0, 2), (1, 2)]):
            a.scatter(F[:, i], F[:, j], s=25)
            a.scatter(F[idx, i], F[idx, j], c="r", marker="*", s=220, label="selected")
            a.set_xlabel(names[i]); a.set_ylabel(names[j]); a.grid(alpha=.3)
        ax[0].legend()
        st.pyplot(fig); plt.close(fig)
        st.write("Selected solution objectives:", dict(zip(names, np.round(F[idx], 3))))
        if st.button("Use this solution as MOGA-ANFIS"):
            st.session_state.mo = repair(X[idx])
            np.save("results/mo_params.npy", st.session_state.mo)
            st.success("MOGA-ANFIS updated. Re-run the Comparison tab.")
        st.dataframe(pd.DataFrame(F, columns=names).round(3), height=250)

# ================= 6. COMPARISON =================
with tabs[5]:
    st.subheader("All controllers on unseen seeds and scenarios")
    nseeds = st.slider("Seeds per scenario", 3, 15, 5)
    if st.button("Run comparison", type="primary"):
        rows = []
        tot = len(TEST_SC) * len(build_controllers(1.0)) * nseeds
        bar = st.progress(0.0); n = 0
        for sc in TEST_SC:
            for c in build_controllers(sc.scale).values():
                for sd in range(2000, 2000 + nseeds):
                    rows.append({"scenario": sc.name, "controller": c.name, "seed": sd, **simulate(c, sc, sd)})
                    n += 1; bar.progress(n / tot)
        st.session_state.cmp = pd.DataFrame(rows)
        st.session_state.cmp.to_csv("results/all_runs.csv", index=False)

    if "cmp" in st.session_state:
        df = st.session_state.cmp
        order = [o for o in ORDER if o in df.controller.unique()]
        cols = ["delay", "queue", "stops", "throughput"]
        mean = df.groupby(["scenario", "controller"])[cols].mean().round(2)
        std = df.groupby(["scenario", "controller"])[cols].std().round(2)
        st.write("**Mean results**"); st.dataframe(mean)

        metric = st.selectbox("Metric to plot", cols)
        piv = df.groupby(["scenario", "controller"])[metric].mean().unstack()[order]
        fig, ax = plt.subplots(figsize=(11, 4)); piv.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric); ax.grid(alpha=.3); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        pd_ = df.groupby(["scenario", "controller"])["delay"].mean().unstack()[order]
        imp = ((1 - pd_.div(pd_["Fixed-time"], axis=0)) * 100).drop(columns="Fixed-time")
        st.write("**Delay reduction vs fixed-time (%)**")
        fig, ax = plt.subplots(figsize=(11, 4)); imp.plot(kind="bar", ax=ax)
        ax.set_ylabel("%"); ax.grid(alpha=.3); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.write("Average improvement over all scenarios (%)")
        st.dataframe(imp.mean().round(1).to_frame("delay reduction %"))
        with st.expander("Standard deviations across seeds"):
            st.dataframe(std)
        st.download_button("Download all runs (CSV)", df.to_csv(index=False), "all_runs.csv")
    else:
        st.info("Press **Run comparison**. Train GA/MOGA first so they appear in the table.")