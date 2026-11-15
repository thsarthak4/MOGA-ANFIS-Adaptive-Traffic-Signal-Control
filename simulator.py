"""Queue-based mesoscopic simulator of a 3-phase isolated intersection
under heterogeneous (2-wheeler / car / auto / bus) mixed traffic."""
import numpy as np
from dataclasses import dataclass

PCU = np.array([0.15, 1.0, 0.75, 2.5])           # 2W, car, auto, bus
MIXES = {                                         # share of VEHICLES
    "2W-dominant":  np.array([0.70, 0.20, 0.07, 0.03]),
    "car-dominant": np.array([0.15, 0.65, 0.15, 0.05]),
}
BASE_DEMAND = np.array([1700., 700., 700.])       # PCU/h per phase (~3100 total, like Hanoi peak)
LANES = np.array([3, 2, 2])                       # lanes serving each phase
SAT = 1800.0                                      # saturation flow PCU/h/lane
LOST = 5                                          # 3s yellow + 2s all-red per phase
SPACE = 6.0                                       # metres of lane per PCU in queue
T_DEFAULT = 1800                                  # simulated seconds


@dataclass
class Scenario:
    name: str
    scale: float = 1.0          # demand multiplier on BASE_DEMAND
    mix: str = "2W-dominant"
    amp: float = 0.4            # amplitude of demand fluctuation (0 = constant demand)


def generate_arrivals(sc, T, rng):
    """Poisson arrivals per second, per phase, per vehicle type, with
    sinusoidal fluctuation shifted between phases (so the best split changes over time)."""
    share = MIXES[sc.mix]
    mean_pcu = float((share * PCU).sum())
    t = np.arange(T)[:, None]
    phi = np.array([0, np.pi, np.pi / 2])[None, :]
    mult = 1 + sc.amp * np.sin(2 * np.pi * t / 600 + phi)          # (T,3)
    pcu_rate = BASE_DEMAND[None, :] * sc.scale * mult / 3600        # PCU/s
    veh_rate = pcu_rate / mean_pcu                                  # veh/s
    lam = veh_rate[:, :, None] * share[None, None, :]               # (T,3,4)
    counts = rng.poisson(lam)
    return counts @ PCU, counts.sum(axis=2)                         # PCU arrivals, vehicle arrivals


def simulate(ctrl, sc, seed, T=T_DEFAULT, trace=None):
    """Run one simulation. ctrl.green(phase, queue_m, arrival_pcu_min_lane) -> green seconds.
    If trace is a dict {"sec": [], "dec": []}, per-second states and per-phase decisions are recorded."""
    rng = np.random.default_rng(seed)
    a_pcu, a_veh = generate_arrivals(sc, T, rng)
    cs = np.cumsum(a_pcu, axis=0).tolist()
    ap, av = a_pcu.tolist(), a_veh.tolist()

    qp = [0.0] * 3          # queue in PCU
    qv = [0.0] * 3          # queue in vehicles
    cap = [SAT * l / 3600.0 for l in LANES]   # discharge PCU/s
    delay = stops = arrived = departed = qsum = 0.0
    t = 0
    while t < T:
        for p in range(3):
            if t >= T:
                break
            ql = qp[p] / LANES[p] * SPACE                          # metres
            t0 = max(0, t - 60)
            v = (cs[t][p] - cs[t0][p]) * (60.0 / max(t - t0, 1)) / LANES[p]   # PCU/min/lane
            g = max(1, int(round(ctrl.green(p, ql, v))))
            if trace is not None:
                trace["dec"].append({"t": t, "phase": p + 1, "queue_m": round(ql, 1),
                                     "arrival": round(v, 1), "green_s": g})
            for s in range(g + LOST):
                if t >= T:
                    break
                is_green = s < g
                for q in range(3):
                    if (not (is_green and q == p)) or qp[q] > 0.5:   # arrives into red or queue -> stop
                        stops += av[t][q]
                    qp[q] += ap[t][q]
                    qv[q] += av[t][q]
                arrived += av[t][0] + av[t][1] + av[t][2]
                if is_green and qp[p] > 0:
                    d = min(qp[p], cap[p])
                    f = d / qp[p]
                    departed += qv[p] * f
                    qv[p] -= qv[p] * f
                    qp[p] -= d
                delay += qv[0] + qv[1] + qv[2]
                qsum += sum(qp[q] / LANES[q] * SPACE for q in range(3)) / 3.0
                if trace is not None:
                    trace["sec"].append((t, p, is_green, qp[0], qp[1], qp[2]))
                t += 1
    return {
        "delay": delay / max(arrived, 1),             # s per vehicle
        "queue": qsum / T,                            # mean queue length (m)
        "stops": stops / max(arrived, 1),             # stops per vehicle
        "throughput": departed * 3600.0 / T,          # veh/h
    }