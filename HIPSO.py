
import math
import random
import statistics
import time
from typing import Callable, List, Tuple

def griewank(x: List[float]) -> float:
    s = sum(xi * xi for xi in x) / 4000.0
    p = 1.0
    for i, xi in enumerate(x, start=1):
        p *= math.cos(xi / math.sqrt(i))
    return s - p + 1.0


def rastrigin(x: List[float]) -> float:
    n = len(x)
    return 10.0 * n + sum(xi * xi - 10.0 * math.cos(2.0 * math.pi * xi) for xi in x)


def ackley(x: List[float]) -> float:
    n = len(x)
    s1 = sum(xi * xi for xi in x)
    s2 = sum(math.cos(2.0 * math.pi * xi) for xi in x)
    return -20.0 * math.exp(-0.2 * math.sqrt(s1 / n)) - math.exp(s2 / n) + 20.0 + math.e


class HIPSO:

    def __init__(self, func: Callable, dim: int, bounds: Tuple[float, float],
                 a: float = 0.75, b: float = 1.5, c: float = 1.5,
                 nPop: int = 50, M: int = 4,
                 z: int = 11, f: int = 13, cmax: int = 4,
                 max_iter: int = 1000,
                 eps: float = 1e-10,
                 p_min: float = 0.05):
        self.func = func
        self.dim = dim
        self.lo, self.hi = bounds
        self.a, self.b, self.c = a, b, c
        self.nPop, self.M0 = nPop, M
        self.z, self.f, self.cmax = z, f, cmax
        self.max_iter, self.eps = max_iter, eps
        self.p_min = p_min

    def _rand_pos(self) -> List[float]:
        return [random.uniform(self.lo, self.hi) for _ in range(self.dim)]

    def _rand_vel(self) -> List[float]:
        span = self.hi - self.lo
        return [random.uniform(-0.1 * span, 0.1 * span) for _ in range(self.dim)]

    def _clamp_reflect(self, x: List[float], v: List[float]) -> None:
        for j in range(self.dim):
            if x[j] < self.lo:
                x[j] = self.lo
                v[j] = -v[j] / 2.0
            elif x[j] > self.hi:
                x[j] = self.hi
                v[j] = -v[j] / 2.0

    @staticmethod
    def _dist(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    @staticmethod
    def _centroid(positions: List[List[float]]) -> List[float]:
        n = len(positions)
        r = len(positions[0])
        return [sum(p[j] for p in positions) / n for j in range(r)]

    def _sigma(self, R_in: float, N_m: int) -> float:
        if N_m <= 1:
            return R_in * 2.0
        ratio = self.p_min / (1.0 - self.p_min)
        return R_in * (ratio * (N_m - 1)) ** (1.0 / self.dim)

    def run(self) -> Tuple[float, int]:
        nPop = self.nPop
        M = self.M0

        X = [self._rand_pos() for _ in range(nPop)]
        V = [self._rand_vel() for _ in range(nPop)]
        groups = [i % M for i in range(nPop)]
        losses = [0] * M

        pbest = [x[:] for x in X]
        pbest_f = [self.func(x) for x in X]
        best_f = min(pbest_f)
        best_x = pbest[pbest_f.index(best_f)][:]

        def update_gbest(M_current):
            gx = [None] * M_current
            gf = [float("inf")] * M_current
            for i, g in enumerate(groups):
                if g < M_current and pbest_f[i] < gf[g]:
                    gf[g] = pbest_f[i]
                    gx[g] = pbest[i][:]
            return gx, gf

        gbest_x, gbest_f = update_gbest(M)
        last_improve_iter = 0
        prev_best = best_f

        for it in range(1, self.max_iter + 1):

            for i in range(nPop):
                g = groups[i]
                lb = gbest_x[g] if gbest_x[g] is not None else best_x
                u = random.random()
                for j in range(self.dim):
                    V[i][j] = (self.a * V[i][j]
                               + self.b * u * (pbest[i][j] - X[i][j])
                               + self.c * u * (lb[j] - X[i][j]))
                    X[i][j] += V[i][j]
                self._clamp_reflect(X[i], V[i])

            centroids = {}
            for g in range(M):
                mpos = [X[i] for i in range(nPop) if groups[i] == g]
                if mpos:
                    centroids[g] = self._centroid(mpos)
            for i in range(nPop):
                g = groups[i]
                if g not in centroids:
                    continue
                d_own = self._dist(X[i], centroids[g])
                for k in range(M):
                    if k == g or k not in centroids:
                        continue
                    if self._dist(X[i], centroids[k]) < d_own:
                        for j in range(self.dim):
                            V[i][j] = self.a * (centroids[g][j] - X[i][j])
                        break

            for i in range(nPop):
                fit = self.func(X[i])
                if fit < pbest_f[i]:
                    pbest_f[i] = fit
                    pbest[i] = X[i][:]
                if fit < best_f:
                    best_f = fit
                    best_x = X[i][:]
            gbest_x, gbest_f = update_gbest(M)

            if it % self.z == 0:
                for g in range(M):
                    members = [i for i in range(nPop) if groups[i] == g]
                    if len(members) < 2:
                        continue
                    N_m = len(members)
                    R_in = math.sqrt(self.dim) / (N_m * max(M, 1) * 2.0 * (1.0 + it))
                    sigma = self._sigma(R_in, N_m)
                    members.sort(key=lambda i: pbest_f[i])
                    leader = members[0]
                    to_replace = set()
                    for j_idx, j_a in enumerate(members):
                        if j_a in to_replace:
                            continue
                        for i_idx in range(j_idx + 1, N_m):
                            j_b = members[i_idx]
                            if j_b in to_replace:
                                continue
                            if self._dist(X[j_a], X[j_b]) < R_in:
                                to_replace.add(j_b)
                    for idx in to_replace:
                        X[idx] = [
                            max(self.lo, min(self.hi,
                                X[leader][j] + random.uniform(-sigma, sigma)))
                            for j in range(self.dim)
                        ]
                        V[idx] = self._rand_vel()
                        pbest[idx] = X[idx][:]
                        pbest_f[idx] = self.func(X[idx])

            if it % self.f == 0 and M > 1:
                gbest_x, gbest_f = update_gbest(M)
                order = sorted(range(M), key=lambda g: gbest_f[g])
                winner, loser = order[0], order[-1]
                losses[winner] = 0
                losses[loser] += 1
                if losses[loser] >= self.cmax and M > 1:
                    keep = order[:-1]
                    remap = {old: new for new, old in enumerate(keep)}
                    new_centroids = {}
                    for kg in keep:
                        mpos = [X[i] for i in range(nPop) if groups[i] == kg]
                        if mpos:
                            new_centroids[kg] = self._centroid(mpos)
                    for i in range(nPop):
                        if groups[i] == loser:
                            nearest = min(
                                (kg for kg in keep if kg in new_centroids),
                                key=lambda kg: self._dist(X[i], new_centroids[kg]),
                                default=keep[0])
                            groups[i] = remap[nearest]
                        else:
                            groups[i] = remap.get(groups[i], 0)
                    M -= 1
                    losses = [0] * M
                    gbest_x, gbest_f = update_gbest(M)

            if best_f <= self.eps:
                return best_f, it
            if abs(prev_best - best_f) < 1e-14:
                if it - last_improve_iter > 150:
                    return best_f, it
            else:
                last_improve_iter = it
            prev_best = best_f

        return best_f, self.max_iter


def evaluate_params(params: dict, dim: int,
                    compute_budget: int = None,
                    n_runs: int = 20) -> float:
    if compute_budget is None:
        compute_budget = {2: 15000, 8: 50000, 20: 100000}.get(dim, 40000)
    max_iter = max(50, compute_budget // params["nPop"])

    funcs = [
        (griewank,  (-600.0,    600.0)),
        (rastrigin, (-5.12,       5.12)),
        (ackley,    (-32.768,   32.768)),
    ]
    fparams = []
    for func, bnds in funcs:
        for _ in range(n_runs):
            s = HIPSO(
                func=func, dim=dim, bounds=bnds, max_iter=max_iter,
                a=params["a"], b=params["b"], c=params["c"],
                nPop=params["nPop"], M=params["M"],
                f=params["f"], z=params["z"], cmax=params["cmax"],
            )
            bf, it = s.run()
            fparams.append(abs(bf) * it)
    return statistics.mean(fparams)


def experiment_table_22(runs: int = 20):
    M_values = [3, 4, 5, 6, 8, 10, 15]
    Cmax_values = [3, 4, 5, 6, 7, 8, 10, 15]
    data = []
    for cmax in Cmax_values:
        row = []
        for M in M_values:
            vals = []
            for _ in range(runs):
                s = HIPSO(func=griewank, dim=2, bounds=(-600, 600),
                         a=0.75, b=1.45, c=1.45,
                         nPop=50, M=M, z=11, f=13, cmax=cmax, max_iter=800)
                bf, it = s.run()
                vals.append(abs(bf) * it)
            row.append(statistics.mean(vals))
            print(f"  M={M:2d}  Cmax={cmax:2d}  Fparam={row[-1]:.4f}")
        data.append(row)

    print(f"\nТаблиця 2.2 — M × Cmax  (Griewank r=2, {runs} запусків)")
    print("=" * 70)
    print(f"{'Cmax':>5}", end="")
    for M in M_values:
        print(f"{M:>10}", end="")
    print()
    best = min(min(row) for row in data)
    for cmax, row in zip(Cmax_values, data):
        print(f"{cmax:>5}", end="")
        for v in row:
            marker = "*" if abs(v - best) < 1e-9 else " "
            print(f"{v:>9.4f}{marker}", end="")
        print()
    print("* — мінімум у таблиці")
    return data, M_values, Cmax_values


def experiment_table_23(runs: int = 10):
    fz_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    data = []
    for f in fz_values:
        row = []
        for z in fz_values:
            if f == z:
                row.append(float("nan"))
                continue
            vals = []
            for _ in range(runs):
                s = HIPSO(func=rastrigin, dim=8, bounds=(-5.12, 5.12),
                         a=0.75, b=1.45, c=1.45,
                         nPop=50, M=5, z=z, f=f, cmax=5, max_iter=1000)
                bf, it = s.run()
                vals.append(abs(bf) * it)
            row.append(statistics.mean(vals))
            print(f"  f={f:2d}  z={z:2d}  Fparam={row[-1]:.2f}")
        data.append(row)

    print(f"\nRastrigin r=8, {runs} запусків)")
    print("=" * 110)
    print(f"{'f/z':>4}", end="")
    for z in fz_values:
        print(f"{z:>10}", end="")
    print()
    best = min(v for row in data for v in row if not math.isnan(v))
    for f, row in zip(fz_values, data):
        print(f"{f:>4}", end="")
        for v in row:
            if math.isnan(v):
                print(f"{'—':>10}", end="")
            else:
                marker = "*" if abs(v - best) < 1e-9 else " "
                print(f"{v:>9.2f}{marker}", end="")
        print()
    print("* мінімум")
    return data, fz_values

PARAM_GRID = {
    "a":    [round(0.5 + 0.05 * i, 2) for i in range(9)],   # 0.50..0.90
    "b":    [round(1.2 + 0.1  * i, 1) for i in range(7)],   # 1.2..1.8
    "c":    [round(1.2 + 0.1  * i, 1) for i in range(7)],   # 1.2..1.8
    "M":    [3, 4, 5, 6, 7, 8],
    "f":    [5, 7, 11, 13, 17, 19, 23],
    "z":    [5, 7, 11, 13, 17, 19],
    "cmax": [3, 4, 5, 6, 7],
}


def random_params(dim: int) -> dict:
    if dim <= 2:
        nPop_range = list(range(20, 60, 5))
    elif dim <= 10:
        nPop_range = list(range(40, 90, 5))
    else:  # 20+
        nPop_range = list(range(60, 130, 10))

    p = {
        "a":    random.choice(PARAM_GRID["a"]),
        "b":    random.choice(PARAM_GRID["b"]),
        "c":    random.choice(PARAM_GRID["c"]),
        "M":    random.choice(PARAM_GRID["M"]),
        "nPop": random.choice(nPop_range),
        "f":    random.choice(PARAM_GRID["f"]),
        "z":    random.choice(PARAM_GRID["z"]),
        "cmax": random.choice(PARAM_GRID["cmax"]),
    }
    while p["f"] == p["z"]:
        p["z"] = random.choice(PARAM_GRID["z"])
    return p


def experiment_table_24(
    n_candidates: int = 80,
    runs_filter: int = 15,
    refine_top_k: int = 10,
    runs_refine: int = 40,
):
    dims = [2, 8, 20]
    results = {}

    for dim in dims:
        print(f"\n{'═'*70}")
        print(f"  ДОСЛІДЖЕННЯ ПАРАМЕТРІВ ДЛЯ {dim}-ВИМІРНОЇ ЗАДАЧІ")
        print(f"{'═'*70}")

        print(f"\n  Фаза 1: випадковий пошук серед {n_candidates} кандидатів "
              f"({runs_filter} запусків кожен)...")
        candidates = []
        t0 = time.time()
        for i in range(n_candidates):
            p = random_params(dim)
            score = evaluate_params(p, dim, n_runs=runs_filter)
            candidates.append((score, p))
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                est_total = elapsed * n_candidates / (i + 1)
                print(f"    кандидат {i+1:3d}/{n_candidates}, "
                      f"min поточний = {min(c[0] for c in candidates):.4f}, "
                      f"час {elapsed:.0f}/{est_total:.0f}c")

        candidates.sort(key=lambda x: x[0])
        top = candidates[:refine_top_k]
        print(f"\n  Топ-{refine_top_k} після фази фільтрації:")
        for i, (sc, pr) in enumerate(top, 1):
            print(f"    {i:2d}. score={sc:.4f}  "
                  + "  ".join(f"{k}={pr[k]}" for k in
                  ["a","b","c","M","nPop","f","z","cmax"]))
        print(f"\n  Фаза 2: уточнення топ-{refine_top_k} з {runs_refine} запусками...")
        refined = []
        for i, (filter_score, p) in enumerate(top, 1):
            true_score = evaluate_params(p, dim, n_runs=runs_refine)
            refined.append((true_score, p))
            print(f"    {i:2d}. фільтр={filter_score:.4f} - уточнено={true_score:.4f}")

        refined.sort(key=lambda x: x[0])

        param_names = ["a", "b", "c", "M", "nPop", "f", "z", "cmax"]
        means = {k: statistics.mean(p[k] for _, p in refined) for k in param_names}
        stds = {k: statistics.pstdev(p[k] for _, p in refined) for k in param_names}

        results[dim] = {"mean": means, "std": stds, "best": refined[0]}

    print(f"\n\n{'═'*100}")
    print("  ОПТИМАЛЬНІ ПАРАМЕТРИ HIPSO ДЛЯ РІЗНИХ РОЗМІРНОСТЕЙ")
    print(f"{'═'*100}\n")

    hdr = f"{'Розмірність':<15}{'Показник':<12}" + "".join(f"{n:>10}" for n in
        ["a", "b", "c", "M", "nPop", "f", "z", "cmax"])
    print(hdr)
    print("─" * len(hdr))

    for dim in dims:
        m = results[dim]["mean"]
        s = results[dim]["std"]
        best_sc, best_p = results[dim]["best"]
        print(f"{f'r = {dim}':<15}{'M{x}':<12}"
              + "".join(f"{m[k]:>10.3f}" for k in
                ["a","b","c","M","nPop","f","z","cmax"]))
        print(f"{'':<15}{'σ':<12}"
              + "".join(f"{s[k]:>10.3f}" for k in
                ["a","b","c","M","nPop","f","z","cmax"]))
        print(f"  → найкращий (Fparam={best_sc:.4f}):  "
              + "  ".join(f"{k}={best_p[k]}" for k in
                ["a","b","c","M","nPop","f","z","cmax"]))
        print()

    return results

if __name__ == "__main__":
    random.seed(42)
    t_start = time.time()

    print("\n" + "─" * 70)
    print(" (Griewank r=2, M × Cmax)")
    print("─" * 70)
    experiment_table_22(runs=20)
    t1 = time.time()
    print(f"\n Таблиця готова за {(t1-t_start)/60:.1f} хв")

    print("\n" + "─" * 70)
    print(" Rastrigin r=8, f × z")
    print("─" * 70)
    experiment_table_23(runs=10)
    t2 = time.time()
    print(f"\n  Таблиця  готова за {(t2-t1)/60:.1f} хв")

    print("\n" + "─" * 70)
    print(" двофазний випадковий пошук, r=2, 8, 20")
    print("─" * 70)
    experiment_table_24(
        n_candidates=80,
        runs_filter=15,
        refine_top_k=10,
        runs_refine=40,
    )
    t3 = time.time()
    print(f"\n Таблиця  готова за {(t3-t2)/60:.1f} хв")

    print(f"\n{'═'*70}")
    print(f" Загальний час: {(t3-t_start)/60:.1f} хв")
    print(f"{'═'*70}")
