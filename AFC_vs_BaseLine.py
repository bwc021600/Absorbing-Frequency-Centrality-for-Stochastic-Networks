import csv
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


OUTDIR = Path("afc_baseline_grouped_bar_outputs")

MODELS = ["ER", "WS"]

N = 100
SEED = 42

P_EDGE_ON = 0.85
K_MIN = 5
ALPHA_STOP = 0.15

M_AFC = 60
R_MC = 500

DPI = 600
SHOW_FIGURES = True

NODE_SELECTION_MODE = "afc_top"
NUM_NODES_TO_PLOT = 10
UNION_TOP_K = 5

RANK_SCOPE = "plotted"

MAIN_METRICS = [
    "AFC",
    "BC (base graph)",
    "Averaged BC",
    "RW-BC",
]

FULL_METRICS = [
    "AFC",
    "BC (base graph)",
    "Averaged BC",
    "Prob-SP BC",
    "RW-BC",
    "PageRank",
    "Freq-argmax",
]

COLORS = {
    "AFC": "#ff4d4d",
    "BC (base graph)": "#6a6aff",
    "Averaged BC": "#55c7e8",
    "Prob-SP BC": "#f0a62e",
    "RW-BC": "#008000",
    "PageRank": "#9b59b6",
    "Freq-argmax": "#7f7f7f",
}

EDGE_COLORS = {
    "AFC": "#cc0000",
    "BC (base graph)": "#3030cc",
    "Averaged BC": "#1b93b8",
    "Prob-SP BC": "#b87300",
    "RW-BC": "#005c00",
    "PageRank": "#6f3d91",
    "Freq-argmax": "#4d4d4d",
}


def tie_key(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return str(v)


def ranked_items(scores: Dict[int, float]) -> List[Tuple[int, float]]:
    return sorted(scores.items(), key=lambda kv: (-float(kv[1]), tie_key(kv[0])))


def argmax_with_tiebreak(scores: Dict[int, float]):
    return ranked_items(scores)[0][0]


def normalize_score_dict(scores: Dict[int, float], eps: float = 1e-15) -> Dict[int, float]:
    clean = {}

    for v, val in scores.items():
        val = float(val)

        if not np.isfinite(val):
            val = 0.0

        if val < 0 and abs(val) < 1e-12:
            val = 0.0

        clean[v] = max(0.0, val)

    total = sum(clean.values())

    if total <= eps:
        return {v: 0.0 for v in clean}

    return {v: clean[v] / total for v in clean}


def make_base_graph(model: str, n: int = N, seed: int = SEED) -> nx.Graph:
    if model == "ER":
        return nx.erdos_renyi_graph(n=n, p=0.08, seed=seed)

    if model == "WS":
        return nx.watts_strogatz_graph(n=n, k=6, p=0.10, seed=seed)

    if model == "BA":
        return nx.barabasi_albert_graph(n=n, m=3, seed=seed)

    raise ValueError("model must be one of {'ER', 'WS', 'BA'}")


def sample_realized_graph(
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    p_edge_on: float,
    rng: np.random.Generator,
) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(nodes)

    if len(edges) == 0:
        return H

    keep = rng.random(len(edges)) < p_edge_on
    H.add_edges_from([e for e, k in zip(edges, keep) if k])

    return H


# ============================================================
# AFC simulator and AMC kernel
# ============================================================
class OneStepSimulator:
    def __init__(
        self,
        base_graph: nx.Graph,
        p_edge_on: float,
        k_min: int,
        alpha_stop: float,
        seed: int = 0,
    ):
        self.G0 = base_graph.copy()
        self.nodes = list(self.G0.nodes())
        self.node_to_idx = {v: i for i, v in enumerate(self.nodes)}
        self.edges = list(self.G0.edges())

        self.n = len(self.nodes)
        self.p_edge_on = float(p_edge_on)
        self.k_min = int(k_min)
        self.alpha_stop = float(alpha_stop)

        self.rng = np.random.default_rng(seed)

    def sample_realized_working_graph(self) -> nx.Graph:
        return sample_realized_graph(
            nodes=self.nodes,
            edges=self.edges,
            p_edge_on=self.p_edge_on,
            rng=self.rng,
        )

    def local_center(self, anchor_i: int, H: nx.Graph):
        comp = set(nx.node_connected_component(H, anchor_i))

        if len(comp) < self.k_min:
            return None

        sub = H.subgraph(comp)
        bc = nx.betweenness_centrality(sub, normalized=False, weight=None)

        return argmax_with_tiebreak(bc)

    def sample_next(self, current_center_i: int):
        if self.rng.random() < self.alpha_stop:
            return None

        H = self.sample_realized_working_graph()

        return self.local_center(current_center_i, H)


def estimate_amc_kernel(
    sim: OneStepSimulator,
    M: int,
    absorb_floor: float = 1e-6,
) -> np.ndarray:
    n = sim.n
    ABS = n

    P_hat = np.zeros((n + 1, n + 1), dtype=float)

    for i_idx, i_node in enumerate(sim.nodes):
        counts = np.zeros(n + 1, dtype=float)

        for _ in range(M):
            z = sim.sample_next(i_node)

            if z is None:
                counts[ABS] += 1.0
            else:
                counts[sim.node_to_idx[z]] += 1.0

        row = counts / float(M)

        if row[ABS] < absorb_floor:
            row[ABS] = absorb_floor
            transient_sum = row[:n].sum()

            if transient_sum > 0:
                row[:n] *= (1.0 - absorb_floor) / transient_sum
            else:
                row[ABS] = 1.0

        P_hat[i_idx, :] = row

    P_hat[ABS, ABS] = 1.0

    return P_hat


def afc_from_kernel(P_hat: np.ndarray, s: np.ndarray) -> np.ndarray:
    n = len(s)
    Q = P_hat[:n, :n]
    A = np.eye(n) - Q

    try:
        mu = np.linalg.solve(A.T, s)
    except np.linalg.LinAlgError:
        mu = np.linalg.solve((A + 1e-10 * np.eye(n)).T, s)

    mu = np.maximum(mu, 0.0)

    if mu.sum() <= 0:
        return np.ones(n) / n

    return mu / mu.sum()


def compute_afc_scores(
    G: nx.Graph,
    p_edge_on: float = P_EDGE_ON,
    k_min: int = K_MIN,
    alpha_stop: float = ALPHA_STOP,
    M: int = M_AFC,
    seed: int = SEED,
) -> Dict[int, float]:
    sim = OneStepSimulator(
        base_graph=G,
        p_edge_on=p_edge_on,
        k_min=k_min,
        alpha_stop=alpha_stop,
        seed=seed + 1,
    )

    P_hat = estimate_amc_kernel(sim, M=M, absorb_floor=1e-6)

    s = np.ones(sim.n) / sim.n
    b = afc_from_kernel(P_hat, s)

    return normalize_score_dict({node: float(b[i]) for i, node in enumerate(sim.nodes)})


# ============================================================
# Baseline centralities
# ============================================================
def base_graph_bc_score(G: nx.Graph) -> Dict[int, float]:
    bc = nx.betweenness_centrality(G, normalized=False, weight=None)
    return normalize_score_dict(bc)


def averaged_bc_score(
    G: nx.Graph,
    p_edge_on: float = P_EDGE_ON,
    R: int = R_MC,
    seed: int = SEED,
) -> Dict[int, float]:
    nodes = list(G.nodes())
    edges = list(G.edges())
    rng = np.random.default_rng(seed + 1000)

    acc = {v: 0.0 for v in nodes}

    for _ in range(R):
        H = sample_realized_graph(nodes, edges, p_edge_on, rng)
        bc = nx.betweenness_centrality(H, normalized=False, weight=None)

        for v in nodes:
            acc[v] += float(bc.get(v, 0.0))

    acc = {v: acc[v] / float(R) for v in nodes}

    return normalize_score_dict(acc)


def probabilistic_shortest_path_bc_score(
    G: nx.Graph,
    p_edge_on: float = P_EDGE_ON,
) -> Dict[int, float]:
    nodes = list(G.nodes())
    CB = {v: 0.0 for v in nodes}

    for s in nodes:
        S = []
        P = {v: [] for v in nodes}
        sigma = {v: 0.0 for v in nodes}
        dist = {}

        sigma[s] = 1.0
        dist[s] = 0

        Q = deque([s])

        while Q:
            v = Q.popleft()
            S.append(v)

            for w in G.neighbors(v):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    Q.append(w)

                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta = {v: 0.0 for v in nodes}

        while S:
            w = S.pop()

            target_weight = 0.0 if w == s else (p_edge_on ** dist[w])
            coeff = target_weight + delta[w]

            for v in P[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * coeff

            if w != s:
                CB[w] += delta[w]

    if not G.is_directed():
        for v in CB:
            CB[v] *= 0.5

    return normalize_score_dict(CB)


def current_flow_bc_score(G: nx.Graph) -> Dict[int, float]:
    scores = {v: 0.0 for v in G.nodes()}

    for comp_nodes in nx.connected_components(G):
        comp_nodes = list(comp_nodes)

        if len(comp_nodes) <= 2:
            continue

        H = G.subgraph(comp_nodes).copy()

        cf = None

        for solver in ("lu", "full", "cg"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cf = nx.current_flow_betweenness_centrality(
                        H,
                        normalized=False,
                        weight=None,
                        solver=solver,
                    )
                break
            except Exception:
                cf = None

        if cf is None:
            continue

        for v, val in cf.items():
            scores[v] = float(val)

    return normalize_score_dict(scores)


def pagerank_score(G: nx.Graph) -> Dict[int, float]:
    pr = nx.pagerank(G, alpha=0.85, max_iter=1000, tol=1e-12)
    return normalize_score_dict(pr)


def frequency_argmax_score(
    G: nx.Graph,
    p_edge_on: float = P_EDGE_ON,
    R: int = R_MC,
    seed: int = SEED,
) -> Dict[int, float]:
    nodes = list(G.nodes())
    edges = list(G.edges())
    rng = np.random.default_rng(seed + 2000)

    counts = {v: 0.0 for v in nodes}

    for _ in range(R):
        H = sample_realized_graph(nodes, edges, p_edge_on, rng)
        bc = nx.betweenness_centrality(H, normalized=False, weight=None)
        winner = argmax_with_tiebreak(bc)
        counts[winner] += 1.0

    return normalize_score_dict(counts)


def compute_all_scores(G: nx.Graph, seed: int = SEED) -> Dict[str, Dict[int, float]]:
    scores = {
        "AFC": compute_afc_scores(
            G,
            p_edge_on=P_EDGE_ON,
            k_min=K_MIN,
            alpha_stop=ALPHA_STOP,
            M=M_AFC,
            seed=seed,
        ),
        "BC (base graph)": base_graph_bc_score(G),
        "Averaged BC": averaged_bc_score(
            G,
            p_edge_on=P_EDGE_ON,
            R=R_MC,
            seed=seed,
        ),
        "Prob-SP BC": probabilistic_shortest_path_bc_score(
            G,
            p_edge_on=P_EDGE_ON,
        ),
        "RW-BC": current_flow_bc_score(G),
        "PageRank": pagerank_score(G),
        "Freq-argmax": frequency_argmax_score(
            G,
            p_edge_on=P_EDGE_ON,
            R=R_MC,
            seed=seed,
        ),
    }

    return scores

def select_plot_nodes(
    scores: Dict[str, Dict[int, float]],
    metrics: Sequence[str],
    mode: str = NODE_SELECTION_MODE,
    num_nodes: int = NUM_NODES_TO_PLOT,
    union_top_k: int = UNION_TOP_K,
) -> List[int]:
    if mode == "afc_top":
        return [v for v, _ in ranked_items(scores["AFC"])[:num_nodes]]

    if mode == "union_topk":
        selected = []
        seen = set()

        for metric in metrics:
            for v, _ in ranked_items(scores[metric])[:union_top_k]:
                if v not in seen:
                    selected.append(v)
                    seen.add(v)

        selected = sorted(
            selected,
            key=lambda v: (-scores["AFC"].get(v, 0.0), tie_key(v)),
        )

        return selected[:num_nodes]

    raise ValueError("NODE_SELECTION_MODE must be 'afc_top' or 'union_topk'")


def make_rank_dict(
    metric_scores: Dict[int, float],
    nodes_to_plot: Sequence[int],
    rank_scope: str = RANK_SCOPE,
) -> Dict[int, int]:
    if rank_scope == "plotted":
        pool = {v: metric_scores.get(v, 0.0) for v in nodes_to_plot}
    elif rank_scope == "global":
        pool = dict(metric_scores)
    else:
        raise ValueError("rank_scope must be either 'plotted' or 'global'")

    ranked = sorted(pool.items(), key=lambda kv: (-float(kv[1]), tie_key(kv[0])))

    return {v: r for r, (v, _) in enumerate(ranked, start=1)}

def plot_grouped_scores(
    scores: Dict[str, Dict[int, float]],
    nodes_to_plot: Sequence[int],
    metrics: Sequence[str],
    model: str,
    savepath: Path,
    title: str = None,
    dpi: int = DPI,
    rank_scope: str = RANK_SCOPE,
    show_rank_labels: bool = True,
):
    if title is None:
        title = f"{model} Network: AFC vs Baseline Centrality Scores"

    fig_width = 13.5 if len(metrics) <= 4 else 16.0
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), dpi=dpi)

    fig.suptitle(
        title,
        fontsize=17,
        y=0.985,
    )

    x = np.arange(len(nodes_to_plot))
    n_metrics = len(metrics)
    width = 0.78 / n_metrics

    rank_maps = {
        metric: make_rank_dict(
            metric_scores=scores[metric],
            nodes_to_plot=nodes_to_plot,
            rank_scope=rank_scope,
        )
        for metric in metrics
    }

    all_vals = [
        scores[metric].get(v, 0.0)
        for metric in metrics
        for v in nodes_to_plot
    ]

    y_max = max(all_vals) if all_vals else 1.0

    if y_max <= 0:
        y_max = 1.0

    rank_offset = y_max * 0.018

    for j, metric in enumerate(metrics):
        vals = [scores[metric].get(v, 0.0) for v in nodes_to_plot]
        offset = (j - (n_metrics - 1) / 2.0) * width

        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=metric,
            color=COLORS.get(metric, None),
            edgecolor=EDGE_COLORS.get(metric, "black"),
            linewidth=0.8,
        )

        if show_rank_labels:
            for bar, node_id, val in zip(bars, nodes_to_plot, vals):
                rank_num = rank_maps[metric].get(node_id, "")

                xpos = bar.get_x() + bar.get_width() / 2.0
                ypos = max(float(val), y_max * 0.004) + rank_offset

                ax.text(
                    xpos,
                    ypos,
                    str(rank_num),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                    clip_on=False,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in nodes_to_plot], fontsize=14)

    ax.set_xlabel("Node", fontsize=16)
    ax.set_ylabel("Normalized score", fontsize=16)

    ax.set_ylim(0.0, y_max * 1.30)

    ax.yaxis.grid(True, linestyle=(0, (6, 6)), alpha=0.45)
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="x",
        top=True,
        labeltop=False,
        direction="out",
        length=7,
        width=1.0,
        color="0.55",
    )
    ax.tick_params(axis="y", labelsize=14)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=len(metrics),
        frameon=False,
        fontsize=14,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.98)

    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

def print_top5(
    model: str,
    scores: Dict[str, Dict[int, float]],
    metrics: Sequence[str],
):
    print(f"\n=== {model}: Top 5 node IDs ===")

    for metric in metrics:
        top5 = ranked_items(scores[metric])[:5]
        ids = [str(v) for v, _ in top5]
        vals = [val for _, val in top5]

        print(f"{metric:16s}: {', '.join(ids)}")
        print(f"{'':16s}  " + ", ".join(f"{v:.6f}" for v in vals))


def write_top5_csv(
    rows: List[Tuple[str, str, int, int, float]],
    path: Path,
):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "rank", "node_id", "normalized_score"])
        writer.writerows(rows)


def write_all_scores_csv(
    model: str,
    scores: Dict[str, Dict[int, float]],
    path: Path,
):
    nodes = sorted(next(iter(scores.values())).keys(), key=tie_key)
    metrics = list(scores.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "node_id"] + metrics)

        for v in nodes:
            writer.writerow(
                [model, v] + [scores[m].get(v, 0.0) for m in metrics]
            )

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    top5_rows = []

    for model in MODELS:
        G = make_base_graph(model, n=N, seed=SEED)

        print(f"\n=== {model} base graph ===")
        print(f"nodes={G.number_of_nodes()}")
        print(f"edges={G.number_of_edges()}")
        print(f"components={nx.number_connected_components(G)}")
        print(f"isolates={len(list(nx.isolates(G)))}")

        scores = compute_all_scores(G, seed=SEED)

        print_top5(model, scores, FULL_METRICS)

        for metric in FULL_METRICS:
            for rank, (node_id, val) in enumerate(ranked_items(scores[metric])[:5], start=1):
                top5_rows.append((model, metric, rank, node_id, val))

        write_all_scores_csv(
            model=model,
            scores=scores,
            path=OUTDIR / f"{model}_all_node_scores.csv",
        )

        nodes_main = select_plot_nodes(
            scores=scores,
            metrics=MAIN_METRICS,
            mode=NODE_SELECTION_MODE,
            num_nodes=NUM_NODES_TO_PLOT,
            union_top_k=UNION_TOP_K,
        )

        plot_grouped_scores(
            scores=scores,
            nodes_to_plot=nodes_main,
            metrics=MAIN_METRICS,
            model=model,
            savepath=OUTDIR / f"{model}_AFC_baselines_main4_grouped_bar_ranked.png",
            title=f"{model} Network: AFC vs Baseline Centrality Scores",
            dpi=DPI,
            rank_scope=RANK_SCOPE,
            show_rank_labels=True,
        )

        nodes_full = select_plot_nodes(
            scores=scores,
            metrics=FULL_METRICS,
            mode=NODE_SELECTION_MODE,
            num_nodes=NUM_NODES_TO_PLOT,
            union_top_k=UNION_TOP_K,
        )

        plot_grouped_scores(
            scores=scores,
            nodes_to_plot=nodes_full,
            metrics=FULL_METRICS,
            model=model,
            savepath=OUTDIR / f"{model}_AFC_baselines_all7_grouped_bar_ranked.png",
            title=f"{model} Network: AFC vs All Baseline Centrality Scores",
            dpi=DPI,
            rank_scope=RANK_SCOPE,
            show_rank_labels=True,
        )

    write_top5_csv(
        rows=top5_rows,
        path=OUTDIR / "top5_summary.csv",
    )

    print(f"\nSaved outputs to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()