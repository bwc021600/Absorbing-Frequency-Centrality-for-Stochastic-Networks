import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from matplotlib.patches import ConnectionPatch
from typing import Dict, List, Tuple, Optional, Iterable

def argmax_with_tiebreak(scores_dict: Dict[int, float]) -> int:
    best_node = None
    best_val = None
    for v, val in scores_dict.items():
        if best_node is None or (val > best_val) or (val == best_val and v < best_node):
            best_node, best_val = v, val
    return int(best_node)


def rank_vector_desc(vec: np.ndarray) -> List[Tuple[int, float]]:
    return sorted([(i, float(vec[i])) for i in range(len(vec))], key=lambda x: (-x[1], x[0]))


def topk_from_vector(vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
    return rank_vector_desc(vec)[: min(int(k), len(vec))]


def topk_from_dict(scores_dict: Dict[int, float], k: int) -> Tuple[List[int], List[Tuple[int, float]]]:

    ranked = sorted([(int(v), float(val)) for v, val in scores_dict.items()], key=lambda x: (-x[1], x[0]))
    topk_nodes = [v for v, _ in ranked[: min(int(k), len(ranked))]]
    return topk_nodes, ranked

def load_lesmis_integer_graph() -> Tuple[nx.Graph, Dict[int, str], int, int]:

    G_raw = nx.les_miserables_graph()

    # Determine original weight range from the dataset
    orig_weights = [float(d.get("weight", 1.0)) for _, _, d in G_raw.edges(data=True)]
    w_min_orig = int(min(orig_weights)) if orig_weights else 0
    w_max_orig = int(max(orig_weights)) if orig_weights else 0

    # Relabel nodes to integers, keep original names as node attribute 'name'
    G = nx.convert_node_labels_to_integers(G_raw, ordering="sorted", label_attribute="name")
    name_map = {int(k): str(v) for k, v in nx.get_node_attributes(G, "name").items()}

    # Store the original weight as 'weight_orig' (do NOT overwrite dataset 'weight' here)
    for u, v, d in G.edges(data=True):
        d["weight_orig"] = float(d.get("weight", 1.0))

    return G, name_map, w_min_orig, w_max_orig


class PerStepNormalEdgeWeightSampler:

    def __init__(
        self,
        edges_arr: np.ndarray,          # shape (m,2), integer node ids
        w0_arr: np.ndarray,             # shape (m,), original weights (float)
        w_max: int,
        mu_fraction: float = 0.50,
        sigma_fraction: float = 1.0 / 6.0,
        seed: int = 0,
    ):
        self.edges_arr = np.asarray(edges_arr, dtype=int)
        self.w0_arr = np.asarray(w0_arr, dtype=float)

        self.m = int(self.edges_arr.shape[0])
        self.w_max = int(w_max)

        self.mu_fraction = float(mu_fraction)
        self.sigma_fraction = float(sigma_fraction)

        self.rng = np.random.default_rng(int(seed))

        L = np.rint(self.w0_arr).astype(int)
        U = np.maximum(self.w_max, L).astype(int)

        headroom = (U - L).astype(float)
        mu = L.astype(float) + self.mu_fraction * headroom
        sigma = self.sigma_fraction * headroom
        sigma = np.maximum(sigma, 1e-9)  # avoid exactly-zero sigma

        self.L_arr = L
        self.U_arr = U
        self.mu_arr = mu
        self.sigma_arr = sigma

    def sample_weights_for_kept_edges(self, kept_mask: np.ndarray) -> np.ndarray:
        """
        Given a boolean mask over base edges, return sampled weights for the kept edges.
        """
        kept_mask = np.asarray(kept_mask, dtype=bool)
        if kept_mask.size != self.m:
            raise ValueError("kept_mask length must equal number of base edges")

        if not np.any(kept_mask):
            return np.zeros(0, dtype=float)

        mu = self.mu_arr[kept_mask]
        sigma = self.sigma_arr[kept_mask]
        L = self.L_arr[kept_mask]
        U = self.U_arr[kept_mask]

        x = self.rng.normal(mu, sigma)
        x_int = np.rint(x).astype(int)
        x_clip = np.clip(x_int, L, U).astype(int)

        return x_clip.astype(float)


class OneStepSimulatorPerStepWeights:

    def __init__(
        self,
        base_graph: nx.Graph,
        p_edge_on: float,
        k_min: int,
        alpha_stop: float,
        seed: int = 0,
        use_weights: bool = True,
        local_mode: str = "radius",  # "radius" or "component"
        radius: int = 2,

        mu_fraction: float = 0.50,
        sigma_fraction: float = 1.0 / 6.0,
        weight_seed: int = 12345,
    ):
        self.G0 = base_graph
        self.n = self.G0.number_of_nodes()
        self.nodes = list(self.G0.nodes())


        edges = sorted((int(u), int(v)) for (u, v) in self.G0.edges())
        self.edges_arr = np.array(edges, dtype=int) if edges else np.zeros((0, 2), dtype=int)
        self.m = int(self.edges_arr.shape[0])

        self.p_edge_on = float(p_edge_on)
        self.k_min = int(k_min)
        self.alpha_stop = float(alpha_stop)

        self.use_weights = bool(use_weights)
        self.local_mode = str(local_mode)
        self.radius = int(radius)

        self.rng = np.random.default_rng(int(seed))

        w0 = []
        for (u, v) in edges:
            w0.append(float(self.G0[u][v].get("weight_orig", self.G0[u][v].get("weight", 1.0))))
        self.w0_arr = np.array(w0, dtype=float) if w0 else np.zeros((0,), dtype=float)

        w_max = int(np.max(self.w0_arr)) if self.w0_arr.size > 0 else 1

        self.weight_sampler = PerStepNormalEdgeWeightSampler(
            edges_arr=self.edges_arr,
            w0_arr=self.w0_arr,
            w_max=w_max,
            mu_fraction=mu_fraction,
            sigma_fraction=sigma_fraction,
            seed=weight_seed,
        )

    def sample_realized_working_graph(self) -> nx.Graph:

        H = nx.Graph()
        H.add_nodes_from(self.nodes)

        if self.m == 0:
            return H

        kept_mask = self.rng.random(self.m) < self.p_edge_on
        kept_edges = self.edges_arr[kept_mask]

        if kept_edges.size == 0:
            return H

        if not self.use_weights:
            H.add_edges_from([(int(u), int(v)) for (u, v) in kept_edges.tolist()])
            return H

        kept_w = self.weight_sampler.sample_weights_for_kept_edges(kept_mask)

        weighted_edges = [
            (int(u), int(v), float(w))
            for (u, v), w in zip(kept_edges.tolist(), kept_w.tolist())
        ]
        H.add_weighted_edges_from(weighted_edges, weight="weight")
        return H

    def _local_nodes(self, anchor_i: int, H: nx.Graph) -> Optional[set]:
        anchor_i = int(anchor_i)

        if self.local_mode == "component":
            return set(nx.node_connected_component(H, anchor_i))

        if self.local_mode == "radius":
            ball = nx.single_source_shortest_path_length(H, anchor_i, cutoff=self.radius)
            return set(ball.keys())

        raise ValueError("local_mode must be 'radius' or 'component'")

    def local_center(self, anchor_i: int, H: nx.Graph) -> Optional[int]:
        local_nodes = self._local_nodes(anchor_i, H)
        if local_nodes is None or len(local_nodes) < self.k_min:
            return None

        sub = H.subgraph(local_nodes)

        if self.use_weights:
            bc = nx.betweenness_centrality(sub, normalized=False, weight="weight")
        else:
            bc = nx.betweenness_centrality(sub, normalized=False, weight=None)

        return argmax_with_tiebreak({int(v): float(val) for v, val in bc.items()})

    def sample_next(self, i: int) -> Optional[int]:

        if self.rng.random() < self.alpha_stop:
            return None

        H = self.sample_realized_working_graph()
        return self.local_center(int(i), H)

def estimate_amc_kernel(
    sim: OneStepSimulatorPerStepWeights,
    M: int,
    absorb_floor: float = 1e-6,
) -> np.ndarray:
    """
    Construct/estimate the AMC kernel P_hat via Monte Carlo.

    P_hat has shape (n+1, n+1). The last index n represents absorption ⊥.

    NOTE:
      With per-step edge-weight resampling, even if p_edge_on==1,
      the realized graph is stochastic (weights change every step),
      so an exact kernel is generally NOT available in closed form.
    """
    n = sim.n
    ABS = n
    M = int(M)
    if M <= 0:
        raise ValueError("M must be positive for Monte Carlo estimation.")

    P_hat = np.zeros((n + 1, n + 1), dtype=float)

    for i in range(n):
        counts = np.zeros(n + 1, dtype=int)
        for _ in range(M):
            z = sim.sample_next(i)
            if z is None:
                counts[ABS] += 1
            else:
                counts[int(z)] += 1

        row = counts / float(M)

        if row[ABS] < absorb_floor:
            row[ABS] = absorb_floor
            ssum = row[:n].sum()
            if ssum > 0:
                row[:n] *= (1.0 - absorb_floor) / ssum
            else:
                row[ABS] = 1.0

        P_hat[i, :] = row

    P_hat[ABS, ABS] = 1.0
    return P_hat


def afc_from_kernel(P_hat: np.ndarray, s: np.ndarray) -> np.ndarray:
 
    n = len(s)
    Q = P_hat[:n, :n]
    A = np.eye(n) - Q
    mu = np.linalg.solve(A.T, s)
    b = mu / mu.sum()
    return b


def afc_from_kernel_full(P_hat: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    n = len(s)
    Q = P_hat[:n, :n]
    N = np.linalg.inv(np.eye(n) - Q)
    mu = s @ N
    b = mu / mu.sum()
    return b, N, Q

def kl_divergence(p, q, eps=1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def ground_distance_matrix_unweighted(G: nx.Graph) -> np.ndarray:

    n = G.number_of_nodes()
    dist = np.full((n, n), np.inf)

    for s, lengths in nx.all_pairs_shortest_path_length(G):
        for t, l in lengths.items():
            dist[int(s), int(t)] = int(l)

    finite = dist[np.isfinite(dist)]
    max_finite = finite.max() if finite.size > 0 else 0.0
    D = int(max_finite) + 1
    dist[np.isinf(dist)] = D
    return dist.astype(int)


def wasserstein1_cost_matrix(p, q, cost: np.ndarray, flow_scale: int = 3000) -> float:

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()

    n = len(p)
    supply = np.round(p * flow_scale).astype(int)
    demand = np.round(q * flow_scale).astype(int)

    diff = int(supply.sum() - demand.sum())
    if diff > 0:
        demand[int(np.argmax(demand))] += diff
    elif diff < 0:
        supply[int(np.argmax(supply))] += (-diff)

    H = nx.DiGraph()
    for i in range(n):
        H.add_node(("s", i), demand=-int(supply[i]))
    for j in range(n):
        H.add_node(("d", j), demand=int(demand[j]))

    for i in range(n):
        for j in range(n):
            H.add_edge(("s", i), ("d", j), weight=int(cost[i, j]), capacity=flow_scale)

    flow_cost, _ = nx.network_simplex(H)
    return float(flow_cost) / float(flow_scale)


def sample_perturbed_kernel(P0: np.ndarray, delta_rel: float, r_lower: float, rng: np.random.Generator) -> np.ndarray:

    P0 = np.asarray(P0, dtype=float)
    n = P0.shape[0] - 1
    Q0 = P0[:n, :n]

    P = np.zeros_like(P0)

    for i in range(n):
        q0 = Q0[i].copy()
        u = rng.uniform(-float(delta_rel), float(delta_rel), size=n)
        q = np.clip(q0 * (1.0 + u), 0.0, 1.0)

        ssum = q.sum()
        if ssum > 1.0 - float(r_lower):
            if ssum > 0:
                q *= (1.0 - float(r_lower)) / ssum
            else:
                q[:] = 0.0

        r = 1.0 - q.sum()
        if r < float(r_lower):
            if q.sum() > 0:
                q *= (1.0 - float(r_lower)) / q.sum()
            r = 1.0 - q.sum()

        P[i, :n] = q
        P[i, n] = r

    P[n, n] = 1.0
    return P


def find_robust_kernel_random_search(
    P0: np.ndarray,
    s: np.ndarray,
    delta_rel: float,
    r_lower: float,
    num_samples: int,
    objective: str,                 # "kl" or "w1"
    cost: Optional[np.ndarray] = None,
    seed: int = 0,
    w1_flow_scale_search: int = 1200,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:

    rng = np.random.default_rng(int(seed))
    b0 = afc_from_kernel(P0, s)

    best_val = -1e18
    best_P = None
    best_b = None

    for _ in range(int(num_samples)):
        P = sample_perturbed_kernel(P0, delta_rel, r_lower, rng)
        b = afc_from_kernel(P, s)

        if objective == "kl":
            val = kl_divergence(b0, b)
        elif objective == "w1":
            if cost is None:
                raise ValueError("objective='w1' requires a cost matrix")
            val = wasserstein1_cost_matrix(b0, b, cost, flow_scale=int(w1_flow_scale_search))
        else:
            raise ValueError("objective must be 'kl' or 'w1'")

        if val > best_val:
            best_val = float(val)
            best_P = P
            best_b = b

    return best_P, best_b, float(best_val), b0


def annotate_bar_values(ax, bars, fmt: str = "{:.4f}", fontsize: int = 7, y_offset_frac: float = 0.012) -> None:
    heights = [float(b.get_height()) for b in bars]
    y_max = max([1e-12] + heights)
    offset = y_offset_frac * y_max

    for b in bars:
        h = float(b.get_height())
        ax.text(
            float(b.get_x() + b.get_width() / 2.0),
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
            clip_on=False,
        )


def plot_baseline_network_and_bars_with_arrows(
    G_base: nx.Graph,
    name_map: Dict[int, str],
    top_items: List[Tuple[int, float]],   # Top-K bars
    highlight_k: int = 5,                 # Top-5 red squares + arrows
    title: str = "",
    layout_seed: int = 42,
    dpi: int = 600,
    savepath: Optional[str] = None,
    bar_value_format: str = "{:.4f}",
) -> None:

    bar_nodes = [int(v) for v, _ in top_items]
    bar_vals = [float(val) for _, val in top_items]
    bar_labels = [name_map.get(v, str(v)) for v in bar_nodes]

    hk = max(1, int(highlight_k))
    highlight_nodes = [int(v) for v, _ in top_items[: min(hk, len(top_items))]]

    pos = nx.spring_layout(
        G_base,
        seed=int(layout_seed),
        k=1.2 / np.sqrt(max(G_base.number_of_nodes(), 1)),
        iterations=600,
    )

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.05)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    n = G_base.number_of_nodes()
    base_node_size = max(10, int(1800 / max(n, 1)))
    highlight_size = base_node_size * 16

    nx.draw_networkx_edges(G_base, pos, ax=ax_net, alpha=0.15, width=0.7)
    nx.draw_networkx_nodes(G_base, pos, ax=ax_net, node_size=base_node_size)

    nx.draw_networkx_nodes(
        G_base, pos, ax=ax_net,
        nodelist=highlight_nodes,
        node_size=highlight_size,
        node_shape="s",
        node_color="red",
    )

    labels = {v: name_map.get(v, str(v)) for v in highlight_nodes}
    nx.draw_networkx_labels(G_base, pos, ax=ax_net, labels=labels, font_size=8)

    ax_net.set_title(title)
    ax_net.axis("off")

    # --- bars ---
    x = np.arange(len(bar_nodes), dtype=float)
    bars = ax_bar.bar(x, bar_vals)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels, rotation=90, ha="center", fontsize=8)
    ax_bar.set_ylabel("AFC value b(s)")
    ax_bar.set_xlabel("Character (Top-ranked by b)")

    y_max = max([1e-12] + bar_vals)
    ax_bar.set_ylim(0.0, y_max * 1.22)

    annotate_bar_values(ax_bar, bars, fmt=bar_value_format, fontsize=7)

    fig.subplots_adjust(bottom=0.35)

    node_to_bar_index = {v: i for i, v in enumerate(bar_nodes)}
    for v in highlight_nodes:
        if v not in node_to_bar_index:
            continue
        idx = node_to_bar_index[v]
        bar = bars[idx]

        xB = float(bar.get_x() + bar.get_width() / 2.0)
        yB = float(bar.get_height())
        xA, yA = pos[v]

        con = ConnectionPatch(
            xyA=(xA, yA), coordsA=ax_net.transData,
            xyB=(xB, yB), coordsB=ax_bar.transData,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.1,
            color="black",
        )
        con.set_clip_on(False)
        con.set_zorder(10)
        fig.add_artist(con)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_robust_kl_w1_network_and_bars_with_arrows(
    G_base: nx.Graph,
    name_map: Dict[int, str],
    top5_kl: List[int],
    top5_w1: List[int],
    b_kl: np.ndarray,
    b_w1: np.ndarray,
    nodes_bar: List[int],
    title: str,
    layout_seed: int = 42,
    dpi: int = 600,
    savepath: Optional[str] = None,
    bar_value_format: str = "{:.4f}",
) -> None:

    set_kl = set(int(v) for v in top5_kl)
    set_w1 = set(int(v) for v in top5_w1)
    both = sorted(list(set_kl & set_w1))
    only_kl = sorted(list(set_kl - set_w1))
    only_w1 = sorted(list(set_w1 - set_kl))
    union_nodes = sorted(list(set_kl | set_w1))

    pos = nx.spring_layout(
        G_base,
        seed=int(layout_seed),
        k=1.2 / np.sqrt(max(G_base.number_of_nodes(), 1)),
        iterations=600,
    )

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.06)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    # --- network ---
    n = G_base.number_of_nodes()
    base_node_size = max(10, int(1800 / max(n, 1)))
    highlight_size = base_node_size * 18

    nx.draw_networkx_edges(G_base, pos, ax=ax_net, alpha=0.22, width=0.7)
    nx.draw_networkx_nodes(G_base, pos, ax=ax_net, node_size=base_node_size)

    if only_kl:
        nx.draw_networkx_nodes(
            G_base, pos, ax=ax_net,
            nodelist=only_kl,
            node_size=highlight_size,
            node_shape="s",
            node_color="red",
        )
    if only_w1:
        nx.draw_networkx_nodes(
            G_base, pos, ax=ax_net,
            nodelist=only_w1,
            node_size=highlight_size,
            node_shape="^",
            node_color="green",
        )
    if both:
        nx.draw_networkx_nodes(
            G_base, pos, ax=ax_net,
            nodelist=both,
            node_size=highlight_size * 1.15,
            node_shape="*",
            node_color="yellow",
            edgecolors="black",
            linewidths=0.8,
        )

    labels = {v: name_map.get(v, str(v)) for v in union_nodes}
    nx.draw_networkx_labels(G_base, pos, ax=ax_net, labels=labels, font_size=8)

    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color="red", marker="s", linestyle="None", markersize=10, label="KL Top-5 only"),
        mlines.Line2D([], [], color="green", marker="^", linestyle="None", markersize=10, label="W1 Top-5 only"),
        mlines.Line2D([], [], color="yellow", marker="*", markeredgecolor="black",
                      linestyle="None", markersize=12, label="In both Top-5"),
    ]
    ax_net.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax_net.set_title(title)
    ax_net.axis("off")

    nodes_bar = [int(v) for v in nodes_bar]
    x = np.arange(len(nodes_bar), dtype=float)
    width = 0.42

    vals_kl = [float(b_kl[i]) for i in nodes_bar]
    vals_w1 = [float(b_w1[i]) for i in nodes_bar]

    bars_kl = ax_bar.bar(x - width / 2, vals_kl, width, label="KL-robust b", color="orange")
    bars_w1 = ax_bar.bar(x + width / 2, vals_w1, width, label="W1-robust b", color="blue")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([name_map.get(i, str(i)) for i in nodes_bar], rotation=90, ha="center", fontsize=8)
    ax_bar.set_ylabel("AFC b(s)")
    ax_bar.set_xlabel("Character (selected)")
    ax_bar.legend()

    y_max = max([1e-12] + vals_kl + vals_w1)
    ax_bar.set_ylim(0.0, y_max * 1.25)

    annotate_bar_values(ax_bar, bars_kl, fmt=bar_value_format, fontsize=7)
    annotate_bar_values(ax_bar, bars_w1, fmt=bar_value_format, fontsize=7)

    fig.subplots_adjust(bottom=0.35)

    node_to_bar_index = {v: idx for idx, v in enumerate(nodes_bar)}
    for v in union_nodes:
        if v not in node_to_bar_index:
            continue
        idx = node_to_bar_index[v]
        xB = float(x[idx])
        yB = float(max(vals_kl[idx], vals_w1[idx]))
        xA, yA = pos[v]

        con = ConnectionPatch(
            xyA=(xA, yA), coordsA=ax_net.transData,
            xyB=(xB, yB), coordsB=ax_bar.transData,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.2,
            color="black",
        )
        con.set_clip_on(False)
        con.set_zorder(10)
        fig.add_artist(con)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    plt.show()


def pick_reward_hubs_by_degree(G: nx.Graph, m: int = 3) -> List[int]:

    hubs = sorted(G.nodes(), key=lambda v: (-G.degree(v), int(v)))[: int(m)]
    return [int(v) for v in hubs]


def build_distance_decayed_rewards(G: nx.Graph, hubs: List[int], base_rewards: List[float], decay: float = 0.60) -> np.ndarray:

    n = G.number_of_nodes()
    f = np.zeros(n, dtype=float)

    for h, R in zip(hubs, base_rewards):
        lengths = nx.single_source_shortest_path_length(G, int(h))
        for v, d in lengths.items():
            val = float(R) * (float(decay) ** int(d))
            if val > f[int(v)]:
                f[int(v)] = val
    return f


def reward_afc_from_b(b: np.ndarray, f: np.ndarray) -> float:
    return float(np.asarray(b, float) @ np.asarray(f, float))


def transition_reward_switch_rate(P_hat: np.ndarray, s: np.ndarray, N: np.ndarray, Q: np.ndarray) -> float:

    n = len(s)
    r = P_hat[:n, n]
    phi = (1.0 - r) - np.diag(Q)   # sum_{j!=i} Q_ij
    denom = float((s @ N).sum())
    numer = float(s @ N @ phi)
    return numer / denom


def transition_reward_improvement(P_hat: np.ndarray, s: np.ndarray, N: np.ndarray, Q: np.ndarray, f: np.ndarray) -> float:

    n = len(s)
    f = np.asarray(f, dtype=float)
    phi = np.zeros(n, dtype=float)
    for i in range(n):
        diff = np.maximum(f - f[i], 0.0)
        phi[i] = float(Q[i, :] @ diff)

    denom = float((s @ N).sum())
    numer = float(s @ N @ phi)
    return numer / denom


def plot_multi_reward_network_and_bars_with_arrows(
    G_base: nx.Graph,
    name_map: Dict[int, str],
    b: np.ndarray,
    hubs: List[int],
    f: np.ndarray,
    top_items: List[Tuple[int, float]],
    title: str,
    layout_seed: int = 42,
    dpi: int = 600,
    savepath: Optional[str] = None,
    bar_value_format: str = "{:.4f}",
):

    top5_nodes = [int(v) for v, _ in top_items[: min(5, len(top_items))]]

    set_top = set(top5_nodes)
    set_hub = set(int(h) for h in hubs)
    both = sorted(list(set_top & set_hub))
    only_top = sorted(list(set_top - set_hub))
    only_hub = sorted(list(set_hub - set_top))
    union_nodes = sorted(list(set_top | set_hub))

    pos = nx.spring_layout(
        G_base,
        seed=int(layout_seed),
        k=1.2 / np.sqrt(max(G_base.number_of_nodes(), 1)),
        iterations=600,
    )

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.06)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    n = G_base.number_of_nodes()
    base_node_size = max(10, int(1800 / max(n, 1)))
    highlight_size = base_node_size * 18

    nx.draw_networkx_edges(G_base, pos, ax=ax_net, alpha=0.20, width=0.7)
    nx.draw_networkx_nodes(G_base, pos, ax=ax_net, node_size=base_node_size)

    if only_hub:
        nx.draw_networkx_nodes(
            G_base, pos, ax=ax_net,
            nodelist=only_hub,
            node_size=highlight_size,
            node_shape="D",
            node_color="orange",
        )
    if only_top:
        nx.draw_networkx_nodes(
            G_base, pos, ax=ax_net,
            nodelist=only_top,
            node_size=highlight_size,
            node_shape="s",
            node_color="red",
        )
    if both:
        nx.draw_networkx_nodes(
            G_base, pos, ax=ax_net,
            nodelist=both,
            node_size=highlight_size * 1.2,
            node_shape="*",
            node_color="yellow",
            edgecolors="black",
            linewidths=0.8,
        )

    def node_label(v: int) -> str:
        base = name_map.get(int(v), str(int(v)))
        if v in set_hub:
            return f"{base}\nR={float(f[int(v)]):.2f}"
        return base

    labels = {v: node_label(v) for v in union_nodes}
    nx.draw_networkx_labels(G_base, pos, ax=ax_net, labels=labels, font_size=8)

    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color="red", marker="s", linestyle="None", markersize=10, label="Top-5 by b(s)"),
        mlines.Line2D([], [], color="orange", marker="D", linestyle="None", markersize=10, label="High-reward hubs"),
        mlines.Line2D([], [], color="yellow", marker="*", markeredgecolor="black",
                      linestyle="None", markersize=12, label="In both"),
    ]
    ax_net.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax_net.set_title(title)
    ax_net.axis("off")

    bar_nodes = [int(v) for v, _ in top_items]
    bar_vals = [float(val) for _, val in top_items]
    x = np.arange(len(bar_nodes), dtype=float)
    bars = ax_bar.bar(x, bar_vals)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([name_map.get(v, str(v)) for v in bar_nodes], rotation=90, ha="center", fontsize=8)
    ax_bar.set_ylabel("AFC value b(s)")
    ax_bar.set_xlabel("Node (Top-ranked)")

    y_max = max([1e-12] + bar_vals)
    ax_bar.set_ylim(0.0, y_max * 1.25)

    annotate_bar_values(ax_bar, bars, fmt=bar_value_format, fontsize=7)
    fig.subplots_adjust(bottom=0.35)

    node_to_bar_index = {v: i for i, v in enumerate(bar_nodes)}
    for v in top5_nodes:
        if v not in node_to_bar_index:
            continue
        idx = node_to_bar_index[v]
        bar = bars[idx]
        xB = float(bar.get_x() + bar.get_width() / 2.0)
        yB = float(bar.get_height())
        xA, yA = pos[v]

        con = ConnectionPatch(
            xyA=(xA, yA), coordsA=ax_net.transData,
            xyB=(xB, yB), coordsB=ax_bar.transData,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.2,
            color="black",
        )
        con.set_clip_on(False)
        con.set_zorder(10)
        fig.add_artist(con)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    plt.show()


def all_k_cliques(G: nx.Graph, k: int) -> List[Tuple[int, ...]]:

    cliques_k = []
    for c in nx.enumerate_all_cliques(G):
        if len(c) < k:
            continue
        if len(c) == k:
            cliques_k.append(tuple(sorted(int(v) for v in c)))
        else:
            continue
    return sorted(list(set(cliques_k)))


def choose_k_clique_shapes(G: nx.Graph, k: int = 3, num_cliques: int = 8):

    cliques_k = all_k_cliques(G, k=k)
    if len(cliques_k) == 0:
        return [], set(), {int(v): [] for v in G.nodes()}, []

    def score_clique(c: Tuple[int, ...]) -> int:
        return sum(int(G.degree(int(v))) for v in c)

    scored = [(c, score_clique(c)) for c in cliques_k]
    scored.sort(key=lambda x: (-x[1], x[0]))

    chosen = scored[: min(int(num_cliques), len(scored))]
    cliques = [c for c, _ in chosen]
    clique_scores = [int(s) for _, s in chosen]

    W = set()
    node_to_clique_ids = {int(v): [] for v in G.nodes()}
    for cid, c in enumerate(cliques):
        for v in c:
            W.add(int(v))
            node_to_clique_ids[int(v)].append(int(cid))

    return cliques, W, node_to_clique_ids, clique_scores


class ShapeConstrainedSimulatorCliquePerStepWeights:

    def __init__(
        self,
        base_graph: nx.Graph,
        p_edge_on: float,
        alpha_stop: float,
        k_min: int,
        radius: int,
        k_filter: int,
        cliques: List[Tuple[int, ...]],
        W: set,
        fallback_node: int,
        seed: int = 0,
        mu_fraction: float = 0.50,
        sigma_fraction: float = 1.0 / 6.0,
        weight_seed: int = 12345,
    ):
        self.base_graph = base_graph
        self.n = base_graph.number_of_nodes()

        self.alpha_stop = float(alpha_stop)
        self.k_min = int(k_min)
        self.radius = int(radius)
        self.k_filter = int(k_filter)

        self.cliques = cliques
        self.W = set(int(v) for v in W)
        self.v_fb = int(fallback_node)

        self.core = OneStepSimulatorPerStepWeights(
            base_graph=base_graph,
            p_edge_on=p_edge_on,
            k_min=k_min,
            alpha_stop=alpha_stop,
            seed=seed,
            use_weights=True,
            local_mode="radius",
            radius=radius,
            mu_fraction=mu_fraction,
            sigma_fraction=sigma_fraction,
            weight_seed=weight_seed,
        )

    def sample_next(self, i: int) -> Optional[int]:
        if self.core.rng.random() < self.core.alpha_stop:
            return None

        H = self.core.sample_realized_working_graph()

        # Local nodes (radius ball)
        ball = nx.single_source_shortest_path_length(H, int(i), cutoff=int(self.radius))
        local_nodes = list(ball.keys())
        if len(local_nodes) < self.k_min:
            return None

        sub = H.subgraph(local_nodes)
        bc = nx.betweenness_centrality(sub, normalized=False, weight="weight")
        topk_nodes, _ = topk_from_dict({int(v): float(val) for v, val in bc.items()}, k=self.k_filter)

        cand = [int(v) for v in topk_nodes if int(v) in self.W]
        if len(cand) > 0:
            return int(cand[0])
        return int(self.v_fb)


def plot_kclique_network_and_bars_with_arrows(
    G_base: nx.Graph,
    name_map: Dict[int, str],
    top5_nodes: List[int],
    top_items: List[Tuple[int, float]],
    cliques: List[Tuple[int, ...]],
    node_to_clique_ids: Dict[int, List[int]],
    clique_scores: List[int],
    title: str,
    layout_seed: int = 42,
    dpi: int = 600,
    savepath: Optional[str] = None,
    bar_value_format: str = "{:.4f}",
) -> None:
  
    pos = nx.spring_layout(
        G_base,
        seed=int(layout_seed),
        k=1.2 / np.sqrt(max(G_base.number_of_nodes(), 1)),
        iterations=600,
    )

    clique_ids_to_draw = set()
    for v in top5_nodes:
        ids = node_to_clique_ids.get(int(v), [])
        if not ids:
            continue
        best_id = max(ids, key=lambda cid: (clique_scores[int(cid)], -int(cid)))
        clique_ids_to_draw.add(int(best_id))

    clique_edges = []
    for cid in sorted(list(clique_ids_to_draw)):
        c = cliques[int(cid)]
        for u, w in itertools.combinations(c, 2):
            if G_base.has_edge(int(u), int(w)):
                clique_edges.append((int(u), int(w)))

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.06)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    n = G_base.number_of_nodes()
    base_node_size = max(10, int(1800 / max(n, 1)))
    highlight_size = base_node_size * 18

    nx.draw_networkx_edges(G_base, pos, ax=ax_net, alpha=0.20, width=0.7)
    nx.draw_networkx_nodes(G_base, pos, ax=ax_net, node_size=base_node_size)

    if clique_edges:
        nx.draw_networkx_edges(G_base, pos, ax=ax_net, edgelist=clique_edges, width=2.6, edge_color="green", alpha=0.9)

    nx.draw_networkx_nodes(
        G_base, pos, ax=ax_net,
        nodelist=top5_nodes,
        node_size=highlight_size,
        node_shape="s",
        node_color="red",
    )

    labels = {int(v): name_map.get(int(v), str(int(v))) for v in top5_nodes}
    nx.draw_networkx_labels(G_base, pos, ax=ax_net, labels=labels, font_size=8)

    ax_net.set_title(title)
    ax_net.axis("off")

    bar_nodes = [int(v) for v, _ in top_items]
    bar_vals = [float(val) for _, val in top_items]
    x = np.arange(len(bar_nodes), dtype=float)
    bars = ax_bar.bar(x, bar_vals)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([name_map.get(v, str(v)) for v in bar_nodes], rotation=90, ha="center", fontsize=8)
    ax_bar.set_ylabel("AFC value b(s)")
    ax_bar.set_xlabel("Character (Top-ranked by b)")

    y_max = max([1e-12] + bar_vals)
    ax_bar.set_ylim(0.0, y_max * 1.25)

    annotate_bar_values(ax_bar, bars, fmt=bar_value_format, fontsize=7)
    fig.subplots_adjust(bottom=0.35)

    node_to_bar_index = {v: i for i, v in enumerate(bar_nodes)}
    for v in top5_nodes:
        if v not in node_to_bar_index:
            continue
        idx = node_to_bar_index[v]
        bar = bars[idx]

        xB = float(bar.get_x() + bar.get_width() / 2.0)
        yB = float(bar.get_height())
        xA, yA = pos[v]

        con = ConnectionPatch(
            xyA=(xA, yA), coordsA=ax_net.transData,
            xyB=(xB, yB), coordsB=ax_bar.transData,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.2,
            color="black",
        )
        con.set_clip_on(False)
        con.set_zorder(10)
        fig.add_artist(con)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    plt.show()

def run_lesmis_baseline_afc(
    seed_sim: int = 42,
    seed_weights: int = 12345,
    p_edge_on: float = 1.0,
    alpha_stop: float = 0.15,
    radius: int = 2,
    k_min: int = 5,
    mu_fraction: float = 0.50,
    sigma_fraction: float = 1.0 / 6.0,
    M: int = 60,
    top_k: int = 10,
    highlight_k: int = 5,
    dpi: int = 600,
    save: bool = True,
):
    G_base, name_map, w_min, w_max = load_lesmis_integer_graph()
    print("\n=== Les Miserables | Baseline AFC (per-step weights) ===")
    print(f"Nodes={G_base.number_of_nodes()}, Edges={G_base.number_of_edges()}")
    print(f"Original edge weight range: min={w_min}, max={w_max}")
    print(f"Per-step Normal: mu_fraction={mu_fraction:.4f}, sigma_fraction={sigma_fraction:.4f}")
    print(f"AMC params: p_edge_on={p_edge_on}, alpha_stop={alpha_stop}, radius={radius}, k_min={k_min}, M={M}")

    sim = OneStepSimulatorPerStepWeights(
        base_graph=G_base,
        p_edge_on=p_edge_on,
        k_min=k_min,
        alpha_stop=alpha_stop,
        seed=seed_sim,
        use_weights=True,
        local_mode="radius",
        radius=radius,
        mu_fraction=mu_fraction,
        sigma_fraction=sigma_fraction,
        weight_seed=seed_weights,
    )

    P_hat = estimate_amc_kernel(sim, M=M, absorb_floor=1e-6)
    n = sim.n
    s = np.ones(n, dtype=float) / n
    b = afc_from_kernel(P_hat, s)

    top_items = topk_from_vector(b, k=top_k)

    print(f"\nTop-{len(top_items)} by b(s):")
    for r, (v, val) in enumerate(top_items, 1):
        print(f"  {r:>2d}. {name_map.get(v, str(v)):<18s}  b={val:.6f}")

    savepath = None
    if save:
        savepath = f"lesmis_afc_top{len(top_items)}_highlight{min(highlight_k, len(top_items))}_seed{seed_sim}_dpi{dpi}.png"

    plot_baseline_network_and_bars_with_arrows(
        G_base=G_base,
        name_map=name_map,
        top_items=top_items,
        highlight_k=highlight_k,
        title=f"Les Mis | Baseline AFC Top-{len(top_items)} (highlight Top-{min(highlight_k, len(top_items))})",
        layout_seed=seed_sim,
        dpi=dpi,
        savepath=savepath,
        bar_value_format="{:.4f}",
    )

    if save:
        print(f"Saved figure: {savepath}")

    return G_base, name_map, P_hat, b


def run_lesmis_robust_kl_w1(
    seed_sim: int = 42,
    seed_weights: int = 12345,
    p_edge_on: float = 1.0,
    alpha_stop: float = 0.15,
    radius: int = 2,
    k_min: int = 5,
    mu_fraction: float = 0.50,
    sigma_fraction: float = 1.0 / 6.0,
    M: int = 60,
    delta_rel: float = 0.50,
    r_lower: float = 0.05,
    robust_samples_kl: int = 100,
    robust_samples_w1: int = 100,
    top_n_bars: int = 10,
    dpi: int = 600,
    save: bool = True,
):
    G_base, name_map, w_min, w_max = load_lesmis_integer_graph()
    print("\n=== Les Miserables | Robust KL vs W1 (per-step weights) ===")
    print(f"Nodes={G_base.number_of_nodes()}, Edges={G_base.number_of_edges()}")
    print(f"Original edge weight range: min={w_min}, max={w_max}")
    print(f"Per-step Normal: mu_fraction={mu_fraction:.4f}, sigma_fraction={sigma_fraction:.4f}")
    print(f"AMC params: p_edge_on={p_edge_on}, alpha_stop={alpha_stop}, radius={radius}, k_min={k_min}, M={M}")

    sim = OneStepSimulatorPerStepWeights(
        base_graph=G_base,
        p_edge_on=p_edge_on,
        k_min=k_min,
        alpha_stop=alpha_stop,
        seed=seed_sim,
        use_weights=True,
        local_mode="radius",
        radius=radius,
        mu_fraction=mu_fraction,
        sigma_fraction=sigma_fraction,
        weight_seed=seed_weights,
    )
    P0 = estimate_amc_kernel(sim, M=M, absorb_floor=1e-6)

    n = sim.n
    s = np.ones(n, dtype=float) / n
    b0 = afc_from_kernel(P0, s)

    cost = ground_distance_matrix_unweighted(G_base)

    _, b_kl, kl_star, _ = find_robust_kernel_random_search(
        P0=P0,
        s=s,
        delta_rel=delta_rel,
        r_lower=r_lower,
        num_samples=robust_samples_kl,
        objective="kl",
        cost=cost,
        seed=seed_sim + 100,
        w1_flow_scale_search=1000,
    )

    _, b_w1, _, _ = find_robust_kernel_random_search(
        P0=P0,
        s=s,
        delta_rel=delta_rel,
        r_lower=r_lower,
        num_samples=robust_samples_w1,
        objective="w1",
        cost=cost,
        seed=seed_sim + 200,
        w1_flow_scale_search=1000,
    )

    w1_star = wasserstein1_cost_matrix(b0, b_w1, cost, flow_scale=5000)

    top5_kl = [v for v, _ in topk_from_vector(b_kl, 5)]
    top5_w1 = [v for v, _ in topk_from_vector(b_w1, 5)]

    print("\nNominal/robust summary:")
    print(f"  KL*={kl_star:.6f}, KL Top-5:", [name_map.get(v, str(v)) for v in top5_kl])
    print(f"  W1*={w1_star:.6f}, W1 Top-5:", [name_map.get(v, str(v)) for v in top5_w1])

    include_nodes = set(top5_kl) | set(top5_w1)
    score = np.maximum(np.asarray(b_kl, float), np.asarray(b_w1, float))
    ranked_all = sorted(range(n), key=lambda i: (-float(score[i]), int(i)))

    nodes_bar = []
    for v in sorted(include_nodes, key=lambda i: (-float(score[int(i)]), int(i))):
        if v not in nodes_bar:
            nodes_bar.append(int(v))
    for v in ranked_all:
        if v not in nodes_bar:
            nodes_bar.append(int(v))
        if len(nodes_bar) >= max(int(top_n_bars), len(include_nodes)):
            break
    nodes_bar = nodes_bar[: max(int(top_n_bars), len(include_nodes))]

    savepath = None
    if save:
        savepath = f"lesmis_robust_kl_w1_bars_arrows_dpi{dpi}.png"

    plot_robust_kl_w1_network_and_bars_with_arrows(
        G_base=G_base,
        name_map=name_map,
        top5_kl=top5_kl,
        top5_w1=top5_w1,
        b_kl=b_kl,
        b_w1=b_w1,
        nodes_bar=nodes_bar,
        title="Les Mis | KL Top-5 vs W1 Top-5 (Robust AMC/AFC, per-step weights)",
        layout_seed=seed_sim,
        dpi=dpi,
        savepath=savepath,
        bar_value_format="{:.4f}",
    )

    if save:
        print(f"Saved figure: {savepath}")

    return G_base, name_map, P0, b0, b_kl, b_w1


def run_lesmis_multi_reward(
    seed_sim: int = 42,
    seed_weights: int = 12345,
    p_edge_on: float = 1.0,
    alpha_stop: float = 0.15,
    radius: int = 2,
    k_min: int = 5,
    mu_fraction: float = 0.50,
    sigma_fraction: float = 1.0 / 6.0,
    M: int = 60,
    reward_hubs_m: int = 5,
    decay: float = 0.60,
    base_rewards: Tuple[float, ...] = (10.0, 10.0, 10.0, 10.0, 10.0),
    top_bar_k: int = 10,
    dpi: int = 600,
    save: bool = True,
):
    G_base, name_map, w_min, w_max = load_lesmis_integer_graph()
    print("\n=== Les Miserables | Multi-reward (per-step weights) ===")
    print(f"Nodes={G_base.number_of_nodes()}, Edges={G_base.number_of_edges()}")
    print(f"Original edge weight range: min={w_min}, max={w_max}")
    print(f"Per-step Normal: mu_fraction={mu_fraction:.4f}, sigma_fraction={sigma_fraction:.4f}")
    print(f"AMC params: p_edge_on={p_edge_on}, alpha_stop={alpha_stop}, radius={radius}, k_min={k_min}, M={M}")

    sim = OneStepSimulatorPerStepWeights(
        base_graph=G_base,
        p_edge_on=p_edge_on,
        k_min=k_min,
        alpha_stop=alpha_stop,
        seed=seed_sim,
        use_weights=True,
        local_mode="radius",
        radius=radius,
        mu_fraction=mu_fraction,
        sigma_fraction=sigma_fraction,
        weight_seed=seed_weights,
    )
    P_hat = estimate_amc_kernel(sim, M=M, absorb_floor=1e-6)

    n = sim.n
    s = np.ones(n, dtype=float) / n
    b, N, Q = afc_from_kernel_full(P_hat, s)

    # Rewards
    hubs = pick_reward_hubs_by_degree(G_base, m=reward_hubs_m)
    base_rewards_list = list(base_rewards)[: int(reward_hubs_m)]
    if len(base_rewards_list) < int(reward_hubs_m) and len(base_rewards_list) > 0:
        base_rewards_list += [base_rewards_list[-1]] * (int(reward_hubs_m) - len(base_rewards_list))
    elif int(reward_hubs_m) > 0 and len(base_rewards_list) == 0:
        base_rewards_list = [1.0] * int(reward_hubs_m)

    f = build_distance_decayed_rewards(G_base, hubs=hubs, base_rewards=base_rewards_list, decay=decay)

    b_f = reward_afc_from_b(b, f)
    sw = transition_reward_switch_rate(P_hat, s, N, Q)
    imp = transition_reward_improvement(P_hat, s, N, Q, f)

    ranked = rank_vector_desc(b)
    k_bar = max(5, int(top_bar_k))
    top_items = ranked[: min(k_bar, len(ranked))]

    print("\nTop-5 by b(s):", [name_map.get(v, str(v)) for v, _ in top_items[:5]])
    print("Reward hubs:", [name_map.get(h, str(h)) for h in hubs])
    print(f"Node-reward AFC b_f(s)={b_f:.6f}")
    print(f"Transition reward (switch rate) b_eta_sw={sw:.6f}")
    print(f"Transition reward (improvement) b_eta_imp={imp:.6f}")

    savepath = None
    if save:
        savepath = f"LES_multi_reward_network_bars_arrows_600dpi.png"

    plot_multi_reward_network_and_bars_with_arrows(
        G_base=G_base,
        name_map=name_map,
        b=b,
        hubs=hubs,
        f=f,
        top_items=top_items,
        title="LES | Top-5 by b(s) + high-reward hubs (per-step weights)",
        layout_seed=seed_sim,
        dpi=dpi,
        savepath=savepath,
        bar_value_format="{:.4f}",
    )

    if save:
        print(f"Saved figure: {savepath}")

    return G_base, name_map, P_hat, b


def run_lesmis_kclique_constraint(
    seed_sim: int = 42,
    seed_weights: int = 12345,
    p_edge_on: float = 1.0,
    alpha_stop: float = 0.15,
    radius: int = 2,
    k_min: int = 5,
    mu_fraction: float = 0.50,
    sigma_fraction: float = 1.0 / 6.0,
    M: int = 60,
    clique_k: int = 3,
    num_cliques: int = 8,
    k_filter: int = 5,
    top_bar_k: int = 10,
    dpi: int = 600,
    save: bool = True,
):
    G_base, name_map, w_min, w_max = load_lesmis_integer_graph()
    print("\n=== Les Miserables | k-clique constraint (per-step weights) ===")
    print(f"Nodes={G_base.number_of_nodes()}, Edges={G_base.number_of_edges()}")
    print(f"Original edge weight range: min={w_min}, max={w_max}")
    print(f"Per-step Normal: mu_fraction={mu_fraction:.4f}, sigma_fraction={sigma_fraction:.4f}")
    print(f"AMC params: p_edge_on={p_edge_on}, alpha_stop={alpha_stop}, radius={radius}, k_min={k_min}, M={M}")

    cliques, W, node_to_clique_ids, clique_scores = choose_k_clique_shapes(G_base, k=clique_k, num_cliques=num_cliques)
    if len(cliques) == 0:
        print(f"No {clique_k}-cliques found; try clique_k=3 (triangles) or adjust num_cliques.")
        return None

    v_fb = min(cliques[0])
    print(f"Selected {len(cliques)} cliques, |W|={len(W)}, fallback v_fb={name_map.get(v_fb, str(v_fb))}")

    sim = ShapeConstrainedSimulatorCliquePerStepWeights(
        base_graph=G_base,
        p_edge_on=p_edge_on,
        alpha_stop=alpha_stop,
        k_min=k_min,
        radius=radius,
        k_filter=k_filter,
        cliques=cliques,
        W=W,
        fallback_node=v_fb,
        seed=seed_sim,
        mu_fraction=mu_fraction,
        sigma_fraction=sigma_fraction,
        weight_seed=seed_weights,
    )

    class _Adapter:
        def __init__(self, sim_obj, n):
            self.sim_obj = sim_obj
            self.n = n
        def sample_next(self, i):
            return self.sim_obj.sample_next(i)

    P_hat = estimate_amc_kernel(_Adapter(sim, G_base.number_of_nodes()), M=M, absorb_floor=1e-6)

    n = G_base.number_of_nodes()
    s = np.ones(n, dtype=float) / n
    b = afc_from_kernel(P_hat, s)

    ranked = rank_vector_desc(b)
    top5_nodes = [int(v) for v, _ in ranked[:5]]

    k_bar = max(5, int(top_bar_k))
    top_items = ranked[: min(k_bar, len(ranked))]

    print("Top-5 under clique constraint:", [name_map.get(v, str(v)) for v in top5_nodes])

    savepath = None
    if save:
        savepath = f"LES_kclique_{clique_k}_combined_600dpi.png"

    plot_kclique_network_and_bars_with_arrows(
        G_base=G_base,
        name_map=name_map,
        top5_nodes=top5_nodes,
        top_items=top_items,
        cliques=cliques,
        node_to_clique_ids=node_to_clique_ids,
        clique_scores=clique_scores,
        title=f"LES | Top-5 under {clique_k}-clique constraint (per-step weights)",
        layout_seed=seed_sim,
        dpi=dpi,
        savepath=savepath,
        bar_value_format="{:.4f}",
    )

    if save:
        print(f"Saved figure: {savepath}")

    return G_base, name_map, P_hat, b


if __name__ == "__main__":

    SEED_SIM = 42
    SEED_WEIGHTS = 12345

    MU_FRACTION = 0.50
    SIGMA_FRACTION = 1.0 / 6.0

    P_EDGE_ON = 1.0      
    ALPHA_STOP = 0.15
    RADIUS = 2
    K_MIN = 5

    M_KERNEL = 60

    DPI = 600

    run_lesmis_baseline_afc(
        seed_sim=SEED_SIM,
        seed_weights=SEED_WEIGHTS,
        p_edge_on=P_EDGE_ON,
        alpha_stop=ALPHA_STOP,
        radius=RADIUS,
        k_min=K_MIN,
        mu_fraction=MU_FRACTION,
        sigma_fraction=SIGMA_FRACTION,
        M=M_KERNEL,
        top_k=10,
        highlight_k=5,  
        dpi=DPI,
        save=True,
    )

    run_lesmis_robust_kl_w1(
        seed_sim=SEED_SIM,
        seed_weights=SEED_WEIGHTS,
        p_edge_on=P_EDGE_ON,
        alpha_stop=ALPHA_STOP,
        radius=RADIUS,
        k_min=K_MIN,
        mu_fraction=MU_FRACTION,
        sigma_fraction=SIGMA_FRACTION,
        M=M_KERNEL,
        delta_rel=0.50,
        r_lower=0.05,
        robust_samples_kl=100,
        robust_samples_w1=100,
        top_n_bars=10,
        dpi=DPI,
        save=True,
    )

    run_lesmis_multi_reward(
        seed_sim=SEED_SIM,
        seed_weights=SEED_WEIGHTS,
        p_edge_on=P_EDGE_ON,
        alpha_stop=ALPHA_STOP,
        radius=RADIUS,
        k_min=K_MIN,
        mu_fraction=MU_FRACTION,
        sigma_fraction=SIGMA_FRACTION,
        M=M_KERNEL,
        reward_hubs_m=5,
        decay=0.60,
        base_rewards=(10.0, 10.0, 10.0, 10.0, 10.0),
        top_bar_k=10,
        dpi=DPI,
        save=True,
    )

    run_lesmis_kclique_constraint(
        seed_sim=SEED_SIM,
        seed_weights=SEED_WEIGHTS,
        p_edge_on=P_EDGE_ON,
        alpha_stop=ALPHA_STOP,
        radius=RADIUS,
        k_min=K_MIN,
        mu_fraction=MU_FRACTION,
        sigma_fraction=SIGMA_FRACTION,
        M=M_KERNEL,
        clique_k=3,
        num_cliques=8,
        k_filter=5,
        top_bar_k=10,
        dpi=DPI,
        save=True,
    )
