import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from typing import Optional


def argmax_with_tiebreak(scores_dict):
    best_node = None
    best_val = None
    for v, val in scores_dict.items():
        if best_node is None:
            best_node, best_val = v, val
        else:
            if (val > best_val) or (val == best_val and v < best_node):
                best_node, best_val = v, val
    return best_node

class OneStepSimulator:
    def __init__(
        self,
        base_graph: nx.Graph,
        p_edge_on: float,
        k_min: int,
        alpha_stop: float,
        seed: int = 0,
        use_weights: bool = False,
        weight_low: float = 1.0,
        weight_high: float = 10.0,
    ):
        self.G0 = base_graph
        self.n = self.G0.number_of_nodes()
        self.nodes = list(self.G0.nodes())

        edges = list(self.G0.edges())
        self.edges_arr = (
            np.array(edges, dtype=int) if len(edges) > 0 else np.zeros((0, 2), dtype=int)
        )
        self.m = self.edges_arr.shape[0]

        self.p_edge_on = float(p_edge_on)
        self.k_min = int(k_min)
        self.alpha_stop = float(alpha_stop)

        self.use_weights = bool(use_weights)
        self.weight_low = float(weight_low)
        self.weight_high = float(weight_high)

        self.rng = np.random.default_rng(seed)

    def sample_realized_working_graph(self):

        H = nx.Graph()
        H.add_nodes_from(self.nodes)

        if self.m == 0:
            return H

        keep_mask = self.rng.random(self.m) < self.p_edge_on
        kept_edges = self.edges_arr[keep_mask]

        if kept_edges.size > 0:
            if not self.use_weights:
                H.add_edges_from(kept_edges.tolist())
            else:
                weights = self.rng.uniform(
                    self.weight_low, self.weight_high, size=kept_edges.shape[0]
                )
                for (u, v), w in zip(kept_edges.tolist(), weights.tolist()):
                    H.add_edge(u, v, weight=float(w))

        return H

    def local_center(self, anchor_i: int, H: nx.Graph):
        comp = set(nx.node_connected_component(H, anchor_i))
        if len(comp) < self.k_min:
            return None  # absorb

        sub = H.subgraph(comp)

        if self.use_weights:
            bc = nx.betweenness_centrality(sub, normalized=False, weight="weight")
        else:
            bc = nx.betweenness_centrality(sub, normalized=False, weight=None)

        return argmax_with_tiebreak(bc)

    def sample_next(self, current_center_i: int):

        if self.rng.random() < self.alpha_stop:
            return None

        H = self.sample_realized_working_graph()
        return self.local_center(current_center_i, H)

def estimate_amc_kernel(sim: OneStepSimulator, M: int, absorb_floor: float = 1e-6):
    n = sim.n
    ABS = n
    P_hat = np.zeros((n + 1, n + 1), dtype=float)

    for i in range(n):
        counts = np.zeros(n + 1, dtype=int)
        for _ in range(M):
            z = sim.sample_next(i)
            if z is None:
                counts[ABS] += 1
            else:
                counts[z] += 1

        row = counts / M

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

def afc_from_kernel(P_hat: np.ndarray, s: np.ndarray):
    n = len(s)
    Q = P_hat[:n, :n]
    N = np.linalg.inv(np.eye(n) - Q)
    mu = s @ N
    return mu / mu.sum()

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def ground_distance_matrix(G: nx.Graph):
    n = G.number_of_nodes()
    dist = np.full((n, n), np.inf)

    for s, lengths in nx.all_pairs_shortest_path_length(G):
        for t, l in lengths.items():
            dist[s, t] = l

    finite = dist[np.isfinite(dist)]
    max_finite = finite.max() if finite.size > 0 else 0.0
    D = int(max_finite) + 1
    dist[np.isinf(dist)] = D
    return dist.astype(int)


def wasserstein1_cost_matrix(p, q, cost, flow_scale=3000):

    n = len(p)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    q = np.asarray(q, dtype=float)
    q = q / q.sum()

    supply = np.round(p * flow_scale).astype(int)
    demand = np.round(q * flow_scale).astype(int)

    diff = supply.sum() - demand.sum()
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

def sample_perturbed_kernel(P0, delta_rel, r_lower, rng):
    P0 = np.asarray(P0, dtype=float)
    n = P0.shape[0] - 1
    Q0 = P0[:n, :n]

    P = np.zeros_like(P0)

    for i in range(n):
        q0 = Q0[i].copy()
        u = rng.uniform(-delta_rel, delta_rel, size=n)
        q = np.clip(q0 * (1.0 + u), 0.0, 1.0)

        ssum = q.sum()
        if ssum > 1.0 - r_lower:
            if ssum > 0:
                q *= (1.0 - r_lower) / ssum
            else:
                q[:] = 0.0

        r = 1.0 - q.sum()
        if r < r_lower:
            if q.sum() > 0:
                q *= (1.0 - r_lower) / q.sum()
            r = 1.0 - q.sum()

        P[i, :n] = q
        P[i, n] = r

    P[n, n] = 1.0
    return P


def find_robust_kernel_random_search(
    P0,
    s,
    delta_rel,
    r_lower,
    num_samples,
    objective,
    cost=None,
    seed=0,
    w1_flow_scale_search=1200,
):
    rng = np.random.default_rng(seed)
    b0 = afc_from_kernel(P0, s)

    best_val = -1e18
    best_P = None
    best_b = None

    for _ in range(num_samples):
        P = sample_perturbed_kernel(P0, delta_rel, r_lower, rng)
        b = afc_from_kernel(P, s)

        if objective == "kl":
            val = kl_divergence(b0, b)
        elif objective == "w1":
            if cost is None:
                raise ValueError("objective='w1' requires cost matrix")
            val = wasserstein1_cost_matrix(b0, b, cost, flow_scale=w1_flow_scale_search)
        else:
            raise ValueError("objective must be 'kl' or 'w1'")

        if val > best_val:
            best_val = val
            best_P = P
            best_b = b

    return best_P, best_b, float(best_val), b0


def drop_isolates_except(G: nx.Graph, keep_nodes=None):

    if keep_nodes is None:
        keep_nodes = set()
    keep_nodes = set(keep_nodes)

    isolates = set(nx.isolates(G))
    drop = list(isolates - keep_nodes)

    H = G.copy()
    H.remove_nodes_from(drop)
    return H, drop


def packed_component_layout_grid(G: nx.Graph, seed: int = 0, comp_scale: float = 7.0, pad: float = 3.0):

    rng = np.random.default_rng(seed)
    comps = [list(c) for c in nx.connected_components(G)]
    comps.sort(key=len, reverse=True)

    k = len(comps)
    cols = int(np.ceil(np.sqrt(k))) if k > 0 else 1
    cell = 2.0 * comp_scale + pad

    pos = {}
    for idx, nodes in enumerate(comps):
        Hc = G.subgraph(nodes)

        if len(nodes) == 1:
            coords = np.array([[0.0, 0.0]])
            order = nodes
        else:
            pos_raw = nx.spring_layout(
                Hc,
                seed=int(rng.integers(1_000_000_000)),
                k=6.5 / np.sqrt(len(nodes)),
                iterations=1200,
            )
            order = list(pos_raw.keys())
            coords = np.array([pos_raw[v] for v in order], dtype=float)
            coords -= coords.mean(axis=0)
            span = float(np.max(np.ptp(coords, axis=0)))
            if span > 0:
                coords = coords / span * comp_scale

        r = idx // cols
        c = idx % cols
        shift = np.array([c * cell, -r * cell], dtype=float)
        coords = coords + shift

        for v, xy in zip(order, coords):
            pos[v] = (float(xy[0]), float(xy[1]))

    if len(pos) == 0:
        return pos

    all_xy = np.array(list(pos.values()), dtype=float)
    all_xy -= all_xy.mean(axis=0)
    keys = list(pos.keys())
    for i, v in enumerate(keys):
        pos[v] = (float(all_xy[i, 0]), float(all_xy[i, 1]))

    return pos


def set_axis_limits_from_pos(ax, pos, margin=0.06):
    xs = np.array([xy[0] for xy in pos.values()], dtype=float)
    ys = np.array([xy[1] for xy in pos.values()], dtype=float)
    xspan = xs.max() - xs.min()
    yspan = ys.max() - ys.min()
    if xspan == 0:
        xspan = 1.0
    if yspan == 0:
        yspan = 1.0
    ax.set_xlim(xs.min() - margin * xspan, xs.max() + margin * xspan)
    ax.set_ylim(ys.min() - margin * yspan, ys.max() + margin * yspan)


def rank_topk(b, k=5):
    ranked = sorted([(i, float(b[i])) for i in range(len(b))], key=lambda x: (-x[1], x[0]))
    return [v for v, _ in ranked[:k]], ranked


def select_nodes_for_bar_including(b_kl, b_w1, include_nodes, top_n=10):
    b_kl = np.asarray(b_kl, dtype=float)
    b_w1 = np.asarray(b_w1, dtype=float)
    score = np.maximum(b_kl, b_w1)
    n = len(score)

    include_nodes = [int(v) for v in include_nodes if 0 <= int(v) < n]
    include_nodes = sorted(set(include_nodes), key=lambda i: (-score[i], i))

    top_n = max(int(top_n), len(include_nodes))

    ranked_all = sorted(range(n), key=lambda i: (-score[i], i))

    out = []
    for v in include_nodes:
        if v not in out:
            out.append(v)
    for v in ranked_all:
        if v not in out:
            out.append(v)
        if len(out) >= top_n:
            break

    return out[:top_n]


def plot_dual_objective_network_and_bars_with_arrows(
    G_base: nx.Graph,
    top5_kl,
    top5_w1,
    b_kl,
    b_w1,
    nodes_bar,
    title_prefix: str,
    layout_seed: int = 0,
    dpi: int = 600,
    savepath: Optional[str] = None,
):

    set_kl = set(top5_kl)
    set_w1 = set(top5_w1)
    highlight_nodes = set_kl | set_w1

    G_plot, dropped_isolates = drop_isolates_except(G_base, keep_nodes=highlight_nodes)

    if G_plot.number_of_nodes() == 0:
        print("Plot skipped: graph is empty after dropping isolates.")
        return

    pos = packed_component_layout_grid(G_plot, seed=layout_seed, comp_scale=7.0, pad=3.0)

    set_kl = set_kl & set(G_plot.nodes())
    set_w1 = set_w1 & set(G_plot.nodes())
    highlight_nodes = set_kl | set_w1

    both = sorted(list(set_kl & set_w1))
    only_kl = sorted(list(set_kl - set_w1))
    only_w1 = sorted(list(set_w1 - set_kl))

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.06)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    n_plot = G_plot.number_of_nodes()
    base_node_size = max(10, int(1800 / max(n_plot, 1)))
    highlight_size = base_node_size * 18

    nx.draw_networkx_edges(G_plot, pos, ax=ax_net, alpha=0.22, width=0.7)
    nx.draw_networkx_nodes(G_plot, pos, ax=ax_net, node_size=base_node_size)

    if only_kl:
        nx.draw_networkx_nodes(
            G_plot, pos, ax=ax_net,
            nodelist=only_kl,
            node_size=highlight_size,
            node_shape="s",
            node_color="red",
        )
    if only_w1:
        nx.draw_networkx_nodes(
            G_plot, pos, ax=ax_net,
            nodelist=only_w1,
            node_size=highlight_size,
            node_shape="^",
            node_color="green",
        )
    if both:
        nx.draw_networkx_nodes(
            G_plot, pos, ax=ax_net,
            nodelist=both,
            node_size=highlight_size * 1.15,
            node_shape="*",
            node_color="yellow",
            edgecolors="black",
            linewidths=0.8,
        )
    union_nodes = sorted(list(highlight_nodes))
    nx.draw_networkx_labels(
        G_plot, pos, ax=ax_net,
        labels={v: str(v) for v in union_nodes},
        font_size=9,
    )

    ax_net.set_title(title_prefix)
    ax_net.axis("off")
    set_axis_limits_from_pos(ax_net, pos, margin=0.05)

    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color="red", marker="s", linestyle="None", markersize=10, label="KL Top-5 only"),
        mlines.Line2D([], [], color="green", marker="^", linestyle="None", markersize=10, label="W1 Top-5 only"),
        mlines.Line2D([], [], color="yellow", marker="*", markeredgecolor="black",
                      linestyle="None", markersize=12, label="In both Top-5"),
    ]
    ax_net.legend(handles=legend_handles, loc="upper right", frameon=True)

    nodes_bar = list(nodes_bar)
    x = np.arange(len(nodes_bar), dtype=float)
    width = 0.42

    vals_kl = [float(b_kl[i]) for i in nodes_bar]
    vals_w1 = [float(b_w1[i]) for i in nodes_bar]

    ax_bar.bar(x - width / 2, vals_kl, width, label="KL-robust b", color="orange")
    ax_bar.bar(x + width / 2, vals_w1, width, label="W1-robust b", color="blue")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(i) for i in nodes_bar], rotation = 0, ha="center")

    ax_bar.set_ylabel("AFC b(s)")
    ax_bar.set_xlabel("Node id (selected)")
    ax_bar.legend()

    y_max = max([1e-12] + vals_kl + vals_w1)
    ax_bar.set_ylim(0.0, y_max * 1.15)

    fig.subplots_adjust(bottom=0.22)

    node_to_bar_index = {v: idx for idx, v in enumerate(nodes_bar)}

    for v in union_nodes:
        if v not in node_to_bar_index:
            continue
        if v not in pos:
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
    return fig

def make_base_graph(model: str, n: int, seed: int):
    if model == "ER":
        return nx.erdos_renyi_graph(n=n, p=0.08, seed=seed)
    if model == "WS":
        return nx.watts_strogatz_graph(n=n, k=6, p=0.10, seed=seed)
    if model == "BA":
        return nx.barabasi_albert_graph(n=n, m=3, seed=seed)
    raise ValueError("Unknown model: choose from {'ER','WS','BA'}")

def run_demo(
    model: str,
    n=60,
    seed=42,
    p_edge_on=0.85,
    k_min=5,
    alpha_stop=0.15,
    M=60,
    delta_rel=0.50,
    r_lower=0.05,
    robust_samples_kl=25,
    robust_samples_w1=15,
    top_n_bars=10,
    dpi=600,
    save=True,
):

    G_base = make_base_graph(model, n=n, seed=seed)
    isolates = list(nx.isolates(G_base))
    print(f"\n=== {model} ===")
    print(
        f"Base: nodes={G_base.number_of_nodes()}, edges={G_base.number_of_edges()}, "
        f"components={nx.number_connected_components(G_base)}, isolates={len(isolates)}"
    )

    sim = OneStepSimulator(
        G_base, p_edge_on=p_edge_on, k_min=k_min, alpha_stop=alpha_stop, seed=seed + 1
    )

    P0 = estimate_amc_kernel(sim, M=M, absorb_floor=1e-6)
    s = np.ones(n) / n
    b0 = afc_from_kernel(P0, s)

    cost = ground_distance_matrix(G_base)

    _, b_kl, kl_star, _ = find_robust_kernel_random_search(
        P0,
        s,
        delta_rel=delta_rel,
        r_lower=r_lower,
        num_samples=robust_samples_kl,
        objective="kl",
        cost=cost,
        seed=seed + 100,
        w1_flow_scale_search=1000,
    )

    _, b_w1, _, _ = find_robust_kernel_random_search(
        P0,
        s,
        delta_rel=delta_rel,
        r_lower=r_lower,
        num_samples=robust_samples_w1,
        objective="w1",
        cost=cost,
        seed=seed + 200,
        w1_flow_scale_search=1000,
    )
    w1_star = wasserstein1_cost_matrix(b0, b_w1, cost, flow_scale=5000)

    top5_kl, _ = rank_topk(b_kl, k=5)
    top5_w1, _ = rank_topk(b_w1, k=5)

    print(f"KL-robust: KL*={kl_star:.6f}, Top5={top5_kl}")
    print(f"W1-robust: W1*={w1_star:.6f}, Top5={top5_w1}")

    include_nodes = set(top5_kl) | set(top5_w1)
    nodes_bar = select_nodes_for_bar_including(b_kl, b_w1, include_nodes=include_nodes, top_n=top_n_bars)

    savepath = f"{model}_robust_network_bars_arrows_600dpi.png" if save else None

    plot_dual_objective_network_and_bars_with_arrows(
        G_base=G_base,
        top5_kl=top5_kl,
        top5_w1=top5_w1,
        b_kl=b_kl,
        b_w1=b_w1,
        nodes_bar=nodes_bar,
        title_prefix=f"{model} | KL Top-5 vs W1 Top-5",
        layout_seed=seed,
        dpi=dpi,
        savepath=savepath,
    )

    if save:
        print(f"Saved: {savepath}")


if __name__ == "__main__":

    for model in ["ER", "WS", "BA"]:
        run_demo(
            model=model,
            n=100,
            seed=42,
            p_edge_on=0.85,
            k_min=5,
            alpha_stop=0.15,
            M=60,
            delta_rel=0.50,
            r_lower=0.05,
            robust_samples_kl=100,
            robust_samples_w1=100,
            top_n_bars=10,
            dpi=600,
            save=True,
        )
