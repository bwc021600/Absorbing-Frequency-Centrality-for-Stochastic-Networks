import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from typing import Optional

def argmax_with_tiebreak(scores_dict):
    """argmax with tie-break: smaller node id wins"""
    best_node = None
    best_val = None
    for v, val in scores_dict.items():
        if best_node is None or (val > best_val) or (val == best_val and v < best_node):
            best_node, best_val = v, val
    return best_node

def rank_by_value(vec, top_k=5):
    """Deterministic ranking: (-value, node_id)"""
    ranked = sorted([(i, float(vec[i])) for i in range(len(vec))], key=lambda x: (-x[1], x[0]))
    return ranked[:top_k], ranked

def drop_isolates_and_relabel(G: nx.Graph):
    isolates = list(nx.isolates(G))
    H = G.copy()
    H.remove_nodes_from(isolates)

    H = nx.convert_node_labels_to_integers(H, ordering="sorted")
    return H, isolates


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

class OneStepSimulator:

    def __init__(self, base_graph, p_edge_on, k_min, alpha_stop, seed=0):
        self.G0 = base_graph
        self.n = self.G0.number_of_nodes()
        self.nodes = list(self.G0.nodes())

        edges = list(self.G0.edges())
        self.edges_arr = np.array(edges, dtype=int) if edges else np.zeros((0, 2), dtype=int)
        self.m = self.edges_arr.shape[0]

        self.p_edge_on = float(p_edge_on)
        self.k_min = int(k_min)
        self.alpha_stop = float(alpha_stop)
        self.rng = np.random.default_rng(seed)

    def sample_realized_working_graph(self):
 
        H = nx.Graph()
        H.add_nodes_from(self.nodes)
        if self.m == 0:
            return H
        keep = self.rng.random(self.m) < self.p_edge_on
        kept_edges = self.edges_arr[keep]
        if kept_edges.size > 0:
            H.add_edges_from(kept_edges.tolist())
        return H

    def local_center(self, anchor_i, H):
        comp = set(nx.node_connected_component(H, anchor_i))
        if len(comp) < self.k_min:
            return None
        sub = H.subgraph(comp)
        bc = nx.betweenness_centrality(sub, normalized=False, weight=None)
        return argmax_with_tiebreak(bc)

    def sample_next(self, i):

        if self.rng.random() < self.alpha_stop:
            return None
        H = self.sample_realized_working_graph()
        return self.local_center(i, H)  


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
                counts[int(z)] += 1

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
    return mu / mu.sum(), N, Q


def reward_afc_from_b(b, f):
  
    return float(np.asarray(b, float) @ np.asarray(f, float))


def pick_reward_hubs_by_degree(G: nx.Graph, m: int = 3):

    hubs = sorted(G.nodes(), key=lambda v: (-G.degree(v), v))[:m]
    return [int(v) for v in hubs]


def build_distance_decayed_rewards(G: nx.Graph, hubs, base_rewards, decay=0.6):

    n = G.number_of_nodes()
    f = np.zeros(n, dtype=float)

    for h, R in zip(hubs, base_rewards):
        lengths = nx.single_source_shortest_path_length(G, h)
        for v, d in lengths.items():
            val = float(R) * (float(decay) ** int(d))
            if val > f[int(v)]:
                f[int(v)] = val
    return f

def transition_reward_switch_rate(P_hat: np.ndarray, s: np.ndarray, N: np.ndarray, Q: np.ndarray):

    n = len(s)
    r = P_hat[:n, n]
    phi = (1.0 - r) - np.diag(Q)

    denom = float((s @ N).sum())
    numer = float(s @ N @ phi)
    return numer / denom


def transition_reward_improvement(P_hat: np.ndarray, s: np.ndarray, N: np.ndarray, Q: np.ndarray, f: np.ndarray):

    n = len(s)
    f = np.asarray(f, dtype=float)
    phi = np.zeros(n, dtype=float)
    for i in range(n):
        diff = np.maximum(f - f[i], 0.0)
        phi[i] = float(Q[i, :] @ diff)

    denom = float((s @ N).sum())
    numer = float(s @ N @ phi)
    return numer / denom

def plot_top5_reward_hubs_and_bars_with_arrows(
    G: nx.Graph,
    b: np.ndarray,
    hubs,
    f: np.ndarray,
    top_items,                 
    title: str,
    seed: int = 0,
    dpi: int = 600,
    savepath: Optional[str] = None,
    arrow_from: str = "top5",   
):


    pos = packed_component_layout_grid(G, seed=seed, comp_scale=7.0, pad=3.0)

    top5_nodes, _ = rank_by_value(b, top_k=5)
    top5_nodes = [v for v, _ in top5_nodes]

    hubs = list(hubs)
    set_top = set(top5_nodes)
    set_hub = set(hubs)

    both = sorted(list(set_top & set_hub))
    only_top = sorted(list(set_top - set_hub))
    only_hub = sorted(list(set_hub - set_top))

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.06)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    n_nodes = G.number_of_nodes()
    base_node_size = max(10, int(1800 / max(n_nodes, 1)))
    highlight_size = base_node_size * 18

    nx.draw_networkx_edges(G, pos, ax=ax_net, alpha=0.20, width=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=base_node_size)  # default blue

    if only_hub:
        nx.draw_networkx_nodes(
            G, pos, ax=ax_net,
            nodelist=only_hub,
            node_size=highlight_size,
            node_shape="D",
            node_color="orange",
        )

    if only_top:
        nx.draw_networkx_nodes(
            G, pos, ax=ax_net,
            nodelist=only_top,
            node_size=highlight_size,
            node_shape="s",
            node_color="red",
        )

    if both:
        nx.draw_networkx_nodes(
            G, pos, ax=ax_net,
            nodelist=both,
            node_size=highlight_size * 1.2,
            node_shape="*",
            node_color="yellow",
            edgecolors="black",
            linewidths=0.8,
        )

    union_nodes = sorted(list(set_top | set_hub))
    label_map = {}
    for v in union_nodes:
        if v in set_hub:
            label_map[v] = f"{v}\nR={f[v]:.2f}"
        else:
            label_map[v] = str(v)

    nx.draw_networkx_labels(G, pos, ax=ax_net, labels=label_map, font_size=8)

    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color="red", marker="s", linestyle="None", markersize=10, label="Top-5 by b(s)"),
        mlines.Line2D([], [], color="orange", marker="D", linestyle="None", markersize=10, label="High-reward hubs"),
        mlines.Line2D(
            [], [], color="yellow", marker="*", markeredgecolor="black",
            linestyle="None", markersize=12, label="In both"
        ),
    ]
    ax_net.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax_net.set_title(title)
    ax_net.axis("off")
    if len(pos) > 0:
        set_axis_limits_from_pos(ax_net, pos, margin=0.05)

    bar_nodes = [int(v) for v, _ in top_items]
    bar_vals = [float(val) for _, val in top_items]

    x = np.arange(len(bar_nodes), dtype=float)
    bars = ax_bar.bar(x, bar_vals)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(v) for v in bar_nodes], rotation = 0, ha="center")
    ax_bar.set_ylabel("AFC value b(s)")
    ax_bar.set_xlabel("Node id (Top-ranked)")

    y_max = max([1e-12] + bar_vals)
    ax_bar.set_ylim(0.0, y_max * 1.15)

    fig.subplots_adjust(bottom=0.22)

    node_to_bar_index = {v: i for i, v in enumerate(bar_nodes)}

    if arrow_from == "top5":
        arrow_nodes = sorted(set_top)
    elif arrow_from == "union":
        arrow_nodes = sorted(set_top | set_hub)
    else:
        raise ValueError("arrow_from must be 'top5' or 'union'")

    for v in arrow_nodes:
        if v not in node_to_bar_index:
            continue
        if v not in pos:
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
    return fig

def make_base_graph(model: str, n: int, seed: int):
    if model == "ER":
        return nx.erdos_renyi_graph(n=n, p=0.08, seed=seed)
    if model == "WS":
        return nx.watts_strogatz_graph(n=n, k=6, p=0.10, seed=seed)
    if model == "BA":
        return nx.barabasi_albert_graph(n=n, m=3, seed=seed)
    raise ValueError("Unknown model")

def run_multi_reward_demo(
    model: str,
    n=60,
    seed=42,
    p_edge_on=0.85,
    alpha_stop=0.15,
    k_min=5,
    M=40,
    reward_hubs_m=3,
    decay=0.6,
    base_rewards=(10.0, 7.0, 5.0),
    show_bar=True,
    top_bar_k=10,
    dpi=600,
    save=True,
    arrow_from="top5", 
):

    G0 = make_base_graph(model, n=n, seed=seed)

    G, isolates = drop_isolates_and_relabel(G0)
    if G.number_of_nodes() == 0:
        print(f"{model}: all nodes are isolates, skipping.")
        return

    sim = OneStepSimulator(G, p_edge_on=p_edge_on, k_min=k_min, alpha_stop=alpha_stop, seed=seed + 1)

    P_hat = estimate_amc_kernel(sim, M=M, absorb_floor=1e-6)

    n2 = sim.n
    s = np.ones(n2) / n2
    b, N, Q = afc_from_kernel(P_hat, s)

    hubs = pick_reward_hubs_by_degree(G, m=reward_hubs_m)
    base_rewards = list(base_rewards)[:reward_hubs_m]
    if len(base_rewards) < reward_hubs_m and len(base_rewards) > 0:
        base_rewards += [base_rewards[-1]] * (reward_hubs_m - len(base_rewards))
    elif reward_hubs_m > 0 and len(base_rewards) == 0:
        base_rewards = [1.0] * reward_hubs_m

    f = build_distance_decayed_rewards(G, hubs=hubs, base_rewards=base_rewards, decay=decay)

    b_f = reward_afc_from_b(b, f)
    switch_rate = transition_reward_switch_rate(P_hat, s, N, Q)
    improve_rate = transition_reward_improvement(P_hat, s, N, Q, f)

    top5, ranked_all = rank_by_value(b, top_k=5)

    k_bar = int(top_bar_k)
    if k_bar < 5:
        k_bar = 5

    topbar = ranked_all[: min(k_bar, len(ranked_all))]

    print(f"\n=== {model} | multi-rewards demo ===")
    print(
        f"nodes(after drop isolates)={G.number_of_nodes()}, edges={G.number_of_edges()}, "
        f"isolates_removed={len(isolates)}"
    )
    print(f"AMC params: p_edge_on={p_edge_on}, alpha_stop={alpha_stop}, k_min={k_min}, M={M}")
    print("Top-5 by b(s):")
    for r, (v, val) in enumerate(top5, 1):
        print(f"  {r:>2d}. node={v:<4}  b={val:.6f}")

    print(f"High-reward hubs (degree-based): {hubs}")
    for h, R in zip(hubs, base_rewards):
        print(f"  hub={h:<4} base_reward={float(R):.2f}  f(hub)={f[h]:.2f}")

    print(f"Node-reward AFC (b_f): {b_f:.6f}")
    print(f"Transition reward (switch rate) b_eta_sw: {switch_rate:.6f}  (higher = more switching)")
    print(f"Transition reward (improvement) b_eta_imp: {improve_rate:.6f} (higher = moves toward high reward)")

    if show_bar:
        savepath = f"{model}_multi_reward_network_bars_arrows_600dpi.png" if save else None
        plot_top5_reward_hubs_and_bars_with_arrows(
            G=G,
            b=b,
            hubs=hubs,
            f=f,
            top_items=topbar,
            title=f"{model} | Top-5 by b(s) + high-reward hubs",
            seed=seed,
            dpi=dpi,
            savepath=savepath,
            arrow_from=arrow_from,
        )
        if save:
            print(f"Saved: {savepath}")
    else:
        pass


if __name__ == "__main__":
    for model in ["ER", "WS", "BA"]:
        run_multi_reward_demo(
            model=model,
            n=100,
            seed=42,
            p_edge_on=0.85,
            alpha_stop=0.15,
            k_min=5,
            M=100,
            reward_hubs_m=5,
            decay=0.60,
            base_rewards=(10.0, 10.0, 10.0),
            show_bar=True,
            top_bar_k=10,
            dpi=600,
            save=True,
            arrow_from="top5", 
        )
