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
                # assign random positive edge weights (travel times)
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
            transient_sum = row[:n].sum()
            if transient_sum > 0:
                row[:n] *= (1.0 - absorb_floor) / transient_sum
            else:
                row[ABS] = 1.0

        P_hat[i, :] = row

    P_hat[ABS, ABS] = 1.0
    return P_hat


def afc_from_kernel(P_hat: np.ndarray, s: np.ndarray):

    n = len(s)
    Q = P_hat[:n, :n]
    A = np.eye(n) - Q  # A = I - Q

    mu = np.linalg.solve(A.T, s)
    b = mu / mu.sum()
    return b


def packed_component_layout(
    G: nx.Graph, seed: int = 0, comp_scale: float = 1.0, pad: float = 2.0
):

    rng = np.random.default_rng(seed)

    comps = [list(c) for c in nx.connected_components(G)]
    comps.sort(key=len, reverse=True)

    comp_layouts = []
    for nodes in comps:
        H = G.subgraph(nodes)

        if len(nodes) == 1:
            pos_c = {nodes[0]: np.array([0.0, 0.0])}
        else:
            pos_raw = nx.spring_layout(
                H,
                seed=int(rng.integers(1_000_000_000)),
                k=1.0 / np.sqrt(len(nodes)),
                iterations=300,
            )
            coords = np.array(list(pos_raw.values()))
            coords = coords - coords.mean(axis=0)
            span = float(np.max(np.ptp(coords, axis=0)))
            if span > 0:
                coords = coords / span * comp_scale
            pos_c = {node: coords[i] for i, node in enumerate(pos_raw.keys())}

        coords_c = np.array(list(pos_c.values()))
        w = float(np.ptp(coords_c[:, 0])) if len(nodes) > 1 else 0.0
        h = float(np.ptp(coords_c[:, 1])) if len(nodes) > 1 else 0.0
        comp_layouts.append((nodes, pos_c, w, h))

    pos = {}
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0
    max_row_width = 10.0 

    for nodes, pos_c, w, h in comp_layouts:
        w_eff = max(w, 0.4)
        h_eff = max(h, 0.4)

        if x_cursor > 0 and (x_cursor + w_eff) > max_row_width:
            x_cursor = 0.0
            y_cursor -= (row_height + pad)
            row_height = 0.0

        for node, xy in pos_c.items():
            pos[node] = (float(xy[0] + x_cursor), float(xy[1] + y_cursor))

        x_cursor += (w_eff + pad)
        row_height = max(row_height, h_eff)

    all_xy = np.array(list(pos.values()))
    center = all_xy.mean(axis=0)
    for node, (x, y) in pos.items():
        pos[node] = (x - center[0], y - center[1])

    return pos


def plot_graph_and_bars_with_arrows(
    G_base: nx.Graph,
    top_nodes,          
    top_items,       
    title: str,
    layout_seed: int = 0,
    dpi: int = 600,
    savepath: Optional[str] = None,
):

    pos = packed_component_layout(G_base, seed=layout_seed, comp_scale=1.0, pad=2.0)

    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.05)

    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    n = G_base.number_of_nodes()
    base_node_size = max(8, int(1400 / max(n, 1)))
    top_node_size = base_node_size * 12

    nx.draw_networkx_edges(G_base, pos, ax=ax_net, alpha=0.20, width=0.7)
    nx.draw_networkx_nodes(G_base, pos, ax=ax_net, node_size=base_node_size)

    nx.draw_networkx_nodes(
        G_base, pos, ax=ax_net,
        nodelist=list(top_nodes),
        node_size=top_node_size,
        node_shape="s",
        node_color="red",
    )
    nx.draw_networkx_labels(
        G_base, pos, ax=ax_net,
        labels={v: str(v) for v in top_nodes},
        font_size=9,
    )

    ax_net.set_title(title)
    ax_net.axis("off")

    bar_nodes = [v for v, _ in top_items]
    bar_vals  = [float(val) for _, val in top_items]

    x = np.arange(len(bar_nodes))
    bars = ax_bar.bar(x, bar_vals)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(v) for v in bar_nodes], rotation=0, ha="center")
    ax_bar.set_ylabel("AFC value b(s)")
    ax_bar.set_xlabel("Node id (Top-ranked by b)")
    ax_bar.margins(x=0.01)

    y_max = max(bar_vals) if len(bar_vals) else 1.0
    ax_bar.set_ylim(0.0, y_max * 1.15)

    fig.subplots_adjust(bottom=0.22)

    node_to_bar_index = {v: i for i, v in enumerate(bar_nodes)}

    for v in top_nodes:
        if v not in node_to_bar_index:
            continue 

        i = node_to_bar_index[v]
        bar = bars[i]

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


def rank_nodes_by_b(b: np.ndarray, top_k=5, top_plot=10):
    ranked = sorted([(i, float(b[i])) for i in range(len(b))], key=lambda x: (-x[1], x[0]))
    return ranked[:top_k], ranked[:top_plot]


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
    n: int = 100,
    seed: int = 42,
    p_edge_on: float = 0.85,
    k_min: int = 5,
    alpha_stop: float = 0.15,
    M: int = 60,
    use_weights: bool = False,
    top_plot: int = 10,
    dpi: int = 600,
    save: bool = True,
):

    G_base = make_base_graph(model, n=n, seed=seed)

    print(f"\n=== {model} ===")
    print(
        f"Base graph: nodes={G_base.number_of_nodes()}, edges={G_base.number_of_edges()}, "
        f"components={nx.number_connected_components(G_base)}, isolates={len(list(nx.isolates(G_base)))}"
    )

    sim = OneStepSimulator(
        base_graph=G_base,
        p_edge_on=p_edge_on,
        k_min=k_min,
        alpha_stop=alpha_stop,
        seed=seed + 1,
        use_weights=use_weights,
    )

    P_hat = estimate_amc_kernel(sim, M=M, absorb_floor=1e-6)

    s = np.ones(n) / n
    b = afc_from_kernel(P_hat, s)

    top5, top_bars = rank_nodes_by_b(b, top_k=5, top_plot=top_plot)

    print(
        f"Params: p_edge_on={p_edge_on}, k_min={k_min}, alpha_stop={alpha_stop}, "
        f"M={M}, use_weights={use_weights}"
    )
    print("Top 5 by AMC b(s):")
    for r, (v, val) in enumerate(top5, 1):
        print(f"  {r:>2d}. node={v:<4}  b={val:.6f}")

    savepath = f"{model}_amc_topnodes_600dpi.png" if save else None

    plot_graph_and_bars_with_arrows(
        G_base=G_base,
        top_nodes=[v for v, _ in top5],
        top_items=top_bars,
        title=f"{model} | Top 5 by AMC b(s)",
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
            use_weights=False,
            top_plot=10,   
            dpi=600,      
            save=True,    
        )
