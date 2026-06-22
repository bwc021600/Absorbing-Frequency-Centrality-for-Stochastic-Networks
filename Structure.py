import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
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


def topk_with_tiebreak(scores_dict, k):
    """Top-k by score desc; tie-break by smaller node id."""
    ranked = sorted(scores_dict.items(), key=lambda x: (-x[1], x[0]))
    return [v for v, _ in ranked[: min(k, len(ranked))]], ranked

def drop_isolates_and_relabel(G: nx.Graph):
    isolates = list(nx.isolates(G))
    H = G.copy()
    H.remove_nodes_from(isolates)
    # relabel for array indexing
    H = nx.convert_node_labels_to_integers(H, ordering="sorted")
    return H, isolates

def all_k_cliques(G: nx.Graph, k: int):

    cliques_k = []
    for c in nx.enumerate_all_cliques(G):
        if len(c) < k:
            continue
        if len(c) == k:
            cliques_k.append(tuple(sorted(c)))
        else:
            continue
    cliques_k = sorted(list(set(cliques_k)))
    return cliques_k


def choose_k_clique_shapes(G: nx.Graph, k: int = 4, num_cliques: int = 8):

    cliques_k = all_k_cliques(G, k=k)

    if len(cliques_k) == 0:
        return [], set(), {v: [] for v in G.nodes()}, []

    def score_clique(c):
        return sum(G.degree(v) for v in c)

    scored = [(c, score_clique(c)) for c in cliques_k]
    scored.sort(key=lambda x: (-x[1], x[0]))

    chosen = scored[: min(num_cliques, len(scored))]
    cliques = [c for c, _ in chosen]
    clique_scores = [s for _, s in chosen]

    W = set()
    node_to_clique_ids = {v: [] for v in G.nodes()}
    for cid, c in enumerate(cliques):
        for v in c:
            W.add(v)
            node_to_clique_ids[v].append(cid)

    return cliques, W, node_to_clique_ids, clique_scores

class BaseOneStepSampler:

    def __init__(self, base_graph, p_edge_on, alpha_stop, seed=0):
        self.G0 = base_graph
        self.nodes = list(self.G0.nodes())
        edges = list(self.G0.edges())
        self.edges_arr = np.array(edges, dtype=int) if edges else np.zeros((0, 2), dtype=int)
        self.m = self.edges_arr.shape[0]

        self.p_edge_on = float(p_edge_on)
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

    def exogenous_stop(self):
        return (self.rng.random() < self.alpha_stop)


def local_topk_componentwise(anchor_i: int, H: nx.Graph, k: int, k_min: int):

    comp = set(nx.node_connected_component(H, anchor_i))
    if len(comp) < k_min:
        return None, None

    sub = H.subgraph(comp)
    bc = nx.betweenness_centrality(sub, normalized=False, weight=None)
    topk, ranked = topk_with_tiebreak(bc, k=k)
    return topk, ranked

class ShapeConstrainedSimulatorClique:

    def __init__(self, base_graph, p_edge_on, k_min, alpha_stop,
                 k_filter=5, cliques=None, W=None, fallback_node=None, seed=0):
        self.G0 = base_graph
        self.n = self.G0.number_of_nodes()

        self.k_min = int(k_min)
        self.k_filter = int(k_filter)

        self.sampler = BaseOneStepSampler(self.G0, p_edge_on=p_edge_on, alpha_stop=alpha_stop, seed=seed)

        self.cliques = cliques if cliques is not None else []
        self.W = set(W) if W is not None else set()

        if fallback_node is None:
            if len(self.cliques) == 0:
                raise ValueError("Need at least one clique to define a fallback node.")
            fallback_node = min(self.cliques[0])  # deterministic
        self.v_fb = int(fallback_node)

    def sample_next(self, current_i: int):
        if self.sampler.exogenous_stop():
            return None

        H = self.sampler.sample_realized_working_graph()

        topk, _ = local_topk_componentwise(current_i, H, k=self.k_filter, k_min=self.k_min)
        if topk is None:
            return None

        cand = [v for v in topk if v in self.W]
        if len(cand) > 0:
            return cand[0]
        else:
            return self.v_fb

def estimate_amc_kernel(sim, M: int, absorb_floor: float = 1e-6):
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
    return mu / mu.sum()


def rank_nodes_by_b(b, top_k=5, top_plot=20):
    ranked = sorted([(i, float(b[i])) for i in range(len(b))], key=lambda x: (-x[1], x[0]))
    return ranked[:top_k], ranked[:top_plot]

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
                iterations=1200
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


def _set_axis_limits_from_pos(ax, pos, margin=0.06):
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


def plot_top5_cliques_and_bars_with_arrows(
    G: nx.Graph,
    top5_nodes,
    top_items,                 # list[(node, bval)] used for bars
    cliques,
    node_to_clique_ids,
    clique_scores,
    title: str,
    seed: int = 0,
    dpi: int = 600,
    savepath: Optional[str] = None,
):

    pos = packed_component_layout_grid(G, seed=seed, comp_scale=7.0, pad=3.0)

    clique_ids_to_draw = set()
    for v in top5_nodes:
        ids = node_to_clique_ids.get(v, [])
        if not ids:
            continue
        best_id = max(ids, key=lambda cid: (clique_scores[cid], -cid))
        clique_ids_to_draw.add(best_id)

    clique_edges = []
    for cid in sorted(list(clique_ids_to_draw)):
        c = cliques[cid]
        for u, w in itertools.combinations(c, 2):
            if G.has_edge(u, w):
                clique_edges.append((u, w))

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.06)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    n = G.number_of_nodes()
    base_node_size = max(10, int(1800 / max(n, 1)))
    top_node_size = base_node_size * 18

    nx.draw_networkx_edges(G, pos, ax=ax_net, alpha=0.20, width=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=base_node_size)

    if clique_edges:
        nx.draw_networkx_edges(G, pos, ax=ax_net, edgelist=clique_edges, width=2.6, edge_color="green", alpha=0.9)

    nx.draw_networkx_nodes(
        G, pos, ax=ax_net,
        nodelist=top5_nodes,
        node_size=top_node_size,
        node_shape="s",
        node_color="red",
    )
    nx.draw_networkx_labels(G, pos, ax=ax_net, labels={v: str(v) for v in top5_nodes}, font_size=9)

    ax_net.set_title(title)
    ax_net.axis("off")
    if len(pos) > 0:
        _set_axis_limits_from_pos(ax_net, pos, margin=0.05)

    bar_nodes = [int(v) for v, _ in top_items]
    bar_vals = [float(val) for _, val in top_items]

    x = np.arange(len(bar_nodes), dtype=float)
    bars = ax_bar.bar(x, bar_vals)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(v) for v in bar_nodes], rotation=0, ha="center")  # vertical labels
    ax_bar.set_ylabel("AFC value b(s)")
    ax_bar.set_xlabel("Node id (Top-ranked)")

    y_max = max([1e-12] + bar_vals)
    ax_bar.set_ylim(0.0, y_max * 1.15)

    # extra room for vertical tick labels
    fig.subplots_adjust(bottom=0.22)

    node_to_bar_index = {v: i for i, v in enumerate(bar_nodes)}
    for v in top5_nodes:
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
        # NOTE: p=0.08 often yields zero 4-cliques when n=60.
        return nx.erdos_renyi_graph(n=n, p=0.08, seed=seed)
    if model == "WS":
        return nx.watts_strogatz_graph(n=n, k=6, p=0.10, seed=seed)
    if model == "BA":
        return nx.barabasi_albert_graph(n=n, m=3, seed=seed)
    raise ValueError("Unknown model")


def run_kclique_structure_demo(
    model: str,
    n=60,
    seed=42,

    p_edge_on=0.85,
    alpha_stop=0.15,
    k_min=5,

    k_filter=5,

    clique_k=4,
    num_cliques=8,

    M=40,
    show_bar=True,
    top_bar_k=10,      
    dpi=600,
    save=True,
):
    G0 = make_base_graph(model, n=n, seed=seed)

    G, isolates = drop_isolates_and_relabel(G0)
    print(f"\n=== {model} (k-clique constrained, k={clique_k}) ===")
    print(f"original n={G0.number_of_nodes()}, isolates removed={len(isolates)}, final n={G.number_of_nodes()}, edges={G.number_of_edges()}")

    cliques, W, node_to_clique_ids, clique_scores = choose_k_clique_shapes(G, k=clique_k, num_cliques=num_cliques)

    if len(cliques) == 0:
        print(f"No {clique_k}-cliques found on this base graph.")
        print("Try: increase ER p (e.g., 0.12~0.20), increase n, increase BA m, or set k=3.")
        return

    print(f"selected primitive {clique_k}-cliques = {len(cliques)}, target pool |W| = {len(W)}")
    v_fb = min(cliques[0])
    print(f"fallback node v_fb = {v_fb}")

    sim = ShapeConstrainedSimulatorClique(
        base_graph=G,
        p_edge_on=p_edge_on,
        k_min=k_min,
        alpha_stop=alpha_stop,
        k_filter=k_filter,
        cliques=cliques,
        W=W,
        fallback_node=v_fb,
        seed=seed + 1
    )

    P_hat = estimate_amc_kernel(sim, M=M, absorb_floor=1e-6)
    n2 = sim.n
    s = np.ones(n2) / n2
    b = afc_from_kernel(P_hat, s)

    top5, top_plot_items = rank_nodes_by_b(b, top_k=5, top_plot=max(int(top_bar_k), 5))
    top5_nodes = [v for v, _ in top5]

    print("Top-5 nodes by clique-constrained AFC b(s):")
    for r, (v, val) in enumerate(top5, 1):
        print(f"  {r:>2d}. node={v:<4}  b={val:.6f}")

    if show_bar:
        savepath = f"{model}_kclique_constraint_{clique_k}_combined_600dpi.png" if save else None
        plot_top5_cliques_and_bars_with_arrows(
            G=G,
            top5_nodes=top5_nodes,
            top_items=top_plot_items,
            cliques=cliques,
            node_to_clique_ids=node_to_clique_ids,
            clique_scores=clique_scores,
            title=f"{model} | Top-5 under {clique_k}-clique constraint (red) + clique edges (green)",
            seed=seed,
            dpi=dpi,
            savepath=savepath,
        )
        if save:
            print(f"Saved: {savepath}")


if __name__ == "__main__":
    for model in ["ER", "WS", "BA"]:
        run_kclique_structure_demo(
            model=model,
            n=100,
            seed=42,
            p_edge_on=0.85,
            alpha_stop=0.15,
            k_min=5,
            k_filter=5,
            clique_k=3,
            num_cliques=8,
            M=100,
            show_bar=True,
            top_bar_k=10,
            dpi=600,
            save=True,
        )
