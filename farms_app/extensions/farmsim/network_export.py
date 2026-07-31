"""Export network topology to TikZ (.tex) or matplotlib figure."""

import networkx as nx
from farms_core import pylog


def _network_to_nx(network):
    """Convert a farms_network Network to a networkx DiGraph."""
    G = nx.DiGraph()
    for node in network.options.nodes:
        pos = node.visual['position'][:2]
        G.add_node(
            node.name,
            x=pos[0],
            y=pos[1],
            label=node.visual.get('label', node.name),
            color=node.visual.get('color', [0.5, 0.5, 0.5]),
        )
    for edge in network.options.edges:
        G.add_edge(
            edge.source, edge.target,
            weight=edge.weight,
            type=edge.type,
        )
    return G


def _mute_color(c, saturation=0.7, value=0.85):
    """Reduce saturation and cap brightness for print-friendly colors.

    Works in HSV space: scales saturation down and clamps value,
    so colors stay recognizable but aren't neon.
    """
    import colorsys
    r, g, b = max(0.0, min(1.0, c[0])), max(0.0, min(1.0, c[1])), max(0.0, min(1.0, c[2]))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s *= saturation
    v = min(v, value)
    return list(colorsys.hsv_to_rgb(h, s, v))


def _rgb_to_tikz(c):
    """Convert [r,g,b] floats to TikZ rgb color string."""
    r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
    return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


# Relay models: invisible zero-size pass-through nodes
_RELAY_MODELS = {"relay"}


# --- TikZ export ---

_TIKZ_PREAMBLE = r"""\documentclass[tikz]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, backgrounds, calc, positioning, shadows, shadings}

% Edge colors (muted for print)
\definecolor{col-excitatory}{RGB}{60,140,90}
\definecolor{col-inhibitory}{RGB}{140,70,100}
\definecolor{col-generic}{RGB}{120,120,120}

\tikzset{
  % Base neuron — flat fill with subtle drop shadow
  neuron/.style={
    circle, double, draw=black!50, semithick,
    minimum size=0.9cm, inner sep=0.5pt, outer sep=1pt,
    font=\footnotesize\bfseries, text=white,
    drop shadow={shadow xshift=0.4pt, shadow yshift=-0.4pt, opacity=0.25},
  },
  % Relay: invisible node body, readable label
  relay/.style={
    circle, inner sep=1pt, outer sep=2pt,
    minimum size=0pt, draw=none, fill=none,
    font=\footnotesize, text=black,
  },
  % Edges
  edge-base/.style={thick, on background layer},
  excitatory-edge/.style={edge-base, col-excitatory, -{Latex[scale=1.0]}},
  inhibitory-edge/.style={edge-base, col-inhibitory, -{Circle[scale=0.8, fill]}},
  generic-edge/.style={edge-base, col-generic, -{Latex[scale=0.9]}},
}
"""

_EDGE_STYLE_MAP = {
    "excitatory": "excitatory-edge",
    "inhibitory": "inhibitory-edge",
    "generic": "generic-edge",
    "cholinergic": "excitatory-edge",
    "phase_coupling": "generic-edge",
}


def export_tikz(network, path, neuron_shading="ball"):
    """Export network topology to a standalone TikZ .tex file.

    neuron_shading: "ball" for 3-D sphere shading, "flat" for solid fill.
    """
    G = _network_to_nx(network)

    pos = {
        node.name: (node.visual['position'][0], node.visual['position'][1])
        for node in network.options.nodes
    }

    node_label = {}
    node_options = {}
    for node in network.options.nodes:
        label = node.visual.get('label', node.name)
        is_relay = node.model in _RELAY_MODELS
        if is_relay:
            node_options[node.name] = "relay"
        else:
            c = _mute_color(node.visual.get('color', [0.5, 0.5, 0.5]), saturation=0.85, value=0.9)
            col = _rgb_to_tikz(c)
            if neuron_shading == "ball":
                node_options[node.name] = f"neuron, shading=ball, ball color={col}"
            else:
                node_options[node.name] = f"neuron, fill={col}"
        node_label[node.name] = label

    edge_options = {}
    for edge in network.options.edges:
        style = _EDGE_STYLE_MAP.get(edge.type, "generic-edge")
        edge_options[(edge.source, edge.target)] = style

    raw = nx.to_latex_raw(
        G,
        pos=pos,
        node_options=node_options,
        node_label=node_label,
        edge_options=edge_options,
        default_edge_options="[generic-edge]",
    )

    # Strip the outer tikzpicture that to_latex_raw wraps
    lines = raw.strip().split('\n')
    if lines and lines[0].strip().startswith('\\begin{tikzpicture}'):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith('\\end{tikzpicture}'):
        lines = lines[:-1]
    body = '\n'.join(lines)

    tex = (
        _TIKZ_PREAMBLE
        + "\\begin{document}\n"
        + "\\begin{tikzpicture}\n"
        + body + "\n"
        + "\\end{tikzpicture}\n"
        + "\\end{document}\n"
    )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    pylog.info("Exported TikZ network to %s", path)


# --- Matplotlib export ---

def export_matplotlib(network, path):
    """Export network topology to an image file via matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    G = _network_to_nx(network)
    nodes = network.options.nodes
    edges = network.options.edges

    pos = {
        node.name: (node.visual['position'][0], node.visual['position'][1])
        for node in nodes
    }

    # Compute figure size from bounding box to preserve aspect ratio
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    max_dim = 12.0
    if x_range >= y_range:
        figw, figh = max_dim, max_dim * y_range / x_range
    else:
        figw, figh = max_dim * x_range / y_range, max_dim
    figw = max(figw, 6.0)
    figh = max(figh, 6.0)

    visible_nodes = [n for n in nodes if n.model not in _RELAY_MODELS]
    relay_nodes = [n for n in nodes if n.model in _RELAY_MODELS]

    fig, ax = plt.subplots(1, 1, figsize=(figw, figh))

    # Draw edges first (behind nodes)
    exc_edges = [(e.source, e.target) for e in edges
                 if e.type in ('excitatory', 'cholinergic')]
    inh_edges = [(e.source, e.target) for e in edges
                 if e.type == 'inhibitory']
    other_edges = [(e.source, e.target) for e in edges
                   if e.type not in ('excitatory', 'cholinergic', 'inhibitory')]

    edge_kw = dict(
        arrows=True,
        min_source_margin=12,
        min_target_margin=12,
        connectionstyle="arc3,rad=0.08",
    )
    if exc_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=exc_edges,
            edge_color=[0.24, 0.55, 0.35], width=1.5, arrowsize=12,
            arrowstyle='-|>', **edge_kw,
        )
    if inh_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=inh_edges,
            edge_color=[0.55, 0.27, 0.39], width=1.5, arrowsize=12,
            arrowstyle='-|>', **edge_kw,
        )
    if other_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=other_edges,
            edge_color=[0.47, 0.47, 0.47], width=1.2, arrowsize=10,
            arrowstyle='-|>', **edge_kw,
        )

    # Draw visible nodes
    if visible_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[n.name for n in visible_nodes],
            node_color=[
                n.visual.get('color', [0.5, 0.5, 0.5])
                for n in visible_nodes
            ],
            alpha=0.9,
            edgecolors='black',
            linewidths=2.0,
            node_size=[500 * n.visual.get('radius', 1.0) for n in visible_nodes],
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={n.name: n.visual.get('label', n.name) for n in visible_nodes},
            font_size=9,
            font_weight='bold',
            font_family='sans-serif',
            font_color='white',
        )

    # Relay nodes: invisible body, visible label
    if relay_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[n.name for n in relay_nodes],
            node_color='none',
            edgecolors='none',
            node_size=1,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={n.name: n.visual.get('label', n.name) for n in relay_nodes},
            font_size=9,
            font_weight='bold',
            font_family='sans-serif',
            font_color='black',
        )

    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    pylog.info("Exported matplotlib network to %s", path)
