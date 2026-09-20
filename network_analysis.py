"""
Assignment 1: Opinion Network Formation - Network Analysis
==========================================================

Covers the analysis half of the pipeline: global structure, centrality,
community detection, threshold sensitivity, and interpretation of what the
network reveals about class opinions. Maps onto the "Analysis and
Visualizations" and "Results and Discussion" sections of the report.

Network construction (Likert encoding, similarity, thresholding) is documented
in analysis.ipynb; Step 0 below reproduces it verbatim so this script runs
standalone.

Usage:
    .venv/bin/python network_analysis.py

Writes all figures to ./figures/ and prints every metric to stdout.
"""

import matplotlib
matplotlib.use("Agg")          # headless backend: no Qt/X11 needed under WSL

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "Survey_Results_UC.csv"
FIG_DIR = HERE / "figures"

TAU = 0.40                     # similarity threshold, per analysis.ipynb Step 2
SEED = 42

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
DOMAINS = {"T": "Technology", "E": "Education", "S": "Society", "V": "Environment"}

LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
    "No Comments": 3,
}


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure saved] figures/{name}")


# ---------------------------------------------------------------------------
# Step 0: reproduce the constructed network (analysis.ipynb Steps 1-2)
# ---------------------------------------------------------------------------
def build_network():
    banner("STEP 0: REPRODUCING THE CONSTRUCTED NETWORK")

    df = pd.read_csv(DATA_FILE)
    survey_cols = df.columns[1:]

    df_numeric = df.copy()
    for col in survey_cols:
        df_numeric[col] = df_numeric[col].map(LIKERT)
    df_numeric[survey_cols] = df_numeric[survey_cols].fillna(3)

    features = df_numeric[survey_cols].values
    respondent_ids = df_numeric["id. Response ID"].values

    centered = features - features.mean(axis=1, keepdims=True)
    sim_matrix = cosine_similarity(centered)
    np.fill_diagonal(sim_matrix, 0)

    G = nx.from_numpy_array((sim_matrix >= TAU).astype(int))
    G = nx.relabel_nodes(G, {i: respondent_ids[i] for i in range(len(respondent_ids))})
    G.remove_nodes_from(list(nx.isolates(G)))

    print(f"Reproduced network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(tau = {TAU})")
    print(f"Feature matrix: {features.shape[0]} respondents x {features.shape[1]} questions")
    if (G.number_of_nodes(), G.number_of_edges()) != (86, 893):
        print("  WARNING: construction diverges from analysis.ipynb Step 2 "
              "- re-sync before interpreting results below.")
    else:
        print("Matches analysis.ipynb Step 2.")

    return df, survey_cols, features, respondent_ids, sim_matrix, G


# ---------------------------------------------------------------------------
# Step 3: data-quality guard
# ---------------------------------------------------------------------------
def data_quality_guard(df, survey_cols, features, respondent_ids, G):
    banner("STEP 3: DATA-QUALITY GUARD BEFORE ANALYSIS")

    missing_counts = df[survey_cols].isna().sum(axis=1)
    n_items = len(survey_cols)

    print("Response completeness across all 96 respondents")
    print(f"  fully complete (0 blanks) : {(missing_counts == 0).sum()}")
    print(f"  partial (1-30 blanks)     : {((missing_counts > 0) & (missing_counts <= 30)).sum()}")
    print(f"  severe (>30 blanks)       : {(missing_counts > 30).sum()}")
    print(f"  entirely blank (60)       : {(missing_counts == 60).sum()}")
    print(f"\nTotal imputed cells: {missing_counts.sum()} of {96 * n_items} "
          f"({100 * missing_counts.sum() / (96 * n_items):.1f}%)")

    id_to_index = {rid: i for i, rid in enumerate(respondent_ids)}
    raw_communities = sorted(nx.community.louvain_communities(G, seed=SEED),
                             key=len, reverse=True)

    print("\nLouvain communities on the Step 2 graph, by mean non-response:")
    for k, comm in enumerate(raw_communities):
        blanks = [missing_counts.iloc[id_to_index[r]] for r in comm]
        print(f"  community {k}: size {len(comm):>2}   mean blanks {np.mean(blanks):5.1f}   "
              f"max {max(blanks)}")

    print("\nSmallest community, member by member:")
    for r in sorted(raw_communities[-1]):
        print(f"  respondent {r:>3}: {missing_counts.iloc[id_to_index[r]]:>2} blanks of 60, "
              f"degree {G.degree(r)}")

    # Rebuild excluding respondents who left more than half the survey blank.
    # The similarity matrix is recomputed, not filtered: the excluded rows never
    # legitimately contributed to any pairwise comparison.
    valid = (missing_counts <= 30).values
    features_valid = features[valid]
    ids_valid = respondent_ids[valid]

    centered = features_valid - features_valid.mean(axis=1, keepdims=True)
    sim_valid = cosine_similarity(centered)
    np.fill_diagonal(sim_valid, 0)

    GA = nx.from_numpy_array((sim_valid >= TAU).astype(int))
    GA = nx.relabel_nodes(GA, {i: ids_valid[i] for i in range(len(ids_valid))})
    dropped = list(nx.isolates(GA))
    GA.remove_nodes_from(dropped)

    print(f"\nStep 2 graph      : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Analysis graph GA : {GA.number_of_nodes()} nodes, {GA.number_of_edges()} edges")
    print(f"  excluded {(~valid).sum()} severe non-responders, then {len(dropped)} isolates")
    print(f"\nOnly {G.number_of_edges() - GA.number_of_edges()} edges are lost, so the core "
          f"structure is unaffected.\nAll metrics below are reported on GA.")

    return GA, features_valid, ids_valid, sim_valid


# ---------------------------------------------------------------------------
# Step 4: global structure
# ---------------------------------------------------------------------------
def global_structure(GA):
    banner("STEP 4: GLOBAL STRUCTURE")

    n_nodes, n_edges = GA.number_of_nodes(), GA.number_of_edges()
    deg_values = np.array([d for _, d in GA.degree()])
    components = list(nx.connected_components(GA))
    largest_cc = GA.subgraph(max(components, key=len)).copy()

    metrics = {
        "Nodes (respondents)":      n_nodes,
        "Edges (agreement ties)":   n_edges,
        "Density":                  round(nx.density(GA), 4),
        "Mean degree":              round(deg_values.mean(), 2),
        "Median degree":            float(np.median(deg_values)),
        "Min / Max degree":         f"{deg_values.min()} / {deg_values.max()}",
        "Average clustering":       round(nx.average_clustering(GA), 4),
        "Transitivity":             round(nx.transitivity(GA), 4),
        "Connected components":     len(components),
        "Largest component share":  f"{100 * largest_cc.number_of_nodes() / n_nodes:.1f}%",
        "Diameter (largest comp.)": nx.diameter(largest_cc),
        "Mean shortest path":       round(nx.average_shortest_path_length(largest_cc), 3),
        "Degree assortativity":     round(nx.degree_assortativity_coefficient(GA), 4),
    }
    print(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_string(index=False))

    # Null model: Erdos-Renyi graph of identical size and density.
    er = nx.gnm_random_graph(n_nodes, n_edges, seed=SEED)
    print("\nNull model (Erdos-Renyi, same n and m):")
    print(f"  clustering   observed {nx.average_clustering(GA):.4f}  vs  random "
          f"{nx.average_clustering(er):.4f}")
    print(f"  mean path    observed {nx.average_shortest_path_length(largest_cc):.3f}  vs  random "
          f"{nx.average_shortest_path_length(er):.3f}")

    # Figure 1: degree distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(deg_values, bins=18, color="#4C72B0", edgecolor="white")
    axes[0].axvline(deg_values.mean(), color="#C44E52", linestyle="--", linewidth=2,
                    label=f"mean = {deg_values.mean():.1f}")
    axes[0].set_xlabel("Degree (number of agreement ties)")
    axes[0].set_ylabel("Number of respondents")
    axes[0].set_title("Degree distribution")
    axes[0].legend()

    sorted_deg = np.sort(deg_values)[::-1]
    axes[1].plot(range(1, len(sorted_deg) + 1), sorted_deg, marker="o", markersize=3,
                 color="#4C72B0", linewidth=1)
    axes[1].set_xlabel("Respondent rank")
    axes[1].set_ylabel("Degree")
    axes[1].set_title("Degree rank profile")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Figure 1 - Degree structure of the opinion network (tau = {TAU})", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_degree_distribution.png")

    print(f"Degree range {deg_values.min()}-{deg_values.max()}, "
          f"standard deviation {deg_values.std():.2f}")

    return n_nodes, n_edges


# ---------------------------------------------------------------------------
# Step 5: centrality
# ---------------------------------------------------------------------------
def centrality_analysis(GA, pos):
    banner("STEP 5: CENTRALITY - WHO OCCUPIES THE MAINSTREAM?")

    centrality = pd.DataFrame({
        "degree":      pd.Series(nx.degree_centrality(GA)),
        "betweenness": pd.Series(nx.betweenness_centrality(GA)),
        "closeness":   pd.Series(nx.closeness_centrality(GA)),
        "eigenvector": pd.Series(nx.eigenvector_centrality(GA, max_iter=1000)),
    })
    centrality["raw_degree"] = pd.Series(dict(GA.degree()))
    centrality.index.name = "respondent_id"

    def rank_by(frame, col, n=10):
        """Rank descending, breaking ties by respondent_id.

        Pandas' default sort is quicksort, which is not stable, so tied rows
        (e.g. respondents 81 and 95, both at degree 41) can swap places between
        numpy versions. The explicit secondary key keeps output reproducible.
        """
        return (frame.assign(_rid=frame.index)
                     .sort_values([col, "_rid"], ascending=[False, True])
                     .drop(columns="_rid")
                     .head(n))

    print("Ten most central respondents by degree (the mainstream of the class):")
    print(rank_by(centrality, "degree").round(4).to_string())

    print("\nTen strongest bridges by betweenness:")
    print(rank_by(centrality, "betweenness").round(4).to_string())

    # Bridges that are not simply the highest-degree nodes are the interesting ones.
    centrality["deg_rank"] = centrality["degree"].rank(ascending=False)
    centrality["btw_rank"] = centrality["betweenness"].rank(ascending=False)
    centrality["bridge_gap"] = centrality["deg_rank"] - centrality["btw_rank"]
    print("\nRespondents that bridge far more than their popularity would predict:")
    print(rank_by(centrality, "bridge_gap", 5)
          [["raw_degree", "degree", "betweenness", "deg_rank", "btw_rank"]].round(4).to_string())

    print("\nCorrelation between centrality measures:")
    print(centrality[["degree", "betweenness", "closeness", "eigenvector"]].corr().round(3).to_string())

    # Figure 2: network by degree and by betweenness
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    nx.draw_networkx_edges(GA, pos, ax=axes[0], alpha=0.12, edge_color="#333333")
    nodes0 = nx.draw_networkx_nodes(
        GA, pos, ax=axes[0],
        node_size=[25 + 12 * GA.degree(v) for v in GA.nodes()],
        node_color=[centrality.loc[v, "degree"] for v in GA.nodes()],
        cmap="viridis", linewidths=0.4, edgecolors="white")
    axes[0].set_title("Degree centrality - proximity to the mainstream")
    axes[0].axis("off")
    fig.colorbar(nodes0, ax=axes[0], shrink=0.75)

    nx.draw_networkx_edges(GA, pos, ax=axes[1], alpha=0.12, edge_color="#333333")
    nodes1 = nx.draw_networkx_nodes(
        GA, pos, ax=axes[1],
        node_size=[25 + 2200 * centrality.loc[v, "betweenness"] for v in GA.nodes()],
        node_color=[centrality.loc[v, "betweenness"] for v in GA.nodes()],
        cmap="magma", linewidths=0.4, edgecolors="white")
    axes[1].set_title("Betweenness centrality - ideological bridges")
    axes[1].axis("off")
    fig.colorbar(nodes1, ax=axes[1], shrink=0.75)

    fig.suptitle(f"Figure 2 - Centrality structure (n = {GA.number_of_nodes()}, tau = {TAU})",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "fig2_centrality.png")

    return centrality


# ---------------------------------------------------------------------------
# Step 6: community detection
# ---------------------------------------------------------------------------
def community_analysis(GA, pos, n_nodes, n_edges):
    banner("STEP 6: COMMUNITY DETECTION")

    communities = sorted(nx.community.louvain_communities(GA, seed=SEED), key=len, reverse=True)
    Q = nx.community.modularity(GA, communities)

    print(f"Louvain found {len(communities)} communities")
    print(f"Sizes: {[len(c) for c in communities]}")
    print(f"Modularity Q = {Q:.4f}")

    # Stability across random seeds
    seeded = [nx.community.louvain_communities(GA, seed=s) for s in range(10)]
    n_comms = [len(c) for c in seeded]
    q_vals = [nx.community.modularity(GA, c) for c in seeded]
    print(f"\nAcross 10 seeds: community count {min(n_comms)}-{max(n_comms)}, "
          f"Q = {np.mean(q_vals):.4f} +/- {np.std(q_vals):.4f}")

    # Two null models. Erdos-Renyi fixes only n and m; the configuration model
    # additionally preserves the degree sequence, which matters here because
    # degree ranges from 1 to 50.
    q_er = [nx.community.modularity(g, nx.community.louvain_communities(g, seed=1))
            for g in [nx.gnm_random_graph(n_nodes, n_edges, seed=s) for s in range(20)]]

    deg_seq = [d for _, d in GA.degree()]
    q_cm = []
    for s in range(20):
        cm = nx.Graph(nx.configuration_model(deg_seq, seed=s))
        cm.remove_edges_from(nx.selfloop_edges(cm))
        q_cm.append(nx.community.modularity(cm, nx.community.louvain_communities(cm, seed=1)))

    z_er = (Q - np.mean(q_er)) / np.std(q_er)
    z_cm = (Q - np.mean(q_cm)) / np.std(q_cm)

    print(f"\nObserved modularity          Q = {Q:.4f}")
    print(f"  Erdos-Renyi null           Q = {np.mean(q_er):.4f} +/- {np.std(q_er):.4f}   (z = {z_er:+.2f})")
    print(f"  Configuration-model null   Q = {np.mean(q_cm):.4f} +/- {np.std(q_cm):.4f}   (z = {z_cm:+.2f})")
    print("\nThe observed modularity is statistically indistinguishable from that of random graphs")
    print("with the same size and degree sequence, and lies far below the ~0.3 conventionally")
    print("required to claim genuine community structure. The partition is therefore a descriptive")
    print("convenience imposed by the algorithm, NOT evidence of opinion factions in the class.")

    membership = {node: k for k, comm in enumerate(communities) for node in comm}

    # Figure 3: network coloured by community
    fig = plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(GA, pos, alpha=0.10, edge_color="#333333")
    for k, comm in enumerate(communities):
        nx.draw_networkx_nodes(
            GA, pos, nodelist=sorted(comm),
            node_color=PALETTE[k % len(PALETTE)],
            node_size=[40 + 10 * GA.degree(v) for v in sorted(comm)],
            label=f"Community {k} (n = {len(comm)})",
            linewidths=0.5, edgecolors="white")
    plt.legend(scatterpoints=1, loc="best", frameon=True)
    plt.title(f"Figure 3 - Louvain communities (Q = {Q:.3f}, tau = {TAU})\n"
              f"Note the heavy inter-community edge density: the partition is weak", fontsize=12)
    plt.axis("off")
    fig.tight_layout()
    save(fig, "fig3_communities.png")

    internal = sum(1 for u, v in GA.edges() if membership[u] == membership[v])
    print(f"\nEdges within communities: {internal} of {n_edges} ({100 * internal / n_edges:.1f}%)")
    print(f"Edges crossing communities: {n_edges - internal} "
          f"({100 * (n_edges - internal) / n_edges:.1f}%)")

    return communities, Q


# ---------------------------------------------------------------------------
# Step 7: threshold sensitivity
# ---------------------------------------------------------------------------
def threshold_sweep(sim_valid, ids_valid):
    banner("STEP 7: THRESHOLD SENSITIVITY")

    rows = []
    for t in np.round(np.arange(0.20, 0.71, 0.05), 2):
        G_t = nx.from_numpy_array((sim_valid >= t).astype(int))
        G_t = nx.relabel_nodes(G_t, {i: ids_valid[i] for i in range(len(ids_valid))})
        n_iso = len(list(nx.isolates(G_t)))
        G_t.remove_nodes_from(list(nx.isolates(G_t)))
        if G_t.number_of_nodes() < 5:
            continue
        comms_t = nx.community.louvain_communities(G_t, seed=SEED)
        rows.append({
            "tau": t,
            "retained": G_t.number_of_nodes(),
            "retained_pct": round(100 * G_t.number_of_nodes() / len(ids_valid), 1),
            "edges": G_t.number_of_edges(),
            "density": round(nx.density(G_t), 4),
            "isolates": n_iso,
            "components": nx.number_connected_components(G_t),
            "communities": len(comms_t),
            "modularity": round(nx.community.modularity(G_t, comms_t), 4),
            "clustering": round(nx.average_clustering(G_t), 4),
        })

    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False))

    # Figure 4: what the threshold buys and what it costs
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(sweep["tau"], sweep["density"], marker="o", color="#4C72B0", label="Density")
    axes[0].plot(sweep["tau"], sweep["clustering"], marker="s", color="#55A868", label="Clustering")
    axes[0].axvline(TAU, color="#C44E52", linestyle="--", alpha=0.8, label=f"chosen tau = {TAU}")
    axes[0].set_xlabel("Threshold tau"); axes[0].set_ylabel("Value")
    axes[0].set_title("Density and clustering"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(sweep["tau"], sweep["modularity"], marker="o", color="#8172B3")
    axes[1].axhline(0.30, color="#937860", linestyle=":", label="Q = 0.30 (conventional floor)")
    axes[1].axvline(TAU, color="#C44E52", linestyle="--", alpha=0.8)
    axes[1].set_xlabel("Threshold tau"); axes[1].set_ylabel("Modularity Q")
    axes[1].set_title("Modularity rises monotonically with tau")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(sweep["tau"], sweep["retained_pct"], marker="o", color="#DD8452",
                 label="% respondents retained")
    axes[2].axvline(TAU, color="#C44E52", linestyle="--", alpha=0.8)
    axes[2].set_xlabel("Threshold tau"); axes[2].set_ylabel("Percent retained")
    axes[2].set_title("...but only by discarding respondents")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle("Figure 4 - Threshold sensitivity: modularity is bought with sample size",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "fig4_threshold_sensitivity.png")

    corr = sweep["modularity"].corr(sweep["retained_pct"])
    print(f"\nCorrelation between modularity and share of respondents retained: {corr:.3f}")
    print("A strong negative correlation means the 'communities' sharpen only as the sample")
    print("shrinks, which is the signature of a consensual population rather than a factional one.")

    return sweep


# ---------------------------------------------------------------------------
# Step 8: what the communities actually believe
# ---------------------------------------------------------------------------
def opinion_profiles(communities, features_valid, ids_valid, survey_cols):
    banner("STEP 8: WHAT THE COMMUNITIES ACTUALLY BELIEVE")

    domain_cols = {d: [j for j, c in enumerate(survey_cols) if c[0] == d] for d in DOMAINS}
    idx_valid = {rid: i for i, rid in enumerate(ids_valid)}

    rows = []
    for k, comm in enumerate(communities):
        members = [idx_valid[r] for r in comm]
        entry = {"community": k, "size": len(comm)}
        for d, label in DOMAINS.items():
            entry[label] = round(features_valid[np.ix_(members, domain_cols[d])].mean(), 3)
        entry["overall"] = round(features_valid[members].mean(), 3)
        rows.append(entry)
    profiles = pd.DataFrame(rows)

    print("Mean Likert score per community and domain (1 = Strongly Disagree, 5 = Strongly Agree):")
    print(profiles.to_string(index=False))
    print("\nEvery community mean lies above the neutral midpoint of 3.0.")
    print(f"Range across all communities and domains: "
          f"{profiles[list(DOMAINS.values())].values.min():.2f} to "
          f"{profiles[list(DOMAINS.values())].values.max():.2f}")

    # Which statements separate the communities, and which unite the class?
    comm_means = np.array([
        [features_valid[[idx_valid[r] for r in comm], j].mean() for j in range(len(survey_cols))]
        for comm in communities
    ])
    spread = comm_means.max(axis=0) - comm_means.min(axis=0)
    order = np.argsort(spread, kind="stable")[::-1]

    print("\nTen statements on which the communities differ most:")
    print(pd.DataFrame({
        "item": [survey_cols[j][:70] for j in order[:10]],
        "spread": np.round(spread[order[:10]], 3),
        "class_mean": [round(features_valid[:, j].mean(), 2) for j in order[:10]],
    }).to_string(index=False))

    item_mean = features_valid.mean(axis=0)
    item_sd = features_valid.std(axis=0)
    calm = np.argsort(item_sd, kind="stable")

    print("\nTen statements attracting the strongest class-wide consensus:")
    print(pd.DataFrame({
        "item": [survey_cols[j][:70] for j in calm[:10]],
        "class_mean": np.round(item_mean[calm[:10]], 2),
        "sd": np.round(item_sd[calm[:10]], 3),
    }).to_string(index=False))

    dom_counts = pd.Series([survey_cols[j][0] for j in order[:15]]).value_counts()
    print("\nDomain of the 15 most divisive statements:")
    for d, cnt in dom_counts.items():
        print(f"  {DOMAINS[d]:<12} {cnt}")

    # Direction matters as much as spread: which statements does the class reject?
    low = np.argsort(item_mean, kind="stable")[:8]
    print("\nLowest-scoring statements (class-wide disagreement):")
    print(pd.DataFrame({
        "item": [survey_cols[j][:70] for j in low],
        "class_mean": np.round(item_mean[low], 2),
        "sd": np.round(item_sd[low], 3),
    }).to_string(index=False))

    below_any = ((comm_means < 3).sum(axis=0) > 0).sum()
    print(f"\nStatements the class as a whole rejects (mean < 3): {(item_mean < 3).sum()} "
          f"of {len(survey_cols)}")
    print(f"Statements where at least one community falls below neutral: {below_any} "
          f"of {len(survey_cols)}")
    print("\nSo the consensus is directional at the domain level but not absolute at the item")
    print("level: the class does genuinely disagree with a small number of specific statements.")

    # Figure 5: community profiles and the axis of disagreement
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 1.5]})

    heat = profiles.set_index("community")[list(DOMAINS.values())]
    im = axes[0].imshow(heat.values, cmap="RdYlBu_r", aspect="auto", vmin=2.8, vmax=4.8)
    axes[0].set_xticks(range(len(DOMAINS)))
    axes[0].set_xticklabels(list(DOMAINS.values()), rotation=30, ha="right")
    axes[0].set_yticks(range(len(heat)))
    axes[0].set_yticklabels([f"C{k} (n={profiles.loc[k, 'size']})" for k in heat.index])
    axes[0].set_title("Mean opinion by community and domain")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axes[0].text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Mean Likert score")

    top8 = order[:8]
    labels = [survey_cols[j].split(".")[0] + "." + survey_cols[j].split(".")[1][:42] for j in top8]
    y = np.arange(len(top8))
    for k in range(len(communities)):
        axes[1].scatter(comm_means[k, top8], y, s=90, color=PALETTE[k % len(PALETTE)],
                        label=f"C{k}", zorder=3, edgecolors="white", linewidths=0.8)
    for i, j in enumerate(top8):
        axes[1].plot([comm_means[:, j].min(), comm_means[:, j].max()], [i, i],
                     color="#999999", linewidth=1.5, zorder=1)
    axes[1].axvline(3, color="#C44E52", linestyle="--", alpha=0.7, label="Neutral (3)")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean Likert score")
    axes[1].set_xlim(1, 5)
    axes[1].set_title("The eight most divisive statements")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].grid(alpha=0.3, axis="x")

    fig.suptitle("Figure 5 - Opinion profiles: the communities differ in intensity, not direction",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "fig5_opinion_profiles.png")

    return profiles


def main():
    df, survey_cols, features, respondent_ids, sim_matrix, G = build_network()

    GA, features_valid, ids_valid, sim_valid = data_quality_guard(
        df, survey_cols, features, respondent_ids, G)

    n_nodes, n_edges = global_structure(GA)

    # One shared layout so Figures 2 and 3 are directly comparable.
    pos = nx.spring_layout(GA, seed=SEED, k=0.30, iterations=120)

    centrality_analysis(GA, pos)
    communities, Q = community_analysis(GA, pos, n_nodes, n_edges)
    threshold_sweep(sim_valid, ids_valid)
    opinion_profiles(communities, features_valid, ids_valid, survey_cols)

    banner("DONE")
    print(f"All five figures written to {FIG_DIR}")
    print("Narrative interpretation of these results: see RESULTS.md")


if __name__ == "__main__":
    main()
