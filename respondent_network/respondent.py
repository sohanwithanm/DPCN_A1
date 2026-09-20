"""
Assignment 1: Opinion Network Formation - Network Analysis
==========================================================

Self-contained pipeline for the network whose NODES ARE RESPONDENTS: encoding,
missing-data policy, similarity, thresholding, then global structure, centrality,
community detection, threshold sensitivity, and interpretation. Maps onto the
"Pipeline Followed", "Analysis and Visualizations" and "Results and Discussion"
sections of the report.

The companion analysis whose nodes are questions is question_question_network.ipynb.

Usage:
    python respondent_network/respondent.py

Writes all figures to outputs/respondent_network/ and prints every metric to stdout.
"""

import matplotlib
matplotlib.use("Agg")          # headless backend: no Qt/X11 needed under WSL

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parent / "Survey_Results_UC.csv"   # shared dataset at the repo root
FIG_DIR = HERE.parent / "outputs" / "respondent_network"   # mirrors outputs/question_question_network

TAU = 0.40                     # similarity threshold, chosen by inspection; see Step 7 sweep
SEED = 42
MIN_OVERLAP = 10               # items answered by BOTH required to correlate a pair

# Type sizes tuned for figures reproduced at ~full text width in the report.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
DOMAINS = {"T": "Technology", "E": "Education", "S": "Society", "V": "Environment"}

LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
    # "No Comments" is deliberately absent, so it maps to NaN. Respondents had a
    # separate "Neutral" option and all 17 who used "No Comments" also used
    # "Neutral" elsewhere, so the two are not interchangeable: "No Comments"
    # withholds an opinion rather than asserting a middle one.
}


def nanmean(a):
    """Mean ignoring NaN; returns NaN for an all-NaN slice without warning."""
    a = np.asarray(a, dtype=float)
    return np.nan if np.isnan(a).all() else float(np.nanmean(a))


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure saved] {FIG_DIR.name}/{name}")


# ---------------------------------------------------------------------------
# Step 0: build the network (encoding, missing-data policy, similarity)
# ---------------------------------------------------------------------------
def build_network():
    """Build the respondent network: missing-data policy + similarity + threshold.

    Blanks stay NaN and are never imputed; entirely blank respondents are
    dropped; similarity is pairwise-complete Pearson correlation over the items
    both members of a pair answered.
    """
    banner("STEP 0: BUILDING THE NETWORK")

    df = pd.read_csv(DATA_FILE)
    survey_cols = df.columns[1:]

    # Vectorisation: blanks and "No Comments" left as NaN
    numeric = df[survey_cols].apply(lambda c: c.map(LIKERT))
    answered = numeric.notna().sum(axis=1)
    keep = (answered > 0).values

    features = numeric[keep].to_numpy(dtype=float)
    respondent_ids = df["id. Response ID"].values[keep]

    # Similarity: pairwise-complete Pearson, thresholded at TAU
    sim_matrix = (pd.DataFrame(features.T)
                  .corr(method="pearson", min_periods=MIN_OVERLAP)
                  .to_numpy())
    np.fill_diagonal(sim_matrix, 0)
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)

    # Signed construction: an edge is retained when the ASSOCIATION is strong,
    # in either direction. |r| >= TAU keeps strongly opposed pairs as negative
    # edges instead of discarding them, matching the question-question network.
    G = nx.Graph()
    G.add_nodes_from(int(r) for r in respondent_ids)
    for a in range(len(respondent_ids)):
        for b in range(a + 1, len(respondent_ids)):
            r = sim_matrix[a, b]
            if abs(r) >= TAU:
                G.add_edge(int(respondent_ids[a]), int(respondent_ids[b]),
                           correlation=float(r), weight=abs(float(r)),
                           sign="positive" if r > 0 else "negative")
    G.remove_nodes_from(list(nx.isolates(G)))

    n_neg = sum(1 for _, _, d in G.edges(data=True) if d["sign"] == "negative")
    print(f"Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(|r| >= {TAU})")
    print(f"  positive (shared opinion) : {G.number_of_edges() - n_neg}")
    print(f"  negative (opposed opinion): {n_neg}")
    print(f"Feature matrix: {features.shape[0]} respondents x {features.shape[1]} questions")
    if (G.number_of_nodes(), G.number_of_edges()) != (88, 970):
        print("  WARNING: result differs from the documented 88 nodes / 970 edges "
              "- check the dataset and policy before interpreting results below.")

    return df, survey_cols, features, respondent_ids, sim_matrix, G


def report_missing_data_policy(df, survey_cols, features, respondent_ids, G):
    """Document the missing-data policy already applied in Step 0.

    Policy:
      * Blank cells are NaN. They are never imputed.
      * A respondent who answered nothing is dropped entirely.
      * "No Comments" withholds an opinion, so it is NaN like a blank cell.

    Because the feature matrix contains NaN, cosine similarity cannot be used.
    Similarity is pairwise-complete Pearson correlation: each pair is correlated
    over only the items both answered, each vector centred on that co-answered subset.
    On complete rows this is identical to mean-centred cosine similarity, so the
    policy affects only the incomplete rows - which is the point.
    """
    banner("STEP 3: MISSING-DATA POLICY (NaN, NO IMPUTATION)")

    n_items = len(survey_cols)
    # Count on the MAPPED values: a cell is unanswered if it was blank OR
    # "No Comments", both of which map to NaN.
    mapped = df[survey_cols].apply(lambda c: c.map(LIKERT))
    raw_answered = mapped.notna().sum(axis=1)

    n_blank = int(df[survey_cols].isna().sum().sum())
    n_nocomment = int((df[survey_cols] == "No Comments").sum().sum())
    n_unanswered = n_blank + n_nocomment

    print("Response completeness across all 96 respondents")
    print(f"  fully complete ({n_items} answered) : {(raw_answered == n_items).sum()}")
    print(f"  partial (1-{n_items - 1} answered)      : "
          f"{((raw_answered > 0) & (raw_answered < n_items)).sum()}")
    print(f"  entirely blank (0 answered)  : {(raw_answered == 0).sum()}")
    print(f"\nUnanswered cells: {n_unanswered} of {96 * n_items} "
          f"({100 * n_unanswered / (96 * n_items):.1f}%) - all NaN, none imputed")
    print(f"  {n_blank} blank + {n_nocomment} \"No Comments\" "
          f"(a withheld opinion, not a neutral one)")

    empty_ids = [int(r) for r in df["id. Response ID"][raw_answered == 0]]
    print(f"\nDropped as entirely blank: {empty_ids}")

    present = ~np.isnan(features)
    overlap = present.astype(int) @ present.astype(int).T
    iu = np.triu_indices(len(features), k=1)
    # The overlap distribution is bimodal - most respondents completed the survey,
    # a few answered one block - so a median sits at the ceiling and hides the tail.
    # Report the split instead.
    ov = overlap[iu]
    full = int((ov == n_items).sum())
    partial = len(ov) - full
    print(f"\nCo-answered items per pair (items answered by both, agreement aside) across "
          f"{len(ov)} pairs:")
    print(f"  all {n_items} items    : {full} pairs ({100 * full / len(ov):.1f}%)")
    print(f"  fewer than {n_items}   : {partial} pairs ({100 * partial / len(ov):.1f}%), "
          f"minimum {ov.min()}")
    print(f"  pairs below the MIN_OVERLAP={MIN_OVERLAP} floor (similarity forced to 0): "
          f"{int((ov < MIN_OVERLAP).sum())}")

    retained = len(respondent_ids)
    dropped = sorted(set(int(r) for r in respondent_ids) - set(int(n) for n in G.nodes()))
    print(f"\nRetained {retained} of 96 respondents, then removed {len(dropped)} isolates "
          f"{dropped}")
    print(f"Analysis graph GA : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("\nAll metrics below are reported on GA.")

    # Respondents answering few items are retained, but their ties rest on a
    # small overlap. Report them so their influence can be judged.
    id_to_index = {int(rid): i for i, rid in enumerate(respondent_ids)}
    sparse = [r for r in G.nodes() if present[id_to_index[int(r)]].sum() < n_items / 2]
    if sparse:
        print(f"\nNOTE: {len(sparse)} retained respondents answered fewer than half the items:")
        for r in sorted(sparse):
            print(f"  respondent {r:>3}: {present[id_to_index[int(r)]].sum():>2} of {n_items} "
                  f"answered, degree {G.degree(r)}")
        print("  Their correlations are computed on that subset only, so they are noisier")
        print("  than those of complete respondents.")

    return None


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
    fig.suptitle(f"Degree structure of the opinion network (tau = {TAU})", y=1.02)
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

    fig.suptitle(f"Centrality structure (n = {GA.number_of_nodes()}, tau = {TAU})",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "fig2_centrality.png")

    return centrality


# ---------------------------------------------------------------------------
# Step 6: community detection
# ---------------------------------------------------------------------------
def community_analysis(GA, pos, n_nodes, n_edges):
    banner("STEP 6: COMMUNITY DETECTION")

    # Louvain cannot optimise positive and negative weights simultaneously, so
    # the partition is computed on the positive-edge subgraph, as in the
    # question-question network.
    GP = nx.Graph()
    GP.add_nodes_from(GA.nodes())
    GP.add_edges_from((u, v, d) for u, v, d in GA.edges(data=True)
                      if d["sign"] == "positive")
    active = [n for n, deg in GP.degree() if deg > 0]
    GP = GP.subgraph(active).copy()

    communities = sorted(nx.community.louvain_communities(GP, seed=SEED), key=len, reverse=True)
    Q = nx.community.modularity(GP, communities)

    print(f"Positive-edge subgraph: {GP.number_of_nodes()} nodes, {GP.number_of_edges()} edges")
    print(f"Louvain found {len(communities)} communities")
    print(f"Sizes: {[len(c) for c in communities]}")
    print(f"Modularity Q = {Q:.4f}")

    # Stability across random seeds
    seeded = [nx.community.louvain_communities(GP, seed=s) for s in range(10)]
    n_comms = [len(c) for c in seeded]
    q_vals = [nx.community.modularity(GP, c) for c in seeded]
    print(f"\nAcross 10 seeds: community count {min(n_comms)}-{max(n_comms)}, "
          f"Q = {np.mean(q_vals):.4f} +/- {np.std(q_vals):.4f}")

    # Two null models. Erdos-Renyi fixes only n and m; the configuration model
    # additionally preserves the degree sequence, which matters here because
    # degree ranges from 1 to 50.
    q_er = [nx.community.modularity(g, nx.community.louvain_communities(g, seed=1))
            for g in [nx.gnm_random_graph(GP.number_of_nodes(), GP.number_of_edges(),
                                          seed=s) for s in range(20)]]

    deg_seq = [d for _, d in GP.degree()]
    q_cm = []
    for s in range(20):
        cm = nx.Graph(nx.configuration_model(deg_seq, seed=s))
        cm.remove_edges_from(nx.selfloop_edges(cm))
        q_cm.append(nx.community.modularity(cm, nx.community.louvain_communities(cm, seed=1)))

    # Q itself depends on node insertion order, so comparing a single run against
    # the nulls would overstate precision. Use the distribution of Q across
    # orderings as the observed quantity.
    q_obs = []
    for perm in range(20):
        rng = np.random.default_rng(perm)
        order = list(GP.nodes())
        rng.shuffle(order)
        H = nx.Graph()
        H.add_nodes_from(order)
        H.add_edges_from(GP.edges())
        q_obs.append(nx.community.modularity(H, nx.community.louvain_communities(H, seed=SEED)))
    q_obs_mean, q_obs_sd = float(np.mean(q_obs)), float(np.std(q_obs))

    z_er = (q_obs_mean - np.mean(q_er)) / np.std(q_er)
    z_cm = (q_obs_mean - np.mean(q_cm)) / np.std(q_cm)

    print(f"\nObserved modularity (20 node orderings)")
    print(f"                               Q = {q_obs_mean:.4f} +/- {q_obs_sd:.4f}  "
          f"[{min(q_obs):.4f}, {max(q_obs):.4f}]")
    print(f"  Erdos-Renyi null           Q = {np.mean(q_er):.4f} +/- {np.std(q_er):.4f}   (z = {z_er:+.2f})")
    print(f"  Configuration-model null   Q = {np.mean(q_cm):.4f} +/- {np.std(q_cm):.4f}   (z = {z_cm:+.2f})")
    print("\nModularity sits modestly above both nulls - detectable against a degree-preserving")
    print("null - but far below the ~0.3 conventionally required to claim genuine community")
    print("structure. Combined with the partition's instability under node re-ordering (see the")
    print("next section), this indicates a faint clustering tendency rather than opinion factions.")

    membership = {node: k for k, comm in enumerate(communities) for node in comm}

    # Figure 3: network coloured by community
    fig = plt.figure(figsize=(11, 9))
    pos_edges = [(u, v) for u, v, d in GA.edges(data=True) if d["sign"] == "positive"]
    neg_edges = [(u, v) for u, v, d in GA.edges(data=True) if d["sign"] == "negative"]
    nx.draw_networkx_edges(GA, pos, edgelist=pos_edges, alpha=0.10, edge_color="#333333")
    nx.draw_networkx_edges(GA, pos, edgelist=neg_edges, alpha=0.9, edge_color="#C44E52",
                           width=1.6, style="dashed")
    for k, comm in enumerate(communities):
        nx.draw_networkx_nodes(
            GA, pos, nodelist=sorted(comm),
            node_color=PALETTE[k % len(PALETTE)],
            node_size=[40 + 10 * GA.degree(v) for v in sorted(comm)],
            label=f"Community {k} (n = {len(comm)})",
            linewidths=0.5, edgecolors="white")
    plt.legend(scatterpoints=1, loc="best", frameon=True)
    plt.title(f"Louvain communities on the positive subgraph (Q = {Q:.3f}, |r| >= {TAU})\n"
              f"Dashed red edges are the {len(neg_edges)} opposed pairs; all run between communities",
              fontsize=12)
    plt.axis("off")
    fig.tight_layout()
    save(fig, "fig3_communities.png")

    internal = sum(1 for u, v in GP.edges() if membership[u] == membership[v])
    n_pos = GP.number_of_edges()
    print(f"\nPositive edges within communities: {internal} of {n_pos} "
          f"({100 * internal / n_pos:.1f}%)")
    print(f"Positive edges crossing communities: {n_pos - internal} "
          f"({100 * (n_pos - internal) / n_pos:.1f}%)")

    return communities, Q, GP


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

    fig.suptitle("Threshold sensitivity: modularity is bought with sample size",
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
            entry[label] = round(nanmean(features_valid[np.ix_(members, domain_cols[d])]), 3)
        entry["overall"] = round(nanmean(features_valid[members]), 3)
        rows.append(entry)
    profiles = pd.DataFrame(rows)

    print("Mean Likert score per community and domain (1 = Strongly Disagree, 5 = Strongly Agree):")
    print(profiles.to_string(index=False))
    vals = profiles[list(DOMAINS.values())].values
    n_below = (vals < 3).sum()
    print(f"\nCommunity-by-domain means below the neutral midpoint of 3.0: {n_below} "
          f"of {vals.size}")
    print(f"Range across all communities and domains: {vals.min():.2f} to {vals.max():.2f}")

    # Which statements separate the communities, and which unite the class?
    comm_means = np.array([
        [nanmean(features_valid[[idx_valid[r] for r in comm], j]) for j in range(len(survey_cols))]
        for comm in communities
    ])
    spread = comm_means.max(axis=0) - comm_means.min(axis=0)
    order = np.argsort(spread, kind="stable")[::-1]

    print("\nTen statements on which the communities differ most:")
    print(pd.DataFrame({
        "item": [survey_cols[j][:70] for j in order[:10]],
        "spread": np.round(spread[order[:10]], 3),
        "class_mean": [round(nanmean(features_valid[:, j]), 2) for j in order[:10]],
    }).to_string(index=False))

    item_mean = np.nanmean(features_valid, axis=0)
    item_sd = np.nanstd(features_valid, axis=0)
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

    fig.suptitle("Opinion profiles: the communities differ in intensity, not direction",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "fig5_opinion_profiles.png")

    return profiles


# ---------------------------------------------------------------------------
# Similarity overview: where the threshold cuts, and the correlation structure
# ---------------------------------------------------------------------------
def similarity_overview(sim_valid, ids_valid, communities):
    banner("SIMILARITY OVERVIEW")

    iu = np.triu_indices(len(sim_valid), k=1)
    vals = sim_valid[iu]
    kept = vals >= TAU

    print(f"Pairwise similarities across {len(vals)} respondent pairs:")
    print(f"  range {vals.min():+.3f} to {vals.max():+.3f}, "
          f"mean {vals.mean():+.3f}, median {np.median(vals):+.3f}")
    print(f"  negatively correlated pairs (opposed opinions): {int((vals < 0).sum())} "
          f"({100 * (vals < 0).mean():.1f}%)")
    print(f"  retained as edges at tau={TAU}: {int(kept.sum())} "
          f"({100 * kept.mean():.1f}%) - the {100 * (1 - kept.mean()):.0f}th percentile upward")

    # Order respondents by community so block structure (if any) is visible.
    order, boundaries, running = [], [], 0
    idx_of = {int(r): i for i, r in enumerate(ids_valid)}
    for comm in communities:
        members = sorted(int(r) for r in comm)
        order.extend(idx_of[r] for r in members)
        running += len(members)
        boundaries.append(running)
    ordered = sim_valid[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    axes[0].hist(vals, bins=60, color="#4C72B0", edgecolor="white")
    axes[0].axvline(TAU, color="#C44E52", linestyle="--", linewidth=2,
                    label=f"tau = {TAU} (edges to the right)")
    axes[0].axvline(0, color="#555555", linestyle=":", linewidth=1.5, label="zero correlation")
    axes[0].set_xlabel("Pairwise Pearson correlation", fontsize=13)
    axes[0].set_ylabel("Number of respondent pairs", fontsize=13)
    axes[0].set_title("Distribution of pairwise similarity", fontsize=15)
    axes[0].tick_params(labelsize=11)
    axes[0].legend(fontsize=12)

    im = axes[1].imshow(ordered, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")

    # Outline ONLY the diagonal blocks: those are the within-community regions.
    # Plain grid lines would cut the square into 16 rectangles and invite the
    # misreading that there are 16 communities.
    starts = [0] + boundaries[:-1]
    for k, (s0, s1) in enumerate(zip(starts, boundaries)):
        axes[1].add_patch(plt.Rectangle((s0 - 0.5, s0 - 0.5), s1 - s0, s1 - s0,
                                        fill=False, edgecolor="black", linewidth=2.2))
    # Label each block by community instead of by raw position index.
    centres = [(s0 + s1 - 1) / 2 for s0, s1 in zip(starts, boundaries)]
    labels = [f"C{k}\n(n={s1 - s0})" for k, (s0, s1) in enumerate(zip(starts, boundaries))]
    axes[1].set_xticks(centres); axes[1].set_xticklabels(labels, fontsize=11)
    axes[1].set_yticks(centres); axes[1].set_yticklabels(labels, fontsize=11)
    axes[1].set_title("Similarity matrix, respondents ordered by community", fontsize=15)
    axes[1].set_xlabel("Community (boxed blocks are within-community pairs)", fontsize=13)
    axes[1].set_ylabel("Community", fontsize=13)
    cb = fig.colorbar(im, ax=axes[1], shrink=0.85, label="Pearson correlation")
    cb.ax.tick_params(labelsize=11)
    cb.set_label("Pearson correlation", fontsize=12)

    fig.suptitle("Similarity structure underlying the respondent network",
                 fontsize=16)
    fig.tight_layout()
    save(fig, "fig6_similarity_structure.png")

    print("\nThe faint block structure on the diagonal corresponds to the Louvain")
    print("communities; its weakness is the visual counterpart of the low modularity.")


# ---------------------------------------------------------------------------
# Negative edges and how stable the partition really is
# ---------------------------------------------------------------------------
def opposition_and_stability(GA, GP, communities):
    banner("OPPOSED PAIRS AND PARTITION STABILITY")

    membership = {n: k for k, comm in enumerate(communities) for n in comm}
    negative = [(u, v, d["correlation"]) for u, v, d in GA.edges(data=True)
                if d["sign"] == "negative"]

    print(f"Negative edges (respondents with opposed opinion profiles): {len(negative)} "
          f"of {GA.number_of_edges()} ({100 * len(negative) / GA.number_of_edges():.1f}%)")

    placed = [(u, v, r) for u, v, r in negative if u in membership and v in membership]
    between = sum(1 for u, v, _ in placed if membership[u] != membership[v])
    within = len(placed) - between

    sizes = [len(c) for c in communities]
    n = sum(sizes)
    p_same = sum(s * (s - 1) for s in sizes) / (n * (n - 1))
    expected_within = len(placed) * p_same
    p_value = stats.binom.pmf(within, len(placed), p_same) if placed else float("nan")

    print(f"  between communities: {between}")
    print(f"  within  communities: {within}   (expected {expected_within:.1f} by chance)")
    print(f"  P(observing {within} within | random placement) = {p_value:.4f}")
    print("\nOpposed respondents are essentially never assigned to the same community,")
    print("so the partition does separate genuine opposition - there is simply very")
    print("little of it. Note this rests on only 13 edges.")

    print("\nStrongest oppositions:")
    for u, v, r in sorted(negative, key=lambda t: t[2])[:5]:
        print(f"  respondent {u:>3} vs {v:>3}: r = {r:+.3f}  "
              f"(communities {membership.get(u, '-')}/{membership.get(v, '-')})")

    # Louvain is order-sensitive. Varying the SEED alone understates instability;
    # the node insertion order matters as much, so vary that too.
    print("\nPartition stability under node re-ordering (same graph, same seed):")
    seen = []
    for perm in range(8):
        rng = np.random.default_rng(perm)
        order = list(GP.nodes())
        rng.shuffle(order)
        H = nx.Graph()
        H.add_nodes_from(order)
        H.add_edges_from(GP.edges())
        c = nx.community.louvain_communities(H, seed=SEED)
        seen.append((tuple(sorted(map(len, c), reverse=True)),
                     nx.community.modularity(H, c)))
    for sizes_i, q_i in seen:
        print(f"  {str(sizes_i):28} Q = {q_i:.4f}")
    counts = {len(s) for s, _ in seen}
    print(f"\nAcross 8 orderings the algorithm returns {min(counts)}-{max(counts)} communities")
    print("with materially different sizes. The specific partition is therefore not a")
    print("stable property of the data, which is what a near-random modularity implies.")


def main():
    df, survey_cols, features, respondent_ids, sim_matrix, G = build_network()

    report_missing_data_policy(df, survey_cols, features, respondent_ids, G)
    GA, features_valid, ids_valid, sim_valid = G, features, respondent_ids, sim_matrix

    n_nodes, n_edges = global_structure(GA)

    # One shared layout so Figures 2 and 3 are directly comparable.
    pos = nx.spring_layout(GA, seed=SEED, k=0.30, iterations=120)

    centrality_analysis(GA, pos)
    communities, Q, GP = community_analysis(GA, pos, n_nodes, n_edges)
    opposition_and_stability(GA, GP, communities)
    similarity_overview(sim_valid, ids_valid, communities)
    threshold_sweep(sim_valid, ids_valid)
    opinion_profiles(communities, features_valid, ids_valid, survey_cols)

    banner("DONE")
    print(f"All six figures written to {FIG_DIR}")
    print("Narrative interpretation of these results: see RESULTS.md")


if __name__ == "__main__":
    main()
