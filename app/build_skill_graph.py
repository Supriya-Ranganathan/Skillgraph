from pathlib import Path
from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "job_skill_edges.csv"
OUT_DIR = ROOT / "data" / "processed"
VIZ_DIR = ROOT / "visualizations"

OUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing processed file: {DATA_FILE}"
        )

    edges = pd.read_csv(DATA_FILE)

    # Create bipartite graph: jobs <-> skills
    graph = nx.Graph()

    job_columns = edges[
        ["job_id", "job_title", "category"]
    ].drop_duplicates("job_id")

    for row in job_columns.itertuples(index=False):
        graph.add_node(
            f"job_{row.job_id}",
            node_type="job",
            title=str(row.job_title),
            category=str(row.category),
        )

    for row in edges.itertuples(index=False):
        job_node = f"job_{row.job_id}"
        skill_node = f"skill_{row.skill}"

        graph.add_node(skill_node, node_type="skill")
        graph.add_edge(job_node, skill_node)

    # Save graph statistics
    job_nodes = [
        n for n, d in graph.nodes(data=True)
        if d["node_type"] == "job"
    ]
    skill_nodes = [
        n for n, d in graph.nodes(data=True)
        if d["node_type"] == "skill"
    ]

    print("=" * 55)
    print("SKILLGRAPH-X GRAPH ANALYSIS")
    print("=" * 55)
    print("Job nodes:", len(job_nodes))
    print("Skill nodes:", len(skill_nodes))
    print("Job-skill edges:", graph.number_of_edges())
    print("Connected components:", nx.number_connected_components(graph))

    # Skill frequency
    skill_counts = Counter(edges["skill"])
    top_skills = skill_counts.most_common(15)

    print("\nTop 15 skills:")
    for skill, count in top_skills:
        print(f"{skill}: {count}")

    # Save graph as a reusable GraphML file
    graph_path = OUT_DIR / "skill_graph.graphml"
    nx.write_graphml(graph, graph_path)

    # Save a skill frequency chart
    labels = [item[0] for item in top_skills][::-1]
    values = [item[1] for item in top_skills][::-1]

    plt.figure(figsize=(11, 7))
    plt.barh(labels, values)
    plt.xlabel("Number of job postings")
    plt.ylabel("Skill")
    plt.title("Top 15 Skills Across Job Postings")
    plt.tight_layout()

    chart_path = VIZ_DIR / "top_skills.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved graph:", graph_path)
    print("Saved visualization:", chart_path)

if __name__ == "__main__":
    main()
