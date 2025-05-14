# app.py
import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from pyvis.network import Network
import streamlit.components.v1 as components
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.express as px

st.set_page_config(page_title="GraphReasoning Explorer", layout="wide")
st.title("🧠 GraphReasoning Explorer")
st.markdown("Explore the scientific knowledge graph extracted by [LAMM@MIT](https://github.com/lamm-mit/GraphReasoning).")

# --- Load GraphML
@st.cache_resource
def load_graph():
    file_path = hf_hub_download(
        repo_id='lamm-mit/bio-graph-1K',
        filename='large_graph_simple_giant.graphml',
        local_dir='./graph_data'
    )
    G = nx.read_graphml(file_path)
    return G

# --- Load Embeddings
@st.cache_resource
def load_embeddings():
    path = hf_hub_download(
        repo_id='lamm-mit/bio-graph-1K',
        filename='embeddings_simple_giant_ge-large-en-v1.5.pkl',
        local_dir='./graph_data'
    )
    with open(path, 'rb') as f:
        raw = pickle.load(f)

    # Clean and flatten all vectors
    embeddings = {k: np.array(v).squeeze() for k, v in raw.items()}
    return embeddings

# --- Reduce to 2D
@st.cache_resource
def reduce_embeddings(embeddings):
    keys = list(embeddings.keys())
    mat = np.stack([np.array(embeddings[k]).squeeze() for k in keys])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    proj = reducer.fit_transform(mat)
    df = pd.DataFrame(proj, columns=["x", "y"])
    df["label"] = keys
    return df

# --- Node similarity function
def get_top_similar_nodes(embeddings, node_id, top_k=5):
    if node_id not in embeddings:
        return []
    vec = embeddings[node_id].reshape(1, -1)
    keys = list(embeddings.keys())
    mat = np.stack([embeddings[k] for k in keys])
    sims = cosine_similarity(vec, mat)[0]
    top_indices = np.argsort(-sims)[1:top_k+1]  # exclude self
    return [(keys[i], sims[i]) for i in top_indices]

# --- Load Graph
st.info("Loading graph (~3 sec)...")
G = load_graph()
st.success(f"Graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges.")

# --- Sample Nodes
st.subheader("📎 Sample Nodes")
if st.checkbox("Show 5 example node IDs"):
    for i, node in enumerate(G.nodes()):
        st.write(f"- `{node}`")
        if i >= 4:
            break

# --- Search Node
st.subheader("🔍 Search Nodes")
keyword = st.text_input("Search keyword (e.g., collagen, structure)")
if keyword:
    matches = [n for n in G.nodes() if keyword.lower() in n.lower()]
    if matches:
        st.success(f"Found {len(matches)} nodes:")
        st.write(matches[:10])
        st.code("Use these node IDs below.")
    else:
        st.warning("No matches found.")

# --- Path Finder
st.subheader("🔗 Find Shortest Path")
col1, col2 = st.columns(2)
with col1:
    start_node = st.text_input("Start node ID")
with col2:
    end_node = st.text_input("End node ID")

if st.button("Find Path"):
    if start_node in G and end_node in G:
        try:
            path = nx.shortest_path(G, source=start_node, target=end_node)
            st.success(" → ".join(path))
        except nx.NetworkXNoPath:
            st.error("No path found.")
    else:
        st.error("Invalid node IDs.")

# --- Node Similarity
st.subheader("🤝 Node Similarity Search")
sim_node = st.text_input("Enter node ID for similarity")
if sim_node:
    embeddings = load_embeddings()
    if sim_node in embeddings:
        results = get_top_similar_nodes(embeddings, sim_node)
        st.success("Top similar nodes:")
        for n, score in results:
            st.write(f"- {n} (similarity: {score:.3f})")
    else:
        st.warning("Node not in embeddings.")

# --- Embedding Map
st.subheader("🗺️ 2D Graph Map")
if st.button("Visualize All Nodes in 2D"):
    embeddings = load_embeddings()
    df = reduce_embeddings(embeddings)
    fig = px.scatter(df, x="x", y="y", hover_name="label", width=900, height=600)
    st.plotly_chart(fig)

# --- Visualization
st.subheader("🌐 Subgraph Visualization")
if keyword and matches:
    subG = G.subgraph(matches[:30])
    net = Network(height="600px", width="100%")
    net.from_nx(subG)
    net.repulsion(node_distance=120)
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
        components.html(html, height=600)
else:
    st.markdown("*Search to view a subgraph.*")

# --- Colab Link
st.subheader("🚀 Launch in Colab for Reasoning")
st.markdown("Use [this Colab notebook](https://colab.research.google.com/github/RgutkinLocal/GraphReasoning/blob/main/graph_reasoning_colab.ipynb) to run GPT-based reasoning or large compute jobs.")
