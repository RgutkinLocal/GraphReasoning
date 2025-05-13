import streamlit as st
import networkx as nx
from huggingface_hub import hf_hub_download
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="GraphReasoning Explorer", layout="wide")
st.title("🧠 GraphReasoning Explorer")
st.markdown("Explore the knowledge graph extracted from 1000 scientific papers by [LAMM@MIT](https://github.com/lamm-mit/GraphReasoning).")

# --- Load the .graphml graph
@st.cache_resource
def load_graph():
    graph_name = 'large_graph_simple_giant.graphml'
    file_path = hf_hub_download(
        repo_id='lamm-mit/bio-graph-1K',
        filename=graph_name,
        local_dir='./graph_giant_component'
    )
    G = nx.read_graphml(file_path)
    return G

st.info("Loading the knowledge graph (~3 sec)...")
G = load_graph()
st.success(f"Graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges.")

# --- Show sample nodes to help user
st.subheader("🧾 Sample Nodes")
if st.checkbox("Show 5 example node IDs"):
    for i, node in enumerate(G.nodes()):
        st.write(f"🔹 `{node}`")
        if i >= 4:
            break

# --- Node search by keyword
st.subheader("🔍 Search for Node IDs by Keyword")
keyword = st.text_input("Enter keyword (e.g., 'collagen', 'protein'):")

if keyword:
    matches = [n for n in G.nodes() if keyword.lower() in n.lower()]
    if matches:
        st.write(f"✅ Found {len(matches)} matches:")
        for m in matches[:10]:
            st.write(f"- **{m}**")
        st.code("Copy and paste these IDs into the path finder below.")
    else:
        st.warning("No matches found.")

# --- Path finder
st.subheader("🔗 Find Shortest Path Between Node IDs")

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
            st.write("Path details:")
            for node in path:
                st.write(f"- {node}")
        except nx.NetworkXNoPath:
            st.error("No path found between those nodes.")
    else:
        st.error("One or both node IDs not found. Use the search above to look them up.")

# --- Visualization
st.subheader("🌐 Visualize a Subgraph")

if keyword and matches:
    subG = G.subgraph(matches[:30])  # Keep small for visualization
    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(subG)
    net.repulsion(node_distance=120)
    net.show_buttons(filter_=['physics'])
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
        components.html(html, height=600)
else:
    st.markdown("*Search above to see a related subgraph.*")
