import streamlit as st
import networkx as nx
from huggingface_hub import hf_hub_download
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

st.set_page_config(page_title="GraphReasoning Explorer", layout="wide")
st.title("🧠 GraphReasoning Explorer")
st.markdown("Explore the [LAMM@MIT GraphReasoning](https://github.com/lamm-mit/GraphReasoning) knowledge graph from 1000 scientific papers.")

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

st.info("Loading the full graph from Hugging Face (~3 sec)...")
G = load_graph()
st.success(f"Graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges.")
st.subheader("🧾 Inspect Sample Nodes")

if st.checkbox("Show 5 example nodes"):
    for i, (node_id, attr) in enumerate(G.nodes(data=True)):
        st.write(f"🔹 ID: `{node_id}`")
        st.write(attr)
        if i >= 4:
            break

# --- Search nodes by label
st.subheader("🔍 Search for a Node")
keyword = st.text_input("Enter keyword to search node labels:")

if keyword:
    matches = [n for n, d in G.nodes(data=True)
               if keyword.lower() in str(d.get('label', '')).lower()]
    if matches:
        st.write(f"✅ Found {len(matches)} matching nodes:")
        for m in matches[:10]:
            st.write(f"- **{G.nodes[m].get('label', m)}** (ID: `{m}`)")
    else:
        st.warning("No matches found.")

# --- Shortest path search
st.subheader("🔗 Find Shortest Path Between Nodes")

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
            st.write("Path labels:")
            for n in path:
                st.write(f"- {G.nodes[n].get('label', n)}")
        except nx.NetworkXNoPath:
            st.error("No path found.")
    else:
        st.error("One or both nodes not found. Use the search above to get node IDs.")

# --- Graph visualization
st.subheader("🌐 Graph Visualization")

if keyword and matches:
    subG = G.subgraph(matches[:30])  # limit to small view
    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(subG)
    net.show_buttons(filter_=['physics'])
    net.repulsion(node_distance=120)
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
        components.html(html, height=600)
else:
    st.markdown("*Search for a keyword above to view a local subgraph.*")
