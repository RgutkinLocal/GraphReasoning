import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

st.title("🧠 Graph Reasoning App")

uploaded_file = st.file_uploader("Upload a graph file (.graphml, .gml, .json)", type=["graphml", "gml", "json"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Try reading the graph
    try:
        G = nx.read_graphml(tmp_path)
        st.success(f"Graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges.")

        # Visualize
        net = Network(height="600px", width="100%", notebook=False)
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])
        net.save_graph("graph.html")

        HtmlFile = open("graph.html", 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=600)

    except Exception as e:
        st.error(f"Error loading graph: {e}")

node_start = st.text_input("Enter start keyword or node ID")
node_end = st.text_input("Enter end keyword or node ID")

if st.button("Find shortest path"):
    if G.has_node(node_start) and G.has_node(node_end):
        try:
            path = nx.shortest_path(G, source=node_start, target=node_end)
            st.success(f"Path found: {' → '.join(path)}")
        except nx.NetworkXNoPath:
            st.warning("No path found.")
    else:
        st.warning("One or both nodes not found in graph.")
