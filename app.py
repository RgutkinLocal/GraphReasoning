import streamlit as st

st.title("GraphReasoning Explorer")

st.write("This app will help you reason over knowledge graphs from scientific literature.")
st.write("Stay tuned for graph upload, querying, and AI-driven insights!")

# Optional test
name = st.text_input("What's your name?")
if name:
    st.success(f"Hi {name}, welcome!")
