# streamlit_client.py
import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_BASE = st.text_input("Backend URL", value=os.getenv("FASTAPI_URL", "http://localhost:8000"))

st.title("Groq LLM — Streamlit Demo Client")

tabs = st.tabs(["Text Chat", "Image → Caption / Q&A", "Items (CRUD demo)"])

# ---------------- Text Chat ----------------
with tabs[0]:
    st.header("Text Chat")
    model = st.text_input("Model", value="openai/gpt-oss-20b")
    prompt = st.text_area("Prompt", value="Write a short friendly greeting.")
    if st.button("Send text prompt"):
        payload = {"prompt": prompt, "model": model}
        r = requests.post(f"{API_BASE}/llm/text", json=payload)
        if r.ok:
            data = r.json()
            st.subheader("Output")
            st.write(data.get("output_text") or data.get("result_raw"))
        else:
            st.error(f"Error: {r.status_code} {r.text}")

# ---------------- Image ----------------
with tabs[1]:
    st.header("Image + Instruction")
    uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    instr = st.text_input("Instruction", value="Describe the image in one paragraph.")
    model_img = st.text_input("Model (vision-capable)", value="openai/gpt-oss-20b")
    if st.button("Send image"):
        if not uploaded:
            st.warning("Upload an image first.")
        else:
            files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
            data = {"instruction": instr, "model": model_img}
            r = requests.post(f"{API_BASE}/llm/image", files=files, data=data)
            if r.ok:
                res = r.json()
                st.subheader("Model response")
                st.json(res.get("result_raw") or res)
            else:
                st.error(f"Error: {r.status_code} {r.text}")

# ---------------- Items CRUD ----------------
with tabs[2]:
    st.header("In-memory Items (CRUD demo)")
    st.subheader("Create new item")
    t = st.text_input("Title")
    c = st.text_area("Content")
    if st.button("Create item"):
        payload = {"title": t, "content": c}
        r = requests.post(f"{API_BASE}/items", json=payload)
        st.write(r.status_code, r.text)
    st.subheader("List items")
    if st.button("Refresh items"):
        r = requests.get(f"{API_BASE}/items")
        if r.ok:
            st.json(r.json())
        else:
            st.error(f"Error: {r.status_code} {r.text}")

    st.subheader("Delete item")
    delete_id = st.text_input("ID to delete")
    if st.button("Delete"):
        r = requests.delete(f"{API_BASE}/items/{delete_id}")
        st.write(r.status_code, r.text)
