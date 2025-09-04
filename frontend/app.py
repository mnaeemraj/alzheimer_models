import os
import requests
import streamlit as st
from dotenv import load_dotenv

# ======================
# Config
# ======================
load_dotenv()
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
DEFAULT_PASSWORD = os.getenv("APP_PASSWORD", "aicenna123")

st.set_page_config(page_title="AiCenna Medical Imaging", page_icon="🧠", layout="centered")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


# ======================
# Login Page
# ======================
def show_login():
    st.title("AiCenna Medical Imaging")
    st.subheader("🔑 Please log in to continue.")
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username")
    with col2:
        password = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if username and password == DEFAULT_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials.")


# ======================
# App Page
# ======================
def show_app():
    st.title("🧠 AiCenna Medical Imaging Prediction Models")
    st.caption("Select a model, target pathology, upload an image, and get probabilities.")

    with st.expander("Backend connection"):
        st.code(f"API: {API_BASE}")

    # Fetch available models
    try:
        resp = requests.get(f"{API_BASE}/models", timeout=20)
        resp.raise_for_status()
        models = resp.json()["available_models"]
    except Exception as e:
        st.error(f"❌ Failed to fetch models: {e}")
        return

    # ======================
    # Model selection
    # ======================
    disease = st.selectbox("Disease Family", list(models.keys()))
    version = st.selectbox("Model Version", list(models[disease].keys()))
    available_targets = models[disease][version]["targets"]
    target = st.selectbox("Target Pathology", available_targets)

    # ======================
    # File upload
    # ======================
    uploaded = st.file_uploader("📤 Upload Medical Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    # ======================
    # Run Prediction
    # ======================
    if uploaded and st.button("Run Prediction", use_container_width=True):
        with st.spinner("Running inference..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
                url = f"{API_BASE}/predict/{disease}/{version}"
                params = {"target": target}
                r = requests.post(url, files=files, params=params, timeout=60)
                r.raise_for_status()
                result = r.json()

                st.success(result.get("message", "✅ Prediction complete"))

                # Display results differently depending on model
                if disease == "ChestX-ray":
                    probs = result["probabilities"]
                    st.metric("Class A (target present)", f"{probs['A']:.3f}")
                    st.metric("Class B (target absent)", f"{probs['B']:.3f}")
                    st.json({"target": result["target"], "probabilities": probs})

                elif disease == "Brain MRI":
                    probs = result["probabilities"]
                    st.metric("Tumor", f"{probs['Tumor']:.3f}")
                    st.metric("No Tumor", f"{probs['No Tumor']:.3f}")
                    st.write(f"**Predicted Class:** {result['predicted_class']}")
                    st.json({
                        "target": result["target"],
                        "predicted_class": result["predicted_class"],
                        "probabilities": probs,
                    })

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request failed: {e}")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

    st.divider()
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()


# ======================
# Main
# ======================
if not st.session_state.logged_in:
    show_login()
else:
    show_app()
