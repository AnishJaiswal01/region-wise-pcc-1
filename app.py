import streamlit as st
import json
import tempfile
import pandas as pd
import pydicom
import numpy as np
from regional_cxr_scorer import calculate_regional_priority, get_xrv_predictions

st.set_page_config(page_title="Regional CXR Triage", layout="wide")
st.title("Chest X-Ray Regional Prioritization Demo")

# Load available states dynamically from your generated JSON
with open("state_disease_priors.json", "r") as f:
    state_priors = json.load(f)

available_states = sorted(list(state_priors.keys()))

# Sidebar Controls
st.sidebar.header("Patient Context")
selected_state = st.sidebar.selectbox("Patient Origin State", available_states, index=available_states.index("India") if "India" in available_states else 0)
recent_travel = st.sidebar.checkbox("Recent Travel History (Bypasses Prior)", value=False)

# Update uploader to accept DICOM
uploaded_file = st.file_uploader("Upload Chest X-Ray (PNG, JPG, JPEG, DCM)", type=["png", "jpg", "jpeg", "dcm"])

if uploaded_file is not None:
    # Extract the actual file extension dynamically
    ext = uploaded_file.name.split('.')[-1].lower()

    # Save uploaded file temporarily for torchxrayvision to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    col1, col2 = st.columns([1, 2])

    with col1:
        # Convert DICOM for Streamlit UI display
        if ext == "dcm":
            try:
                dcm = pydicom.dcmread(tmp_path)
                
                # 1. Cast to float immediately to prevent 16-bit integer overflow
                img_array = dcm.pixel_array.astype(np.float64)
                
                # 2. Normalize to 0-255 safely
                img_array = img_array - np.min(img_array)
                max_val = np.max(img_array)
                if max_val > 0:
                    img_array = (img_array / max_val * 255)
                img_array = img_array.astype(np.uint8)
                
                # 3. Fix inverted X-rays (MONOCHROME1 means 0 is white)
                if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
                    img_array = np.invert(img_array)
                    
                st.image(img_array, caption="Uploaded DICOM", use_container_width=True)
            except Exception as e:
                st.error(f"DICOM Display Error: {e}")
        else:
            st.image(uploaded_file, caption="Uploaded Radiograph", use_container_width=True)

    with col2:
        with st.spinner("Running inference and Bayesian weighting..."):
            # Raw predictions vs adjusted predictions
            raw_scores = get_xrv_predictions(tmp_path)
            adjusted_scores = calculate_regional_priority(tmp_path, selected_state, recent_travel)

        # Calculate total mass to normalize posterior distribution
        total_adj = sum(adjusted_scores.values()) or 1.0

        # Format into a clean comparison dataframe
        comparison_data = []
        for disease, adj_score in adjusted_scores.items():
            prior = state_priors.get(selected_state, {}).get(disease, 0.0)
            comparison_data.append({
                "Pathology": disease,
                "Raw AI Confidence (%)": round(float(raw_scores[disease]) * 100, 2),
                "Regional Prior (%)": round(float(max(prior, 0.025)) * 100, 2),
                "Adjusted Priority (%)": round((float(adj_score) / total_adj) * 100, 2)
            })

        df_results = pd.DataFrame(comparison_data)

        # Highlight top prioritized conditions
        st.subheader(f"Triage Rankings for {selected_state}")
        if recent_travel or max(raw_scores.values()) > 0.95:
            st.warning("Emergency override active (high AI confidence or travel history). Raw scores returned.")
        
        st.dataframe(df_results.head(7), use_container_width=True)

        # Visualizing top shifts using the newly normalized percentage columns
        st.bar_chart(df_results.head(5).set_index("Pathology")[["Raw AI Confidence (%)", "Adjusted Priority (%)"]])