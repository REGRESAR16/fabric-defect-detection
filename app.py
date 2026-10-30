from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

ALL_MODELS = {
    "YOLOv8s (final model)": BASE_DIR / "best_v8s.pt",
    "YOLOv8n (lightweight)": BASE_DIR / "best_v8n.pt",
}
MODELS = {name: path for name, path in ALL_MODELS.items() if path.exists()}

st.set_page_config(page_title="Fabric Defect Detection", layout="wide")
st.title("Fabric Defect Detection Using Deep Learning")
st.caption("Upload a fabric image, pick a model, and see the detected defects.")


def show_image(img):
    # Works on both newer and older Streamlit versions.
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.dataframe(df, use_container_width=True)


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


if not MODELS:
    st.error("No model files found. Put best_v8s.pt and/or best_v8n.pt in the same folder as app.py.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(MODELS.keys()))
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.caption("Lower the threshold to find more defects (more false alarms); raise it to be stricter.")

uploaded = st.file_uploader("Upload a fabric image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    model = load_model(str(MODELS[model_name]))
    result = model.predict(image, conf=conf, verbose=False)[0]

    annotated = Image.fromarray(result.plot()[:, :, ::-1])  # BGR -> RGB

    n_defects = len(result.boxes)
    if n_defects:
        st.error(f"Defective: {n_defects} defect(s) detected")
    else:
        st.success("Non-defective: no defects detected at this confidence threshold")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        show_image(image)
    with col2:
        st.subheader("Prediction")
        show_image(annotated)

    st.subheader("Detected defects")
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
        rows.append(
            {
                "Defect class": result.names[int(box.cls[0])],
                "Confidence": round(float(box.conf[0]), 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    if rows:
        show_table(pd.DataFrame(rows))
    else:
        st.info("No defects detected at this confidence threshold.")

    buf = BytesIO()
    annotated.save(buf, format="PNG")
    st.download_button(
        "Download annotated image",
        data=buf.getvalue(),
        file_name=f"pred_{Path(uploaded.name).stem}.png",
        mime="image/png",
    )
