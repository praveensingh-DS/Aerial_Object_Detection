"""
🦅 Bird vs Drone — Aerial Object Classifier
Streamlit App

Models used (from notebook):
  - best_classification_model.h5  (Fine-Tuned MobileNetV2, 224x224, binary sigmoid)
  - /content/runs/detect/bird_drone/weights/best.pt  (YOLOv8n)

Class encoding: bird = 0, drone = 1  (alphabetical, from flow_from_directory)
Pixel normalisation: divide by 255  (rescale=1./255)

Run: streamlit run app.py
"""

import os
import io
import time
import base64
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime

# ── Page config  ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Aerial Classifier",
    page_icon   = "🦅",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Google Fonts + full CSS  ──────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Root tokens ── */
:root {
    --bg:        #05070f;
    --surface:   #0c1020;
    --panel:     #111827;
    --border:    #1e2d4a;
    --accent1:   #38bdf8;   /* sky blue  — bird */
    --accent2:   #fb923c;   /* orange    — drone */
    --accent3:   #a78bfa;   /* violet    — neutral */
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
.main .block-container { padding: 1.5rem 2rem 4rem; max-width: 1300px; }
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
button[kind="header"] { display: none; }

/* ── Hero banner ── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #060d1f 0%, #0a1628 60%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.8rem 3rem 2.2rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(251,146,60,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--accent1);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-family: var(--font-head) !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    color: var(--text) !important;
    margin: 0 0 0.5rem !important;
}
.hero h1 span { color: var(--accent1); }
.hero p {
    color: var(--muted);
    font-size: 0.88rem;
    margin: 0;
    max-width: 560px;
}
.model-badges { display: flex; gap: 0.6rem; margin-top: 1.2rem; flex-wrap: wrap; }
.badge {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    letter-spacing: 0.08em;
}
.badge-blue  { background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); color: var(--accent1); }
.badge-orange{ background: rgba(251,146,60,0.12);  border: 1px solid rgba(251,146,60,0.3);  color: var(--accent2); }
.badge-violet{ background: rgba(167,139,250,0.12); border: 1px solid rgba(167,139,250,0.3); color: var(--accent3); }

/* ── Cards ── */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-label {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
}

/* ── Prediction result ── */
.pred-bird  { border-left: 4px solid var(--accent1); }
.pred-drone { border-left: 4px solid var(--accent2); }
.pred-class {
    font-family: var(--font-head);
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.2rem 0 0.1rem;
}
.pred-bird  .pred-class { color: var(--accent1); }
.pred-drone .pred-class { color: var(--accent2); }
.pred-icon { font-size: 2.2rem; }

/* ── Probability bar ── */
.prob-row { margin: 0.5rem 0; }
.prob-label {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: var(--muted); margin-bottom: 4px;
}
.prob-label b { color: var(--text); }
.prob-track {
    background: var(--border);
    border-radius: 6px; height: 10px; overflow: hidden;
}
.prob-fill-bird  { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #0ea5e9, #38bdf8); transition: width 1s ease; }
.prob-fill-drone { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #ea580c, #fb923c); transition: width 1s ease; }

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.7rem; margin-top: 1rem; }
.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 0.8rem;
    text-align: center;
}
.stat-val {
    font-family: var(--font-head);
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
}
.stat-key { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; }

/* ── Alert ── */
.alert {
    border-radius: 10px; padding: 0.75rem 1rem;
    font-size: 0.82rem; margin: 0.7rem 0;
    display: flex; align-items: center; gap: 0.6rem;
}
.alert-red    { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3);  color: #fca5a5; }
.alert-yellow { background: rgba(234,179,8,0.10);  border: 1px solid rgba(234,179,8,0.3);  color: #fde68a; }
.alert-green  { background: rgba(34,197,94,0.10);  border: 1px solid rgba(34,197,94,0.3);  color: #86efac; }

/* ── History table ── */
.hist-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0.8rem; border-radius: 8px;
    border: 1px solid var(--border); margin: 0.3rem 0;
    font-size: 0.78rem;
}
.hist-bird  { border-left: 3px solid var(--accent1); }
.hist-drone { border-left: 3px solid var(--accent2); }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent1) !important; }

/* ── Sidebar items ── */
.sidebar-section {
    font-size: 0.68rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
    margin: 1.2rem 0 0.5rem;
}
.stRadio > div { gap: 0.4rem; }
.stSlider > div > div > div { background: var(--accent1) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_bird" not in st.session_state:
    st.session_state.total_bird = 0
if "total_drone" not in st.session_state:
    st.session_state.total_drone = 0

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_clf():
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model("models/best_classification_model.h5")
        return m, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        # Try both save locations from the notebook
        for pt in [
            "models/best.pt"
        ]:
            if os.path.exists(pt):
                return YOLO(pt), None
        return None, "best.pt not found in any known location"
    except Exception as e:
        return None, str(e)

clf_model,  clf_err  = load_clf()
yolo_model, yolo_err = load_yolo()

def preprocess(image: Image.Image):
    """Resize to 224×224 and normalise to [0,1] — matches notebook training."""
    img224 = image.resize((224, 224))
    arr    = np.expand_dims(np.array(img224) / 255.0, axis=0)
    return arr.astype(np.float32), img224

def make_gradcam(model, arr, img224):
    """GradCAM overlay using the last Conv layer of MobileNetV2."""
    import tensorflow as tf
    import cv2
    try:
        # Find last conv layer inside the MobileNetV2 base
        last_conv = None
        for layer in model.layers:
            if hasattr(layer, 'layers'):             # Sequential sub-model
                for sub in layer.layers:
                    if 'conv' in sub.name.lower():
                        last_conv = sub.name
            elif 'conv' in layer.name.lower():
                last_conv = layer.name

        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer(last_conv).output, model.output]
        )
        tensor = tf.cast(arr, tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(tensor)
            loss = preds[:, 0]
        grads   = tape.gradient(loss, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0,1,2))
        heatmap = tf.maximum(
            conv_out[0] @ pooled[..., tf.newaxis], 0
        )
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        hm = cv2.resize(heatmap, (224, 224))
        hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_INFERNO)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)

        orig = np.array(img224)
        overlay = (orig * 0.55 + hm * 0.45).clip(0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception:
        return None

def risk_level(label, conf):
    if label == "Drone":
        if conf > 0.90: return "alert-red",    "🔴 HIGH RISK Drone detected with very high confidence. Recommend immediate alert."
        if conf > 0.70: return "alert-yellow",  "🟡 MEDIUM RISK Probable drone. Manual verification recommended."
        return "alert-yellow", "🟡 LOW CONFIDENCE  Uncertain drone prediction. Review manually."
    else:
        if conf > 0.85: return "alert-green",  "🟢 SAFE Bird confirmed with high confidence."
        return "alert-green", "🟢 LIKELY BIRD — Low-confidence bird prediction. Consider re-checking."

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">Deep Learning · Computer Vision · Aerial Surveillance</div>
  <h1>Aerial Object<br><span>Classifier</span></h1>
  <p>Upload an aerial image to instantly classify it as Bird or Drone using
     fine-tuned MobileNetV2 + YOLOv8 object detection.</p>
  <div class="model-badges">
    <span class="badge badge-blue">MobileNetV2 Transfer Learning</span>
    <span class="badge badge-orange">YOLOv8n Detection</span>
    <span class="badge badge-violet">GradCAM Explainability</span>
    <span class="badge badge-blue">224 × 224 · Binary Classification</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">Task</div>', unsafe_allow_html=True)
    task = st.radio(
        "", ["Classification", "YOLOv8 Detection"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section">Options</div>', unsafe_allow_html=True)
    show_gradcam = False
    clf_thresh   = 0.5
    yolo_conf    = 0.25

    if task == "Classification":
        show_gradcam = st.toggle("Show GradCAM Heatmap", value=True)
        clf_thresh   = st.slider("Decision Threshold", 0.30, 0.80, 0.50, 0.05,
                                 help="Probability above this → Drone. Below → Bird.")
    else:
        yolo_conf = st.slider("YOLO Confidence", 0.10, 0.90, 0.25, 0.05)

    st.markdown('<div class="sidebar-section">Model Status</div>', unsafe_allow_html=True)
    if clf_model:
        st.success(f"✅ Classifier loaded  ({clf_model.count_params():,} params)")
    else:
        st.error(f"❌ Classifier: {clf_err}")

    if yolo_model:
        st.success("✅ YOLOv8 loaded")
    else:
        st.warning(f"⚠️ YOLOv8: {yolo_err}")

    # Session stats
    st.markdown('<div class="sidebar-section">Session Stats</div>', unsafe_allow_html=True)
    total = st.session_state.total_bird + st.session_state.total_drone
    st.markdown(f"""
    <div class="stat-grid" style="grid-template-columns:repeat(3,1fr)">
      <div class="stat-box"><div class="stat-val">{total}</div><div class="stat-key">Total</div></div>
      <div class="stat-box"><div class="stat-val" style="color:var(--accent1)">{st.session_state.total_bird}</div><div class="stat-key">Birds</div></div>
      <div class="stat-box"><div class="stat-val" style="color:var(--accent2)">{st.session_state.total_drone}</div><div class="stat-key">Drones</div></div>
    </div>
    """, unsafe_allow_html=True)

    # History
    if st.session_state.history:
        st.markdown('<div class="sidebar-section">Prediction History</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-10:]):
            css  = "hist-bird" if h["label"] == "Bird" else "hist-drone"
            icon = "🐦" if h["label"] == "Bird" else "🚁"
            col  = "var(--accent1)" if h["label"] == "Bird" else "var(--accent2)"
            st.markdown(f"""
            <div class="hist-row {css}">
              <span>{icon} <b style="color:{col}">{h["label"]}</b></span>
              <span style="color:var(--text)">{h["conf"]:.1%}</span>
              <span style="color:var(--muted);font-size:0.7rem">{h["time"]}</span>
            </div>""", unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state.history      = []
            st.session_state.total_bird   = 0
            st.session_state.total_drone  = 0
            st.rerun()

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.markdown('<div class="card-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop an aerial image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, width="stretch")

        w, h   = image.size
        size_kb = round(uploaded.size / 1024, 1)
        st.markdown(f"""
        <div class="card" style="margin-top:0.8rem; padding:0.9rem 1.2rem;">
          <div class="card-label">Image Metadata</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;font-size:0.8rem;">
            <div><span style="color:var(--muted)">Size</span><br><b>{w}×{h}</b></div>
            <div><span style="color:var(--muted)">File</span><br><b>{size_kb} KB</b></div>
            <div><span style="color:var(--muted)">Mode</span><br><b>RGB</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── PREDICTION ────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card-label">Result</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div class="card" style="min-height:300px; display:flex; align-items:center;
             justify-content:center; flex-direction:column; gap:0.8rem; text-align:center;">
          <div style="font-size:3rem">🛰️</div>
          <div style="color:var(--muted); font-size:0.85rem;">
            Upload an aerial image<br>to begin classification
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif task == "Classification":
        if clf_model is None:
            st.error("Classification model not loaded. Ensure `best_classification_model.h5` is in the working directory.")
        else:
            with st.spinner("Analysing..."):
                arr, img224 = preprocess(image)
                raw_score   = float(clf_model.predict(arr, verbose=0)[0][0])

            # bird=0 drone=1 (alphabetical, matches notebook flow_from_directory)
            label      = "Drone" if raw_score > clf_thresh else "Bird"
            confidence = raw_score if label == "Drone" else 1 - raw_score
            bird_prob  = 1 - raw_score
            drone_prob = raw_score
            icon       = "🚁" if label == "Drone" else "🐦"
            pred_css   = "pred-drone" if label == "Drone" else "pred-bird"
            col_val    = "var(--accent2)" if label == "Drone" else "var(--accent1)"

            # ── Main result card
            st.markdown(f"""
            <div class="card {pred_css}">
              <div class="card-label">Prediction</div>
              <div style="display:flex; align-items:center; gap:1rem;">
                <div class="pred-icon">{icon}</div>
                <div>
                  <div class="pred-class">{label}</div>
                  <div style="color:var(--muted); font-size:0.78rem; margin-top:0.1rem;">
                    Confidence: <b style="color:{col_val}">{confidence:.1%}</b>
                    &nbsp;·&nbsp; Raw score: <b>{raw_score:.4f}</b>
                    &nbsp;·&nbsp; Threshold: <b>{clf_thresh}</b>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Probability bars
            st.markdown(f"""
            <div class="card">
              <div class="card-label">Class Probabilities</div>
              <div class="prob-row">
                <div class="prob-label"><span>🐦 Bird</span><b>{bird_prob:.1%}</b></div>
                <div class="prob-track"><div class="prob-fill-bird" style="width:{bird_prob*100:.1f}%"></div></div>
              </div>
              <div class="prob-row" style="margin-top:0.8rem">
                <div class="prob-label"><span>🚁 Drone</span><b>{drone_prob:.1%}</b></div>
                <div class="prob-track"><div class="prob-fill-drone" style="width:{drone_prob*100:.1f}%"></div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Risk alert
            alert_css, alert_msg = risk_level(label, confidence)
            st.markdown(f"""
            <div class="alert {alert_css}">{alert_msg}</div>
            """, unsafe_allow_html=True)

            # ── Stat grid
            certainty = "High" if confidence > 0.85 else "Medium" if confidence > 0.65 else "Low"
            margin    = abs(drone_prob - bird_prob)
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-box">
                <div class="stat-val" style="color:{col_val}">{confidence:.1%}</div>
                <div class="stat-key">Confidence</div>
              </div>
              <div class="stat-box">
                <div class="stat-val">{certainty}</div>
                <div class="stat-key">Certainty</div>
              </div>
              <div class="stat-box">
                <div class="stat-val">{margin:.3f}</div>
                <div class="stat-key">Margin</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── GradCAM
            if show_gradcam:
                st.markdown('<div style="margin-top:1.2rem" class="card-label">GradCAM — Model Attention</div>',
                            unsafe_allow_html=True)
                with st.spinner("Generating GradCAM..."):
                    gc_img = make_gradcam(clf_model, arr, img224)
                if gc_img:
                    g1, g2 = st.columns(2)
                    with g1:
                        st.image(img224, caption="Input (224×224)", use_column_width=True)
                    with g2:
                        st.image(gc_img, caption="GradCAM Overlay", use_column_width=True)
                    st.caption("🔬 Bright regions = areas the model focused on to make this prediction.")
                else:
                    st.caption("GradCAM unavailable for this model architecture.")

            # ── Download report
            report = (
                f"AERIAL OBJECT CLASSIFICATION REPORT\n"
                f"{'─'*40}\n"
                f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"File        : {uploaded.name}\n"
                f"Resolution  : {w}×{h} px\n"
                f"{'─'*40}\n"
                f"Prediction  : {label}\n"
                f"Confidence  : {confidence:.4f} ({confidence:.1%})\n"
                f"Bird Prob   : {bird_prob:.4f}\n"
                f"Drone Prob  : {drone_prob:.4f}\n"
                f"Raw Score   : {raw_score:.4f}\n"
                f"Threshold   : {clf_thresh}\n"
                f"Margin      : {margin:.4f}\n"
                f"Certainty   : {certainty}\n"
                f"{'─'*40}\n"
                f"Model       : Fine-Tuned MobileNetV2\n"
                f"Input Size  : 224×224 px\n"
                f"Normalise   : pixel / 255\n"
                f"Classes     : bird=0, drone=1\n"
            )
            st.download_button(
                "📥 Download Report",
                data      = report,
                file_name = f"aerial_report_{datetime.now().strftime('%H%M%S')}.txt",
                mime      = "text/plain",
                use_container_width=True
            )

            # ── Update history
            st.session_state.history.append({
                "label": label, "conf": confidence,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            if label == "Bird":
                st.session_state.total_bird  += 1
            else:
                st.session_state.total_drone += 1

    # ── YOLO DETECTION ────────────────────────────────────────────────────────
    else:
        if yolo_model is None:
            st.error(f"YOLOv8 model not loaded. {yolo_err}")
        else:
            import cv2
            with st.spinner("Running YOLOv8..."):
                img_cv    = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                results   = yolo_model.predict(img_cv, conf=yolo_conf, imgsz=640, verbose=False)
                annotated = cv2.cvtColor(results[0].plot(line_width=2), cv2.COLOR_BGR2RGB)

            st.image(annotated, caption="YOLOv8 Detection Output", use_column_width=True)

            boxes = results[0].boxes
            CLASS_NAMES = ["Bird", "Drone"]

            if len(boxes) == 0:
                st.markdown("""
                <div class="alert alert-yellow">
                  ⚠️ No objects detected. Try lowering the confidence threshold in the sidebar.
                </div>""", unsafe_allow_html=True)
            else:
                bird_n  = sum(1 for b in boxes if int(b.cls[0]) == 0)
                drone_n = sum(1 for b in boxes if int(b.cls[0]) == 1)

                st.markdown(f"""
                <div class="stat-grid">
                  <div class="stat-box">
                    <div class="stat-val">{len(boxes)}</div>
                    <div class="stat-key">Detected</div>
                  </div>
                  <div class="stat-box">
                    <div class="stat-val" style="color:var(--accent1)">{bird_n}</div>
                    <div class="stat-key">Birds 🐦</div>
                  </div>
                  <div class="stat-box">
                    <div class="stat-val" style="color:var(--accent2)">{drone_n}</div>
                    <div class="stat-key">Drones 🚁</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div style="margin-top:1rem" class="card-label">Detection Details</div>',
                            unsafe_allow_html=True)

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    lbl    = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                    icon   = "🚁" if lbl == "Drone" else "🐦"
                    col    = "var(--accent2)" if lbl == "Drone" else "var(--accent1)"
                    xyxy   = box.xyxy[0].tolist()
                    bw, bh = int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])
                    css    = "hist-drone" if lbl == "Drone" else "hist-bird"

                    st.markdown(f"""
                    <div class="hist-row {css}" style="margin:0.35rem 0">
                      <span>{icon} <b style="color:{col}">{lbl}</b></span>
                      <span>Conf: <b style="color:var(--text)">{conf:.2f}</b></span>
                      <span style="color:var(--muted)">Box {bw}×{bh}px</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if lbl == "Drone":
                        _, alert_msg = risk_level("Drone", conf)
                        alert_css   = "alert-red" if conf > 0.9 else "alert-yellow"
                        st.markdown(f'<div class="alert {alert_css}">{alert_msg}</div>',
                                    unsafe_allow_html=True)

                # history
                if boxes:
                    top  = max(boxes, key=lambda b: float(b.conf[0]))
                    tlbl = CLASS_NAMES[int(top.cls[0])] if int(top.cls[0]) < 2 else "Unknown"
                    st.session_state.history.append({
                        "label": f"[YOLO] {tlbl}",
                        "conf" : float(top.conf[0]),
                        "time" : datetime.now().strftime("%H:%M:%S")
                    })

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1.5rem; border-top:1px solid var(--border);
     text-align:center; color:var(--muted); font-size:0.72rem; letter-spacing:0.08em;">
  AERIAL OBJECT CLASSIFIER &nbsp;·&nbsp; MobileNetV2 + YOLOv8n &nbsp;·&nbsp;
  Bird=0 · Drone=1 &nbsp;·&nbsp; Input 224×224 · Normalised [0,1]
</div>
""", unsafe_allow_html=True)
