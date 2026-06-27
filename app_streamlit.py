"""ANPR Streamlit Dashboard.

Premium interactive front-end for the ANPR system.
Communicates with the running FastAPI backend at BASE_URL.

Usage:
    streamlit run app_streamlit.py
"""

import io
import time
from pathlib import Path

import requests
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a4a;
}
section[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: #1f1f35;
    border: 1px solid #3a3a5c;
    border-radius: 8px;
    color: #e0e0f0;
}

/* ── Main background ── */
.main { background: #0a0a14; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Status badge ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-success  { background: #0d3320; color: #4ade80; border: 1px solid #166534; }
.badge-warning  { background: #2d1e00; color: #fbbf24; border: 1px solid #92400e; }
.badge-error    { background: #2d0a0a; color: #f87171; border: 1px solid #991b1b; }
.badge-info     { background: #0a1f3d; color: #60a5fa; border: 1px solid #1e3a5f; }

/* ── Metric card ── */
.metric-card {
    background: linear-gradient(135deg, #12122a 0%, #1a1a35 100%);
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #5b5bf6; }
.metric-label { font-size: 0.72rem; color: #8080b0; letter-spacing: 0.08em;
                text-transform: uppercase; font-weight: 500; margin-bottom: 4px; }
.metric-value { font-size: 1.55rem; font-weight: 700; color: #e8e8ff; line-height: 1.2; }
.metric-sub   { font-size: 0.8rem; color: #6060a0; margin-top: 2px; }

/* ── Stage bar ── */
.bar-row { display: flex; align-items: center; margin-bottom: 8px; gap: 10px; }
.bar-label { font-size: 0.76rem; color: #9090c0; min-width: 140px; text-align: right; }
.bar-track { flex: 1; background: #1a1a35; border-radius: 6px; height: 18px;
             overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 6px;
             background: linear-gradient(90deg, #5b5bf6, #9b5bfb); }
.bar-ms    { font-size: 0.76rem; color: #7070b0; min-width: 70px; }

/* ── Plate number display ── */
.plate-box {
    background: #f5f0d0;
    border: 4px solid #c8a800;
    border-radius: 10px;
    padding: 14px 28px;
    display: inline-block;
    font-family: 'Courier New', monospace;
    font-size: 2.2rem;
    font-weight: 800;
    color: #111;
    letter-spacing: 0.15em;
    box-shadow: 0 4px 24px #c8a80044;
    margin: 12px 0;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #2a2a4a; margin: 1.5rem 0; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #5b5bf6, #9b5bfb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover  { opacity: 0.85; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

[data-testid="stFileUploaderDropzone"] {
    background: #12122a;
    border: 2px dashed #3a3a5c;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    api_host = st.text_input("API Host", value="127.0.0.1")
    api_port = st.text_input("API Port", value="8000")
    BASE_URL = f"http://{api_host}:{api_port}"

    # Server health check
    st.markdown("### Server Status")
    _health = {}
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=3)
        _health    = resp.json()
        db_status  = _health.get("database", "unknown")
        mdl_status = _health.get("models", "unknown")
        st.markdown("🟢 **API** — Online")
        db_badge = "🟢" if db_status == "connected" else "🟡"
        st.markdown(f"{db_badge} **Database** — `{db_status}`")
        st.markdown(f"🔵 **Models** — `{mdl_status}`")
    except Exception:
        st.markdown("🔴 **API** — Offline")
        st.caption(f"Cannot reach {BASE_URL}/health")

    # System information panel
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🖥️ System Information")
    if _health:
        gpu_ok   = _health.get("gpu_available", False)
        gpu_icon = "🟢" if gpu_ok else "⚪"
        st.markdown(f"{gpu_icon} **GPU** — `{'Available' if gpu_ok else 'Not available'}`")
        if gpu_ok:
            st.caption(f"🎮 {_health.get('gpu_name', 'N/A')}")
        st.caption(f"CUDA:          `{_health.get('cuda_version', 'N/A')}`")
        st.caption(f"PyTorch:       `{_health.get('torch_version', 'N/A')}`")
        st.caption(f"Ultralytics:   `{_health.get('ultralytics_version', 'N/A')}`")
        st.caption(f"PaddleOCR:     `{_health.get('paddleocr_version', 'N/A')}`")
    else:
        st.caption("Connect to API to view system info.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.caption("ANPR System v5.0 · All Phases Complete")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1rem'>
    <h1 style='color:#e8e8ff; font-weight:800; margin-bottom:2px; font-size:2.2rem;'>
        🚗 ANPR Detection System
    </h1>
    <p style='color:#6060a0; font-size:0.95rem; margin-top:0;'>
        Automatic Number Plate Recognition · Powered by YOLO11s + PP-OCRv5
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_detect, tab_history = st.tabs(["🔍  Detect", "📋  History"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – DETECT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_detect:

    st.markdown("### Upload a Vehicle Image")
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_upload, col_run = st.columns([3, 1])
        with col_upload:
            st.image(uploaded, caption="Uploaded Image", use_container_width=True)
        with col_run:
            st.markdown("<br><br>", unsafe_allow_html=True)
            run_btn = st.button("▶  Run ANPR", use_container_width=True)

        if run_btn:
            with st.spinner("Running ANPR pipeline…"):
                t0 = time.time()
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                try:
                    resp = requests.post(f"{BASE_URL}/detect", files=files, timeout=180)
                    elapsed = time.time() - t0
                    result  = resp.json()
                except Exception as e:
                    st.error(f"❌ Could not reach the API: {e}")
                    st.stop()

            # ── Status banner ──────────────────────────────────────────────
            status = result.get("status", "UNKNOWN")
            badge_class = {
                "SUCCESS":      "badge-success",
                "NO_VEHICLE":   "badge-warning",
                "NO_PLATE":     "badge-warning",
                "OCR_FAILED":   "badge-warning",
                "INVALID_IMAGE":"badge-error",
            }.get(status, "badge-info")

            st.markdown(f"""
            <div style='margin:1rem 0 0.5rem 0;'>
                <span class='badge {badge_class}'>{status}</span>
                <span style='color:#5050a0; font-size:0.82rem; margin-left:12px;'>
                    Processed in {elapsed:.1f}s
                </span>
            </div>""", unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # ── Two-column layout ──────────────────────────────────────────
            col_img, col_meta = st.columns([3, 2], gap="large")

            with col_img:
                st.markdown("**Annotated Image**")
                ann_path = result.get("annotated_image_path")
                if ann_path:
                    img_url = f"{BASE_URL}/{ann_path.replace(chr(92), '/')}"
                    try:
                        img_resp = requests.get(img_url, timeout=10)
                        st.image(img_resp.content, use_container_width=True)
                    except Exception:
                        st.caption("Could not load annotated image.")
                else:
                    st.info("No annotated image generated.")

            with col_meta:
                # ── Plate number display ───────────────────────────────────
                plate = result.get("plate") or {}
                plate_num = plate.get("number") or "—"
                st.markdown(
                    f"<div class='plate-box'>{plate_num}</div>",
                    unsafe_allow_html=True,
                )

                # ── Metric cards ───────────────────────────────────────────
                vehicle = result.get("vehicle") or {}
                v_type  = (vehicle.get("type") or "—").upper()
                v_conf  = vehicle.get("confidence", 0)
                p_conf  = plate.get("confidence", 0)
                o_conf  = plate.get("ocr_confidence", 0)

                def card(label, value, sub=""):
                    return f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value'>{value}</div>
                        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
                    </div>"""

                st.markdown(card("Vehicle Type",   v_type,                   f"Confidence: {v_conf:.1%}"), unsafe_allow_html=True)
                st.markdown(card("Plate Confidence", f"{p_conf:.1%}",        "YOLO Plate Detector"),       unsafe_allow_html=True)
                st.markdown(card("OCR Confidence",   f"{o_conf:.1%}",        "PP-OCRv5"),                  unsafe_allow_html=True)

                # UUID
                uid = result.get("uuid", "")
                if uid:
                    st.markdown(
                        f"<div style='margin-top:8px;color:#404070;font-size:0.72rem;'>UUID: {uid[:24]}…</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # ── Stage latency chart ────────────────────────────────────────
            timings = result.get("timings_ms", {})
            stage_labels = {
                "image_loading":     "Image Loading",
                "preprocessing":     "Preprocessing",
                "vehicle_detection": "Vehicle Detection",
                "vehicle_cropping":  "Vehicle Crop",
                "plate_detection":   "Plate Detection",
                "plate_cropping":    "Plate Crop",
                "image_enhancement": "Enhancement",
                "ocr":               "OCR (PP-OCRv5)",
                "post_processing":   "Post-Processing",
            }
            max_ms = max((timings.get(k, 0) for k in stage_labels), default=1)

            st.markdown("**Pipeline Stage Latency**")
            bars_html = ""
            for key, label in stage_labels.items():
                ms    = timings.get(key, 0)
                pct   = (ms / max_ms * 100) if max_ms else 0
                bars_html += f"""
                <div class='bar-row'>
                    <div class='bar-label'>{label}</div>
                    <div class='bar-track'>
                        <div class='bar-fill' style='width:{pct:.1f}%'></div>
                    </div>
                    <div class='bar-ms'>{ms:,.0f} ms</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

            total_ms = timings.get("total_inference", 0)
            st.markdown(
                f"<div style='color:#5050a0;font-size:0.82rem;margin-top:4px;'>"
                f"Total inference: <b style='color:#9090d0'>{total_ms:,.0f} ms</b></div>",
                unsafe_allow_html=True,
            )

            # ── Download buttons ───────────────────────────────────────────
            dl_col1, dl_col2 = st.columns(2)
            ann_path = result.get("annotated_image_path")
            if ann_path:
                try:
                    img_url  = f"{BASE_URL}/{ann_path.replace(chr(92), '/')}"
                    img_bytes = requests.get(img_url, timeout=10).content
                    with dl_col1:
                        st.download_button(
                            label="⬇️  Download Annotated Image",
                            data=img_bytes,
                            file_name=Path(ann_path).name,
                            mime="image/jpeg",
                            use_container_width=True,
                        )
                except Exception:
                    pass

            import json as _json
            json_bytes = _json.dumps(result, indent=2, default=str).encode()
            with dl_col2:
                st.download_button(
                    label="⬇️  Download JSON Result",
                    data=json_bytes,
                    file_name=f"anpr_{result.get('uuid', 'result')[:8]}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # ── JSON expander ──────────────────────────────────────────────
            with st.expander("📄 Full JSON Response"):
                st.json(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:

    h_col1, h_col2 = st.columns([2, 1])
    with h_col1:
        st.markdown("### Detection History")
    with h_col2:
        limit = st.selectbox("Records per page", [10, 25, 50], index=0, label_visibility="collapsed")

    refresh = st.button("🔄  Refresh")

    try:
        h_resp = requests.get(f"{BASE_URL}/history?limit={limit}", timeout=10)
        records = h_resp.json() if h_resp.status_code == 200 else []
        if isinstance(records, dict):
            records = records.get("data", records.get("items", []))
    except Exception:
        records = []
        st.warning("Could not load history. Is the API server running?")

    if not records:
        st.info("No detection records found yet. Run a detection to get started.")
    else:
        # ── Summary table ──────────────────────────────────────────────────
        table_rows = []
        for r in records:
            table_rows.append({
                "ID":           r.get("id", "—"),
                "UUID":         (r.get("uuid") or "")[:18] + "…",
                "Plate":        r.get("plate_number") or "—",
                "Vehicle":      (r.get("vehicle_type") or "—").capitalize(),
                "Processing ms":f"{r.get('processing_time_ms', 0):,.0f}",
                "Timestamp":    (r.get("timestamp") or "")[:19],
            })
        st.dataframe(table_rows, use_container_width=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 🔎 View / Delete a Record")

        id_options = [str(r.get("id")) for r in records if r.get("id")]
        selected_id = st.selectbox("Select record ID", id_options, label_visibility="visible")

        if selected_id:
            sel = next((r for r in records if str(r.get("id")) == selected_id), None)
            if sel:
                d_col1, d_col2 = st.columns([2, 2], gap="large")
                with d_col1:
                    st.markdown("**Original Image**")
                    orig = sel.get("image_path")
                    if orig:
                        try:
                            img_url = f"{BASE_URL}/{orig.replace(chr(92), '/')}"
                            r2 = requests.get(img_url, timeout=10)
                            st.image(r2.content, use_container_width=True)
                        except Exception:
                            st.caption("Image not accessible via static mount.")

                with d_col2:
                    st.markdown("**Annotated Image**")
                    ann = sel.get("annotated_image_path")
                    if ann:
                        try:
                            img_url = f"{BASE_URL}/{ann.replace(chr(92), '/')}"
                            r3 = requests.get(img_url, timeout=10)
                            st.image(r3.content, use_container_width=True)
                        except Exception:
                            st.caption("Annotated image not accessible.")

                # Detail cards
                st.markdown(f"""
                <div style='display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;'>
                    <div class='metric-card' style='flex:1;min-width:130px;'>
                        <div class='metric-label'>Plate</div>
                        <div class='metric-value' style='font-size:1.2rem;'>{sel.get('plate_number') or '—'}</div>
                    </div>
                    <div class='metric-card' style='flex:1;min-width:130px;'>
                        <div class='metric-label'>Vehicle</div>
                        <div class='metric-value' style='font-size:1.2rem;'>{(sel.get('vehicle_type') or '—').capitalize()}</div>
                    </div>
                    <div class='metric-card' style='flex:1;min-width:130px;'>
                        <div class='metric-label'>Processing Time</div>
                        <div class='metric-value' style='font-size:1.2rem;'>{sel.get('processing_time_ms', 0):,.0f} ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Delete button
                if st.button(f"🗑️  Delete Record #{selected_id}", type="secondary"):
                    del_resp = requests.delete(
                        f"{BASE_URL}/history/{selected_id}", timeout=10
                    )
                    if del_resp.status_code == 200:
                        st.success(f"Record #{selected_id} deleted successfully.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {del_resp.status_code} — {del_resp.text[:200]}")
