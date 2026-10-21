import io
import time

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="FaceAI | Face Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    :root {
        --bg: #080d1b;
        --surface: #111a2e;
        --border: #293651;
        --primary: #42d6bd;
        --purple: #9b8afb;
        --text: #f1f5f9;
        --muted: #9ba9c4;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, #172440 0%, transparent 42%),
            radial-gradient(ellipse at 90% 20%, #211c42 0%, transparent 35%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid var(--border);
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        text-align: center;
        padding: 30px 10px 25px;
    }


    .hero h1 {
        font-size: clamp(2.5rem, 5vw, 3.7rem);
        font-weight: 800;
        letter-spacing: -1.5px;
        color: var(--text);
        margin: 0;
    }

    .hero h1 span {
        color: var(--primary);
    }

    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: 12px;
    }

    .section-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 700;
        margin: 22px 0 15px;
    }

    .section-heading svg {
        width: 23px;
        height: 23px;
        stroke: var(--primary);
        fill: none;
        flex-shrink: 0;
    }

    .image-heading {
        color: var(--text);
        font-size: 1rem;
        font-weight: 650;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #151f35, #10182a);
        border: 1px solid var(--border);
        padding: 18px;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary);
    }

    div[data-testid="stImage"] img {
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #0e1628;
        padding: 7px;
        border: 1px solid var(--border);
        border-radius: 13px;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        border-radius: 9px;
        padding: 10px 18px;
    }

    .stTabs [aria-selected="true"] {
        background: #193448;
        color: var(--primary);
    }

    .stButton button,
    .stDownloadButton button {
        background: linear-gradient(100deg, #42d6bd, #8d80f9);
        color: #08121e;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        color: white;
        border: 1px solid var(--primary);
    }

    [data-testid="stFileUploader"] {
        background: #111a2e;
        border: 1px dashed #405271;
        border-radius: 14px;
        padding: 12px;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        color: #8290aa;
        text-align: center;
        padding: 20px 0 5px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SVG ICONS ====================

ICONS = {
    "target": """
        <circle cx="12" cy="12" r="9"/>
        <circle cx="12" cy="12" r="5"/>
        <circle cx="12" cy="12" r="1"/>
    """,

    "image": """
        <rect x="3" y="3" width="18" height="18" rx="3"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <path d="m21 15-5-5L5 21"/>
    """,

    "settings": """
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-1.5 2.6-.2-.1a1.7 1.7 0 0 0-1.8.1 1.7 1.7 0 0 0-.9 1.5v.2h-3v-.2a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8-.1l-.2.1-1.5-2.6.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H6.5v-3h.2a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1 1.5-2.6.2.1a1.7 1.7 0 0 0 1.8-.1 1.7 1.7 0 0 0 .9-1.5v-.2h3v.2a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8.1l.2-.1 1.5 2.6-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.5 1h.2v3h-.2a1.7 1.7 0 0 0-1.4 1z"/>
    """,

    "scan": """
        <path d="M4 8V5a1 1 0 0 1 1-1h3"/>
        <path d="M16 4h3a1 1 0 0 1 1 1v3"/>
        <path d="M20 16v3a1 1 0 0 1-1 1h-3"/>
        <path d="M8 20H5a1 1 0 0 1-1-1v-3"/>
        <path d="M4 12h16"/>
    """,
}


def render_icon(name, size=22):
    paths = ICONS.get(name, ICONS["target"])

    return f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{size}" height="{size}"
         viewBox="0 0 24 24"
         fill="none"
         stroke="currentColor"
         stroke-width="1.7"
         stroke-linecap="round"
         stroke-linejoin="round"
         aria-hidden="true">
        {paths}
    </svg>
    """


def section_heading(icon_name, title):
    icon_svg = render_icon(icon_name)

    st.html(f"""
        <div class="section-heading">
            {icon_svg}
            <span>{title}</span>
        </div>
    """)


# ==================== LOAD YOLO MODEL ====================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


# ==================== HEADER ====================

# ==================== HEADER ====================

st.html("""
    <div class="hero">
        <h1>Face<span>AI</span></h1>
        <p>Intelligent Face Detection powered by YOLOv8</p>
    </div>
""")

st.divider()


# ==================== SIDEBAR ====================

with st.sidebar:
    section_heading("settings", "Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.40,
        step=0.05,
        help="Minimum confidence required for a detection."
    )

    st.caption(
        "Higher thresholds may reduce false detections, "
        "but can also cause missed faces."
    )

    st.divider()

    st.markdown("#### About FaceAI")

    st.write(
        "A computer vision application that detects faces "
        "in images using a custom-trained YOLOv8 model."
    )

    st.caption("Model: Custom YOLOv8")
    st.caption("Task: Face Detection")


# ==================== INPUT OPTIONS ====================

section_heading("scan", "Choose Your Input")

upload_tab, camera_tab = st.tabs([
    "Upload Image",
    "Camera Capture"
])

image_file = None

with upload_tab:
    st.markdown("#### Upload an image")

    st.write(
        "Select an image from your device to detect faces."
    )

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_file is not None:
        image_file = uploaded_file


with camera_tab:
    st.markdown("#### Capture an image")

    st.write(
        "Allow camera access in your browser and take a photo."
    )

    camera_file = st.camera_input(
        "Take a picture",
        key="camera_capture"
    )

    if camera_file is not None:
        image_file = camera_file


# ==================== FACE DETECTION ====================

if image_file is None:
    st.info(
        "Upload an image or capture a photo to get started."
    )

else:
    try:
        image = Image.open(image_file).convert("RGB")

        with st.spinner("Analyzing image..."):
            model = load_model()

            start_time = time.perf_counter()

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            elapsed_time = time.perf_counter() - start_time

        result = results[0]

        detected_image = Image.fromarray(
            result.plot()[..., ::-1]
        )

        face_count = len(result.boxes)

        # Detection Summary
        st.divider()
        section_heading("scan", "Detection Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Faces Detected", face_count)

        col2.metric(
            "Processing Time",
            f"{elapsed_time:.2f} sec"
        )

        col3.metric(
            "Image Resolution",
            f"{image.width} x {image.height}"
        )

        # Original and Detected Images
        st.divider()
        section_heading("image", "Detection Results")

        original_col, result_col = st.columns(2)

        with original_col:
            st.markdown(
                '<div class="image-heading">Original Image</div>',
                unsafe_allow_html=True
            )

            st.image(image, use_container_width=True)

        with result_col:
            st.markdown(
                '<div class="image-heading">Detected Faces</div>',
                unsafe_allow_html=True
            )

            st.image(
                detected_image,
                use_container_width=True
            )

        # Confidence Details
        if face_count > 0:
            st.divider()
            section_heading("target", "Detection Details")

            scores = result.boxes.conf.cpu().tolist()

            details = pd.DataFrame({
                "Detection": [
                    f"Face {i + 1}"
                    for i in range(face_count)
                ],
                "Confidence Score": [
                    f"{score * 100:.2f}%"
                    for score in scores
                ]
            })

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning(
                "No faces detected. Try another image "
                "or lower the confidence threshold."
            )

        # Download Result
        st.divider()

        buffer = io.BytesIO()
        detected_image.save(buffer, format="PNG")

        st.download_button(
            label="Download Detection Result",
            data=buffer.getvalue(),
            file_name="face_detection_result.png",
            mime="image/png",
            icon=":material/download:"
        )

    except Exception as error:
        st.error(
            "An error occurred while processing the image. "
            "Please check the model file and try again."
        )
        st.exception(error)


# ==================== FOOTER ====================

st.divider()

st.html("""
    <div class="footer">
        FaceAI | Computer Vision | YOLOv8
    </div>
""")
