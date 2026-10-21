import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Face Detection | YOLOv8",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #f7f9fc;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        color: #142c3c;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
    }

    .metric-value {
        color: #142c3c;
        font-size: 1.8rem;
        font-weight: 700;
    }

    div[data-testid="stImage"] img {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">Face Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered application for detecting faces in images using YOLOv8.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Minimum confidence required to display a detected face."
    )

    st.info(
        "Upload an image to detect faces. "
        "The model will highlight detected faces and display confidence scores."
    )

    st.caption("Model: Custom YOLOv8")
    st.caption("Task: Face Detection")

# --------------------------------------------------
# Image Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is None:
    st.markdown("### Get Started")
    st.write(
        "Upload an image using the box above to start face detection."
    )

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Loading model and detecting faces..."):
            model = load_model()
            start_time = time.perf_counter()

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            elapsed_time = time.perf_counter() - start_time

        result = results[0]

        # YOLO plot() returns a BGR image; convert it to RGB.
        detected_image = Image.fromarray(
            result.plot()[..., ::-1]
        )

        face_count = len(result.boxes)

        # ------------------------------------------
        # Summary Metrics
        # ------------------------------------------
        st.subheader("Detection Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Faces Detected", face_count)
        col2.metric("Processing Time", f"{elapsed_time:.2f} sec")
        col3.metric(
            "Image Resolution",
            f"{image.width} × {image.height}"
        )

        st.divider()

        # ------------------------------------------
        # Image Comparison
        # ------------------------------------------
        st.subheader("Detection Results")

        original_col, result_col = st.columns(2)

        with original_col:
            st.markdown("#### Original Image")
            st.image(image, use_container_width=True)

        with result_col:
            st.markdown("#### Detected Faces")
            st.image(detected_image, use_container_width=True)

        # ------------------------------------------
        # Confidence Details
        # ------------------------------------------
        if face_count > 0:
            st.subheader("Detection Details")

            confidence_scores = (
                result.boxes.conf.cpu().tolist()
            )

            details = pd.DataFrame({
                "Face": [
                    f"Face {i + 1}"
                    for i in range(face_count)
                ],
                "Confidence": [
                    f"{score * 100:.2f}%"
                    for score in confidence_scores
                ]
            })

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning(
                "No faces were detected. "
                "Try another image or lower the confidence threshold."
            )

        # ------------------------------------------
        # Download Result
        # ------------------------------------------
        st.divider()

        image_bytes = None
        import io

        image_buffer = io.BytesIO()
        detected_image.save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()

        st.download_button(
            label="Download Detection Result",
            data=image_bytes,
            file_name="face_detection_result.png",
            mime="image/png"
        )

    except Exception as error:
        st.error(
            "An error occurred while processing the image. "
            "Please check that the model file is valid and try again."
        )
        st.exception(error)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Face Detection System | Powered by YOLOv8 and Streamlit"
)
