import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import api

operation_tips = """
        <div style="font-size: 1.1em; line-height: 1.6em;">
        <b>Operation Tips:</b> <br>Drag a rectangle on the <strong>Canvas</strong> to define the area of interest.<br>
        To clear the <strong>Canvas</strong>, please click on the 
        <img src='data:image/png;base64,
        iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAhQAAAIUB4uz/wQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAHUSURBVEiJtda9ahRRFAfw3xHFJCabFEkawcYiYKMWVmohBHwFOwsfQLCxthOENHZqKxZG8DHSxcJC0Qcw7AqSNSoGci3mznp3nP0K8cBlZs7X/5z//RopJeMGlrGL1Bi7WJ4Uf9pk2cQVPMDPrJvHVra9GRc8BBARS7iKU4X6cn5+wvf8vljbIuJr4XuEdyml/YGmoGIdX/xLxayji/U2iu5iJXfwa1zbY2QOOznXE4YpuoE93D5m8lr2cLMN4BWu4TF+4HDGxIEOeng90DaW5BkVj/fyd6dhP4u5hq6Tnxdz7GZpL1eLlNIhvmEtq7Yj4mHhsoWng5Ir23b+rGO6Zc62fdAtnFewUdjOqyaylo3sMxJgqIMsvcJ5X8XrKFlCvwHQmwRQdjArwH5K6fc0AKvHAFjVoGcUwCwUdbKPHNNrOpw0RVN10MVCRCz8T4A6oI/FiIiTBKh5XFN1EDnRkETEnGrnH2sOSoC60rbqoR8R8zg3bQe102oB0DYPAwB/l/VkgJTSgeo+KDtoA6h1fSN2Me1nUV3JOj7iOT5k/csi5jNe4D2uj+og8lE7rIx4hjt4hIMRRZRyX7UYLqWUjoYsI35VLuCt6tKZ5h7ewa22XH8AgjMTispa6ucAAAAASUVORK5CYII=' 
        class='CanvasToolbar_enabled__2bOtL CanvasToolbar_invertx__2gc2O' 
        height='16px' 
        width='16px'>icon below the <strong>Canvas</strong>.
        </div>
        """
footer = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:before {
                content:'MedSam | WangLab'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """

st.set_page_config(page_title="MedSam", layout='wide')
st.markdown("""
    <style>
    .st-emotion-cache-z5fcl4 {
        padding-top: 0rem; 
    }
    </style>
    """, unsafe_allow_html=True)
st.title("Welcome to MedSam")
st.markdown(footer, unsafe_allow_html=True)
# Set sidebar title
st.sidebar.header("Upload Image")



# Set up file uploader
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.markdown(operation_tips, unsafe_allow_html=True)
    # Convert the uploaded file to an image
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    image_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_pil = Image.open(io.BytesIO(bytes_data))

    # Get the width and height of the image
    width, height = image_pil.size
    # Set canvas properties
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Set fill color to transparent
        stroke_width=2,                 # Line width
        stroke_color="#FFA500",         # Set line color to orange (the HEX code for orange is #FFA500)
        background_image=image_pil,     # Background image
        drawing_mode="rect",            # Drawing mode
        width=width,                    # Set canvas width to image width
        height=height,                  # Set canvas height to image height
        display_toolbar=True,
        initial_drawing={'json_data':{'objects':[]}},
    )


    # If there is drawing on the canvas, then process coordinates
    if canvas_result.json_data is not None:
        shapes = canvas_result.json_data["objects"]  # Get all drawn shapes
        if shapes:  # If there are drawn rectangles
            rect_coords = shapes[-1]["left"], shapes[-1]["top"], shapes[-1]["left"] + shapes[-1]["width"], shapes[-1]["top"] + shapes[-1]["height"]
            output_image_bytesio = api.blend_segment_with_original(image_cv2, rect_coords)
            st.image(output_image_bytesio)