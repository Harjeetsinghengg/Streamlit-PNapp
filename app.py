import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.title("🔍 Template Matching (Multi-Scale)")

# --- User Login ---
username = st.text_input("Enter username:", key="user_id")

if username:
    user_folder = f"data/{username}"
    good_folder = f"{user_folder}/good_images"
    os.makedirs(good_folder, exist_ok=True)

    st.success(f"Logged in as: {username}")

    # --- Upload or Capture GOOD Template ---
    st.subheader("Upload/Capture GOOD Template Image")
    good_img_file = st.file_uploader("Upload GOOD Image", type=["jpg", "png"], key="good_upload")
    good_cam_img = st.camera_input("Or Capture GOOD Image", key="good_cam")

    if good_cam_img:
        good_img_file = good_cam_img  # override uploaded image

    # --- Upload or Capture TEST Image ---
    st.subheader("Upload/Capture TEST Image")
    test_img_file = st.file_uploader("Upload TEST Image", type=["jpg", "png"], key="test_upload")
    test_cam_img = st.camera_input("Or Capture TEST Image", key="test_cam")

    if test_cam_img:
        test_img_file = test_cam_img

    threshold = st.slider("Match Threshold", 0.1, 1.0, 0.6)

    if st.button("RUN TEMPLATE MATCHING"):
        if not good_img_file or not test_img_file:
            st.warning("Please upload/capture both template and test image.")
        else:
            # Load images in OpenCV format
            good_img = Image.open(good_img_file).convert("RGB")
            good_np = cv2.cvtColor(np.array(good_img), cv2.COLOR_RGB2GRAY)

            test_img = Image.open(test_img_file).convert("RGB")
            test_np = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2GRAY)

            # Multi-scale Match
            best_val = -1
            best_loc = None
            best_scale = 1.0
            test_img_color = cv2.cvtColor(test_np, cv2.COLOR_GRAY2BGR)

            for scale in np.linspace(0.5, 1.5, 20):  # Multi-scale checking
                scaled_template = cv2.resize(good_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
                if scaled_template.shape[0] > test_np.shape[0] or scaled_template.shape[1] > test_np.shape[1]:
                    continue

                result = cv2.matchTemplate(test_np, scaled_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale

            h, w = int(good_np.shape[0] * best_scale), int(good_np.shape[1] * best_scale)

            if best_val >= threshold:
                cv2.rectangle(test_img_color, best_loc, (best_loc[0] + w, best_loc[1] + h), (0,255,0), 3)
                cv2.putText(test_img_color, "PASS", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                verdict = "PASS"
            else:
                cv2.putText(test_img_color, "FAIL", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                verdict = "FAIL"

            st.subheader(f"Result: {verdict}  | Match Score: {best_val:.2f}")
            st.image(test_img_color, channels="BGR")

            # Save the Good Template for the user
            save_path = os.path.join(good_folder, f"template_{good_img_file.name}")
            with open(save_path, "wb") as f:
                f.write(good_img_file.getbuffer())
            st.info(f"Template saved under user folder: {username}")
