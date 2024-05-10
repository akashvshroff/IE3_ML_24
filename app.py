# streamlit imports
import streamlit as st
from streamlit_lottie import st_lottie

# utils and ml
from utils import *
from ml import *

# media pipe and image imports
import numpy as np
import cv2
import tempfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands


lottie_file = load_lottieurl()  # animation url
st_lottie(lottie_file, height=175, quality="medium")
st.title("AiSL")

st.write(
    "Translate from American Sign Language to English in real-time by leveraging LSTM (Long Short Term Memory) Neural Networks."
)

st.write(
    "To get started, hit the button below and begin signing your phrase or words. You will see the translation in real time. Once you are done signing, move your hands out of the frame. To close the camera, hit the finish button below."
)

st.write(
    "*Disclaimer: captured videos are only stored temporarily and wiped after inference.*"
)

use_webcam = st.button("Record Sign Language")
stframe_cam = st.empty()
translation = ""

if use_webcam:
    vid = cv2.VideoCapture(0)
    st.write("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        stframe_translation = st.empty()
    with col2:
        st.button("Finish", type="primary")
    st.write("---")

    # Values
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc("V", "P", "0", "9")
    out = cv2.VideoWriter("output1.webm", codec, fps, (width, height))

    # Number of frames to process at once:
    frame_count = 30
    frame_ctr = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        hands_mat = [[0] * 126 for _ in range(frame_count)]

        while vid.isOpened():

            ret, image = vid.read()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not ret:
                break

            results = holistic.process(image)

            if (
                results.right_hand_landmarks
                or results.left_hand_landmarks
                or frame_ctr != 0
            ):
                right_hand = [0] * 63
                if results.right_hand_landmarks:
                    right_hand = []
                    for x in results.right_hand_landmarks.landmark:
                        right_hand += [x.x, x.y, x.z]
                hands_mat[frame_ctr] = right_hand

                left_hand = [0] * 63
                if results.left_hand_landmarks:
                    left_hand = []
                    for x in results.left_hand_landmarks.landmark:
                        left_hand += [x.x, x.y, x.z]
                hands_mat[frame_ctr] += left_hand

                frame_ctr += 1

                if frame_ctr >= frame_count - 1:
                    label = infer(hands_mat)
                    translation += f"{label} "
                    stframe_translation.write(f'###  "{translation}"')
                    hands_mat = [[0] * 126 for _ in range(frame_count)]
                    frame_ctr = 0

            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            stframe_cam.image(cv2.flip(image, 1), use_column_width=True)

    vid.release()
    out.release()
    cv2.destroyAllWindows()
