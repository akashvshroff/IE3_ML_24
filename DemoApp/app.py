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
    "Translate from American Sign Language to English in real-time by leveraging LSTM (Long Short Term Memory) Neural Networks and Large Language Models."
)
st.write(
    "*Disclaimer: captured videos are only stored temporarily and wiped after inference.*"
)

use_webcam = st.button("Record Sign Language")
stframe = st.empty()

if use_webcam:
    vid = cv2.VideoCapture(0)

    st.button("Stop Recording")

    # Values
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc("V", "P", "0", "9")
    out = cv2.VideoWriter("output1.webm", codec, fps, (width, height))

    # Number of frames to process at once:
    frame_count = 50
    frame_ctr = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        hands_mat = np.empty((frame_count + 1, 2), dtype=object)

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
                if results.right_hand_landmarks:
                    right_hand = []
                    for x in results.right_hand_landmarks.landmark:
                        right_hand.append((x.x, x.y, x.z))
                    hands_mat[frame_ctr][0] = np.array(right_hand)

                if results.left_hand_landmarks:
                    left_hand = []
                    for x in results.left_hand_landmarks.landmark:
                        left_hand.append((x.x, x.y, x.z))
                    hands_mat[frame_ctr][1] = np.array(left_hand)

                frame_ctr += 1

                if frame_ctr == frame_count:
                    np.save(f"data.npy", hands_mat)
                    # PROCESS DATA
                    hands_mat = np.empty((frame_count + 1, 2), dtype=object)
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

            stframe.image(cv2.flip(image, 1), use_column_width=True)

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    st.success("Video is Processed")
