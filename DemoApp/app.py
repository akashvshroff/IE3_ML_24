import numpy as np
import cv2
import tempfile
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands


def main():
    # Title
    st.title("AiSL")

    # Sidebar title
    st.sidebar.title("Use your webcam or upload a video!")
    st.sidebar.subheader("Parameters")

    # Creating a button for webcam
    use_webcam = st.sidebar.button("Use Webcam")

    st.markdown("## Output")
    stframe = st.empty()

    # File uploader
    video_file_buffer = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"]
    )

    # Temporary file name
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            st.warning("Please upload a video or select 'Use Webcam'")
            return
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    # Values
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc("V", "P", "0", "9")
    out = cv2.VideoWriter("output1.webm", codec, fps, (width, height))

    st.sidebar.text("Input Video")
    st.sidebar.video(tffile.name)

    # Number of frames to process at once:
    frame_count = 20
    frame_ctr = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        hands_mat = np.empty((frame_count, 2), dtype=object)

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

                if frame_ctr == 30:
                    np.save(f"data.npy", hands_mat)
                    # PROCESS DATA
                    hands_mat = np.empty((frame_count, 2), dtype=object)
                    frame_ctr = 0

            # mp_drawing.draw_landmarks(
            #     image,
            #     results.right_hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style())

            # mp_drawing.draw_landmarks(
            #     image,
            #     results.left_hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style())

            stframe.image(cv2.flip(image, 1), use_column_width=True)

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    st.success("Video is Processed")


if __name__ == "__main__":
    main()
