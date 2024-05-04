import cv2
import tempfile
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

def main():
    # Title
    st.title('AiSL')

    # Sidebar title
    st.sidebar.title('Use your webcam or upload a video!')
    st.sidebar.subheader('Parameters')

    # Creating a button for webcam
    use_webcam = st.sidebar.button('Use Webcam')

   
    st.markdown('## Output')
    stframe = st.empty()

    # File uploader
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])

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
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    # holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_count = 0 
    new_word = True

    landmarks = []

    while vid.isOpened():
        ret, image = vid.read()

        new_word = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not ret:
            break

        results = hand.process(image)

        frame_count += 1

        if frame_count == 20:
            new_word = True
            frame_count = 0

        if new_word:
            landmarks = []
        
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
               
                print(hand_landmarks)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks.append(hand_landmarks)

        stframe.image(image, use_column_width=True)

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    st.success('Video is Processed')

if __name__ == '__main__':
    main()
