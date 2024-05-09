# import numpy as np
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic
# mp_hands = mp.solutions.hands

# vid = cv2.VideoCapture(0)

# # Number of frames to process at once:
# frame_count = 30
# frame_ctr = 0

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

#     hands_mat = np.empty((frame_count, 2), dtype=object)

#     while vid.isOpened():
        
#         ret, image = vid.read()

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         if not ret:
#             break

#         results = holistic.process(image)

#         if results.right_hand_landmarks or results.left_hand_landmarks or frame_ctr != 0:
#             if results.right_hand_landmarks:
#                 right_hand = []
#                 for x in results.right_hand_landmarks.landmark:
#                     right_hand.append((x.x, x.y, x.z))
#                 hands_mat[frame_ctr][0] = np.array(right_hand)
            
#             if results.left_hand_landmarks:
#                 left_hand = []
#                 for x in results.left_hand_landmarks.landmark:
#                     left_hand.append((x.x, x.y, x.z))
#                 hands_mat[frame_ctr][1] = np.array(left_hand)

#             frame_ctr += 1

#             if frame_ctr == 30:
#                 np.save(f'data.npy', hands_mat)
#                 # PROCESS DATA
#                 hands_mat = np.empty((frame_count, 2), dtype=object)
#                 frame_ctr = 0

#         mp_drawing.draw_landmarks(
#             image,
#             results.right_hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
        
#         mp_drawing.draw_landmarks(
#             image,
#             results.left_hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
        
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
#         if cv2.waitKey(1) == ord('q'):
#             break

# vid.release()

# import pandas as pd
# import numpy as np

# ## convert your array into a dataframe
# df = pd.DataFrame(np.load('data.npy', allow_pickle=True))

# ## save to xlsx file

# print(np.load('data.npy', allow_pickle=True))