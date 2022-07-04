import pickle

import cv2
import numpy as np

VIDEO_PATH = "VR_standard.mp4"
RESIZED_WIDTH = 210
RESIZED_HEIGHT = 160

capture = cv2.VideoCapture(VIDEO_PATH)

frames = []
while True:
    ret, frame = capture.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (210, 160), interpolation=cv2.INTER_CUBIC)
    frames.append(resized_frame)
    cv2.imshow("resized_frame", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

frames = np.stack(frames, axis=0)
with open("frames.pkl", "wb") as f:
    pickle.dump(frames, f)
