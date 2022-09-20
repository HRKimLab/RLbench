import pickle

import cv2
import numpy as np

RESULT_FILE_NAME = "frames"
VIDEO_PATH = "../custom_envs/track/VR_standard.mp4"
RESIZED_WIDTH = 210
RESIZED_HEIGHT = 160

capture = cv2.VideoCapture(VIDEO_PATH)

frames = []
i = 0

prev = None
while True:
    ret, frame = capture.read()
    if not ret:
        break
    prev = frame

    resized_frame = cv2.resize(frame, (210, 160), interpolation=cv2.INTER_CUBIC)
    frames.append(resized_frame)

    cv2.imshow("resized_frame", resized_frame)

    cv2.imwrite(f"./timing/timing_{i}.png", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    i += 1
print(i)

capture.release()
cv2.destroyAllWindows()

frames = np.stack(frames, axis=0)
with open(f"{RESULT_FILE_NAME}.pkl", "wb") as f:
    pickle.dump(frames, f)
