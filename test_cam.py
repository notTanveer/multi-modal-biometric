import cv2

for i in [0, 1]:
    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print(f"Camera {i} not available")
        continue

    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"CAMERA INDEX = {i}", frame)
        print(f"Showing Camera {i} — press any key for next")
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()