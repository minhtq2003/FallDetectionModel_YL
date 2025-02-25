import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Hoặc thử cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera hu roi!!!!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
