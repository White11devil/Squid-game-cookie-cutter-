import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

# Load images
start_screen = cv2.imread("frames\img1.jpeg")
cookie_frame = cv2.imread("img\sqr(2).png")
lose_screen = cv2.imread("frames\img2.png")
win_screen = cv2.imread("frames\img3.png")
brand_logo = cv2.imread("img\mlsa.png")

# Create a window
cv2.namedWindow("Cookie Cutter Game")

# Initialize hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Capture video
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read video")
        break

    # Detect hand
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    # Display start screen if Q is not pressed
    if lmList is None or len(lmList) == 0:
        cv2.imshow("Cookie Cutter Game", start_screen)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Get index finger tip coordinates
        x, y = lmList[8][1], lmList[8][2]

        # Display cookie frame
        cv2.imshow("Cookie Cutter Game", cookie_frame)

        # Check if index finger is inside the square region
        if 250 < x < 450 and 150 < y < 350:
            # Display win screen
            cv2.imshow("Cookie Cutter Game", win_screen)
            if cv2.waitKey(3000) & 0xFF == ord("q"):
                break
        else:
            # Display lose screen
            cv2.imshow("Cookie Cutter Game", lose_screen)
            if cv2.waitKey(3000) & 0xFF == ord("q"):
                break

# Release the video capture and destroy window
cap.release()
cv2.destroyAllWindows()