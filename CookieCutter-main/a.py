import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
# from cvzone.FaceDetectionModule import FaceDetector

start_screen = cv2.imread("frames\img1.jpeg")
cookie_frame = cv2.imread("img\sqr(2).png")
lose_screen = cv2.imread("frames\img2.png")
win_screen = cv2.imread("frames\img3.png")
brand_logo = cv2.imread("img\mlsa.png")

cookie_cutters = []
cookie_size = (100, 100)

cookie_cutter = cv2.imread("img\sqr(2).png")
cookie_cutter = cv2.resize(cookie_cutter, cookie_size)
cookie_cutters.append(cookie_cutter)
detector = HandDetector(maxHands=1)
# face_detector = FaceDetector(maxFaces=1)
def detect_landmarks(image):
    landmarks = detector.findHands(image)
    # face = face_detector.findFaces(image)[0]
    if landmarks:
        landmarks = landmarks[0]["lmList"]
    # return landmarks, face
def is_hand_inside_face(hand_landmarks, face_box):
    hand_x, hand_y = hand_landmarks[8][1], hand_landmarks[8][2]
    face_x, face_y, face_w, face_h = face_box
    return (face_x <= hand_x <= face_x + face_w) and (face_y <= hand_y <= face_y + face_h)
def get_cookie_cutter(hand_landmarks):
    if hand_landmarks[8][2] < hand_landmarks[6][2]:  # Check if index finger is extended
        return cookie_cutters[0]
    else:
        return None
def draw_cookie_cutter(image, hand_landmarks):
    cookie_cutter = get_cookie_cutter(hand_landmarks)
    if cookie_cutter:
        cookie_cutter_x = hand_landmarks[8][1] - cookie_size[0] // 2
        cookie_cutter_y = hand_landmarks[8][2] - cookie_size[1] // 2
        image[cookie_cutter_y:cookie_cutter_y + cookie_size[1], cookie_cutter_x:cookie_cutter_x + cookie_size[0]] = cookie_cutter
import os

# Initialize hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
hand_detector = HandDetector(maxHands=1, detectionCon=0.5)

# Load cookie cutters
cookie_size = (150, 150)
cookie_cutters = [cv2.imread(os.path.join("cookie_cutters", f"cookie{i}.png"), cv2.IMREAD_UNCHANGED)
                  for i in range(1, 6)]

# Load and process images
image1 = cv2.imread("frames\img1.jpeg")
cv2.imshow("Squid Game", image1)
cv2.waitKey(0)

image2 = cv2.imread("img\sqr(2).png")
hand_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
hand_results2 = hands.process(hand_image2)
hand_landmarks2 = hand_detector.findHands(image2)
if hand_results2.multi_hand_landmarks:
    hand_landmarks2 = hand_results2.multi_hand_landmarks[0]
    mp_hands.draw_landmarks(image2, hand_landmarks2, mp_hands.HAND_CONNECTIONS)
    draw_cookie_cutter(image2, hand_landmarks2)
cv2.imshow("Squid Game", image2)
cv2.waitKey(0)

image3 = cv2.imread("frames\img2.png")
cv2.imshow("Squid Game", image3)
cv2.waitKey(0)

image4 = cv2.imread("frames\img3.png")
cv2.imshow("Squid Game", image4)
cv2.waitKey(0)

image5 = cv2.imread("img\mlsa.png")
cv2.imshow("Squid Game", image5)
cv2.waitKey(0)

# Release resources
cv2.destroyAllWindows()