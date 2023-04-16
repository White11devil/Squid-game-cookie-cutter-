import cv2
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector

# Load images
start_screen = cv2.imread("frames\img1.jpeg")
cookie_frame = cv2.imread("img\sqr(2).png")
lose_screen = cv2.imread("frames\img2.png")
win_screen = cv2.imread("frames\img3.png")
brand_logo = cv2.imread("img\mlsa.png")

# Initialize hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
hand_detector = HandDetector(maxHands=1, detectionCon=0.5)

# Initialize face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Generate cookie cutters
cookie_cutters = []
cookie_size = (100, 100)

for i in range(5):
    # Create blank image
    cookie_cutter = np.zeros((cookie_size[0], cookie_size[1], 4), dtype=np.uint8)
    
    # Draw cookie shape
    cv2.circle(cookie_cutter, (cookie_size[0]//2, cookie_size[1]//2), cookie_size[0]//2, (255, 255, 255, 255), thickness=-1)
    
    # Draw square shape
    if i == 1:
        square_size = int(cookie_size[0] * 0.7)
        square_offset = (cookie_size[0] - square_size) // 2
        cv2.rectangle(cookie_cutter, (square_offset, square_offset), (square_offset+square_size, square_offset+square_size), (0, 0, 0, 0), thickness=-1)
    
    # Append to list
    cookie_cutters.append(cookie_cutter)
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
# Loop through images
for i, image in enumerate([start_screen, cookie_frame, lose_screen, win_screen, brand_logo]):
    if i == 1: # Detect hand in cookie frame
        hand_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(hand_image)
        hand_landmarks = hand_detector.findHands(image)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_hands.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            draw_cookie_cutter(image, hand_landmarks)
    
    elif i == 0 or i == 3 or i == 4: # Detect face in start, win and brand logo images
        face_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(face_image)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * image.shape[1]), int(bbox.ymin * image.shape[0]), int(bbox.width * image.shape[1]), int(bbox.height * image.shape[0])
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Squid Game", image)
    cv2.waitKey(0)

# Release resources
cv2.destroyAllWindows()