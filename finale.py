import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to detect open palm
def is_open_palm(hand_landmarks):
    finger_tips_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcp_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    open_fingers = 0
    for tip_id, mcp_id in zip(finger_tips_ids, finger_mcp_ids):
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[mcp_id].y:
            open_fingers += 1

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    if open_fingers == 4 and thumb_tip.x < thumb_mcp.x:
        return True
    return False

# Swipe functions if you want to add forward/back later
def is_swipe_left(prev_landmarks, curr_landmarks):
    thumb_prev = prev_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_curr = curr_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    return thumb_curr.x < thumb_prev.x - 0.1

def is_swipe_right(prev_landmarks, curr_landmarks):
    thumb_prev = prev_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_curr = curr_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    return thumb_curr.x > thumb_prev.x + 0.1

# Focus on browser window (Chrome, Edge, Firefox etc.)
def focus_browser_window(window_title_contains):
    windows = gw.getWindowsWithTitle(window_title_contains)
    if windows:
        windows[0].activate()

# Main
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_landmarks = None
gesture_cooldown = 1.5  # seconds
last_gesture_time = time.time()

# Browser Window Title - Adjust if needed (Example: "YouTube" or "Chrome")
browser_window_name = "YouTube"  # Or "Google Chrome", "Microsoft Edge", etc.

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_time = time.time()
                if current_time - last_gesture_time > gesture_cooldown:
                    if is_open_palm(hand_landmarks):
                        print("Open Palm - Toggle Play/Pause on YouTube")
                        focus_browser_window(browser_window_name)
                        pyautogui.press('k')  # or 'space', but 'k' is more reliable on YouTube
                        last_gesture_time = current_time
                    elif prev_landmarks and is_swipe_left(prev_landmarks, hand_landmarks):
                        print("Swipe Left - Rewind 5 sec on YouTube")
                        focus_browser_window(browser_window_name)
                        pyautogui.press('left')
                        last_gesture_time = current_time
                    elif prev_landmarks and is_swipe_right(prev_landmarks, hand_landmarks):
                        print("Swipe Right - Forward 5 sec on YouTube")
                        focus_browser_window(browser_window_name)
                        pyautogui.press('right')
                        last_gesture_time = current_time

                prev_landmarks = hand_landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_resized = cv2.resize(frame, (640, 360))
        cv2.imshow('Hand Gesture Controller for YouTube', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
