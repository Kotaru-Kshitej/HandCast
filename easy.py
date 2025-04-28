import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Detect open palm
def is_open_palm(hand_landmarks):
    fingers = []
    for tip_id, mcp_id in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    ]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[mcp_id].y:
            fingers.append(1)
        else:
            fingers.append(0)

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    if sum(fingers) == 4 and thumb_tip.x < thumb_mcp.x:
        return True
    return False

# Detect Victory Sign (2 fingers)
def is_victory(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    return (index_tip.y < index_mcp.y and
            middle_tip.y < middle_mcp.y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y)

# Detect One Finger Up (Index)
def is_one_finger_up(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    return (index_tip.y < index_mcp.y and
            middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y)

# Focus browser window
def focus_browser_window(window_title_contains):
    windows = gw.getWindowsWithTitle(window_title_contains)
    if windows:
        windows[0].activate()

# Main Program
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

gesture_cooldown = 1.5  # seconds
last_gesture_time = time.time()

browser_window_name = "YouTube"  # Change if needed

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
                        print("Open Palm - Play/Pause")
                        focus_browser_window(browser_window_name)
                        pyautogui.press('k')
                        last_gesture_time = current_time
                    elif is_victory(hand_landmarks):
                        print("Victory Sign - Forward 5 sec")
                        focus_browser_window(browser_window_name)
                        pyautogui.press('right')
                        last_gesture_time = current_time
                    elif is_one_finger_up(hand_landmarks):
                        print("One Finger Up - Rewind 5 sec")
                        focus_browser_window(browser_window_name)
                        pyautogui.press('left')
                        last_gesture_time = current_time

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_resized = cv2.resize(frame, (640, 360))
        cv2.imshow('Hand Gesture Controller for YouTube', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
