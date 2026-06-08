import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize mediapipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Finger tip landmarks as per mediapipe documentation
FINGER_TIPS = [4, 8, 12, 16, 20]

# Key mapping based on finger states (index, middle, ring, pinky, thumb) 
# We'll use a dictionary with tuple keys representing finger states (1=up, 0=down)
# Example: (thumb, index, middle, ring, pinky)
# Mapping here is just a demo — you can customize for your own gestures

GESTURE_KEYS = {
    (1, 0, 0, 0, 0): 'a',
    (0, 1, 0, 0, 0): 'b',
    (0, 0, 1, 0, 0): 'c',
    (0, 0, 0, 1, 0): 'd',
    (0, 0, 0, 0, 1): 'e',
    (1, 1, 0, 0, 0): 'f',
    (1, 0, 1, 0, 0): 'g',
    (1, 0, 0, 1, 0): 'h',
    (1, 0, 0, 0, 1): 'i',
    (0, 1, 1, 0, 0): 'j',
    (0, 1, 0, 1, 0): 'k',
    (0, 1, 0, 0, 1): 'l',
    (0, 0, 1, 1, 0): 'm',
    (0, 0, 1, 0, 1): 'n',
    (0, 0, 0, 1, 1): 'o',
    (1, 1, 1, 0, 0): 'p',
    (1, 1, 0, 1, 0): 'q',
    (1, 1, 0, 0, 1): 'r',
    (1, 0, 1, 1, 0): 's',
    (1, 0, 1, 0, 1): 't',
    (1, 0, 0, 1, 1): 'u',
    (0, 1, 1, 1, 0): 'v',
    (0, 1, 1, 0, 1): 'w',
    (0, 1, 0, 1, 1): 'x',
    (0, 0, 1, 1, 1): 'y',
    (1, 1, 1, 1, 0): 'z',
    (0, 0, 0, 0, 0): 'backspace', 
    (1, 1, 1, 1, 1): 'enter',  
}

# Cooldown between key presses to avoid repeated inputs
COOLDOWN = 1.0  # seconds

def fingers_status(hand_landmarks):
    """
    Returns a tuple with finger states: 1 if finger is up, 0 if down
    Order: thumb, index, middle, ring, pinky
    """

    finger_states = []

    # Thumb: Compare tip and IP joint in x-axis for right hand (mirror for left hand)
    # Because thumb bends sideways
    # Using landmark 4 (tip) and 3 (IP joint)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # For other fingers, tip landmark y < pip landmark y means finger is up
    fingers_tips_ids = [8, 12, 16, 20]
    fingers_pip_ids = [6, 10, 14, 18]

    for tip_id, pip_id in zip(fingers_tips_ids, fingers_pip_ids):
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return tuple(finger_states)

def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        img_height, img_width = img.shape[:2]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_state = fingers_status(hand_landmarks)

                current_time = time.time()
                if current_time - prev_time > COOLDOWN:
                    if finger_state in GESTURE_KEYS:
                        key = GESTURE_KEYS[finger_state]
                        if key == 'space':
                            pyautogui.press('space')
                        elif key == 'enter':
                            pyautogui.press('enter')
                        else:
                            pyautogui.press(key)
                        prev_time = current_time
                        cv2.putText(img, f"Pressed: {key.upper()}", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    else:
                        cv2.putText(img, f"Gesture not mapped", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                # Also display finger state for debugging
                cv2.putText(img, f"Fingers: {finger_state}", (10, img_height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        else:
            cv2.putText(img, "Show your hand to control keyboard", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Hand Gesture Keyboard", img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()