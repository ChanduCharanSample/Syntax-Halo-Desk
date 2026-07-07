import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from flask import Flask, jsonify, send_file
import threading

# Initialize Flask
app = Flask(__name__)

# Screen Settings
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False

# MediaPipe Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Global State Variables
class HoloState:
    def __init__(self):
        self.cursor_x, self.cursor_y = 0, 0
        self.prev_x, self.prev_y = 0, 0
        self.smooth_factor = 0.2
        self.current_gesture = "INITIALIZING"
        self.fps = 0
        self.is_paused = False
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.15
        self.prev_hand_y = 0
        self.click_cooldown = 0
        self.pinch_start_time = 0
        self.minimize_cooldown = 0

state = HoloState()

def get_finger_status(hand_landmarks):
    # MediaPipe landmarks
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    status = []
    for tip, pip in zip(tips, pips):
        # Result is True if finger is open (tip higher than pip)
        status.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    
    # Thumb: Check horizontal distance from index base for simplicity
    thumb_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x if hand_landmarks.landmark[17].x > hand_landmarks.landmark[5].x else hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    return thumb_open, status # (Thumb, [Index, Middle, Ring, Pinky])

def process_logic():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    prev_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        curr_time = time.time()
        state.fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                thumb_open, fingers = get_finger_status(hand_lms)
                index_open, middle_open, ring_open, pinky_open = fingers
                
                # Get Landmark Coordinates
                index_tip = hand_lms.landmark[8]
                thumb_tip = hand_lms.landmark[4]
                
                # --- GESTURE PRIORITY SYSTEM ---
                
                # 1. FIST (Pause)
                if not any(fingers) and not thumb_open:
                    state.current_gesture = "SYSTEM PAUSED"
                    state.is_paused = True
                
                # If Paused, only allow "Open Palm" to resume
                elif state.is_paused:
                    if all(fingers) and thumb_open:
                        state.is_paused = False
                        state.current_gesture = "SYSTEM ACTIVE"
                
                else:
                    # 2. OPEN PALM (Minimize All - Win+D)
                    if all(fingers) and thumb_open:
                        if time.time() - state.minimize_cooldown > 2:
                            pyautogui.hotkey('win', 'd')
                            state.minimize_cooldown = time.time()
                        state.current_gesture = "MINIMIZE ALL"

                    # 3. PINCH (Click / Double Click / Drag)
                    else:
                        dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                        if dist < 0.05:
                            if state.pinch_start_time == 0:
                                state.pinch_start_time = time.time()
                            
                            elapsed = time.time() - state.pinch_start_time
                            if elapsed > 0.7:
                                state.current_gesture = "DOUBLE CLICK"
                                pyautogui.doubleClick()
                                state.pinch_start_time = time.time() + 1 # reset
                            else:
                                state.current_gesture = "PINCH HOLD"
                                pyautogui.mouseDown()
                        else:
                            if state.current_gesture == "PINCH HOLD":
                                pyautogui.mouseUp()
                                if time.time() - state.pinch_start_time < 0.4:
                                    pyautogui.click()
                            state.pinch_start_time = 0

                            # 4. THREE FINGER SCROLL
                            if index_open and middle_open and ring_open and not pinky_open:
                                state.current_gesture = "SCROLLING"
                                curr_y = index_tip.y * 1000
                                if state.prev_hand_y != 0:
                                    delta_y = curr_y - state.prev_hand_y
                                    if abs(delta_y) > 15: # Threshold
                                        scroll_amt = -60 if delta_y > 0 else 60
                                        pyautogui.scroll(scroll_amt)
                                state.prev_hand_y = curr_y
                            
                            # 5. TWO FINGER CURSOR MOVE (Relaxed Detection)
                            elif index_open and middle_open:
                                state.current_gesture = "CURSOR MOVE"
                                target_x = np.interp(index_tip.x, [0.1, 0.9], [0, SCREEN_WIDTH])
                                target_y = np.interp(index_tip.y, [0.1, 0.9], [0, SCREEN_HEIGHT])
                                
                                # Exponential Smoothing (LERP)
                                state.cursor_x = state.prev_x + (target_x - state.prev_x) * state.smooth_factor
                                state.cursor_y = state.prev_y + (target_y - state.prev_y) * state.smooth_factor
                                
                                pyautogui.moveTo(state.cursor_x, state.cursor_y)
                                state.prev_x, state.prev_y = state.cursor_x, state.cursor_y
                                state.prev_hand_y = 0 # reset scroll
                            
                            else:
                                state.current_gesture = "IDLE"
                                state.prev_hand_y = 0

        # Draw on frame for visual feedback
        cv2.putText(frame, f"Gesture: {state.current_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Syntax HoloDesk - AI Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/status')
def get_status():
    return jsonify({
        "gesture": state.current_gesture,
        "fps": state.fps,
        "paused": state.is_paused
    })

if __name__ == '__main__':
    # Start tracking in a background thread
    t = threading.Thread(target=process_logic)
    t.daemon = True
    t.start()
    app.run(port=5000, debug=False)