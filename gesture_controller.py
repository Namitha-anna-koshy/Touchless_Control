import cv2 
import mediapipe as mp
import numpy as np
import time
from collections import deque
import pyautogui
import win32api
import win32con
from win32con import MOUSEEVENTF_WHEEL
import math
import pyperclip

class KalmanFilter:
    def __init__(self, process_noise=0.001, measurement_noise=0.1, error_cov=0.1):
        self.process_noise = process_noise        # Process noise
        self.measurement_noise = measurement_noise  # Measurement noise
        self.error_cov = error_cov                # Estimation error covariance
        self.state = np.zeros(4)                  # [x, y, vx, vy]
        self.P = np.eye(4) * self.error_cov       # Estimation error covariance matrix
        self.F = np.array([                       # State transition matrix
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([                       # Measurement matrix
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * self.process_noise   # Process noise covariance
        self.R = np.eye(2) * self.measurement_noise  # Measurement noise covariance
        self.initialized = False
        
    def predict(self):
        # State prediction
        self.state = self.F @ self.state
        # Error covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return x, y
        
    def update(self, measurement):
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return self.state[:2]
            
        # Measurement update
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2]  # Return x, y
        
    def filter(self, x, y):
        # Predict state
        self.predict()
        # Update with new measurement
        filtered_pos = self.update(np.array([x, y]))
        return int(filtered_pos[0]), int(filtered_pos[1])

class StrokeManager:
    def __init__(self, max_strokes=20, max_points_per_stroke=100):
        self.strokes = []
        self.current_stroke = []
        self.max_strokes = max_strokes
        self.max_points_per_stroke = max_points_per_stroke
        self.is_drawing = False
        
    def start_stroke(self, point):
        if self.is_drawing:
            return
            
        self.is_drawing = True
        self.current_stroke = [point]
        
    def add_point(self, point):
        if not self.is_drawing:
            return
            
        if len(self.current_stroke) < self.max_points_per_stroke:
            self.current_stroke.append(point)
        
    def end_stroke(self):
        if not self.is_drawing:
            return
            
        if len(self.current_stroke) > 1:  # Only add strokes with at least 2 points
            self.strokes.append(self.current_stroke[:])
            if len(self.strokes) > self.max_strokes:
                self.strokes.pop(0)  # Remove oldest stroke if limit reached
                
        self.current_stroke = []
        self.is_drawing = False
        
    def get_current_stroke(self):
        return self.current_stroke
        
    def get_all_strokes(self):
        return self.strokes
        
    def clear_strokes(self):
        self.strokes = []
        self.current_stroke = []
        self.is_drawing = False

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face = mp.solutions.face_detection  # Add face detection
        
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        
        # Initialize face detection
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.7)

        # Screen and window setup
        self.whiteboard_width = 1280
        self.whiteboard_height = 720
        
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Mouse control settings
        self.last_mouse_pos = (0, 0)
        self.last_actual_pos = (0, 0)

        # Drawing setup
        self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
        self.drawing = False
        self.draw_color = (0, 0, 0)
        self.draw_thickness = 2
        
        self.erasing = False
        self.eraser_size = 30
        self.eraser_color = (255, 255, 255)
        
        # State tracking
        self.prev_hand_landmarks = None
        self.prev_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.whiteboard_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.whiteboard_height)
        
        # Mouse state
        self.mouse_pressed = False
        self.last_click_time = 0
        self.click_cooldown = 0.3

        # Gesture states and toggles
        self.whiteboard_active = True
        self.mouse_control_active = True  # Virtual mouse toggle
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0
        
        # Toggle gesture state tracking
        self.toggle_wb_gesture_start = None
        self.toggle_mouse_gesture_start = None
        self.toggle_hold_duration = 1.0  # Duration to hold toggle gesture in seconds
        self.last_gesture = None
        
        # Scroll control settings
        self.scroll_active = False
        self.scroll_speed = 1.0
        self.scroll_velocity = 0
        self.smooth_scroll_amount = 0
        self.scroll_smoothing = 0.3
        self.last_scroll_y = None
        self.scroll_threshold = 0.02
        self.scroll_gesture_start_y = None
        
        # Virtual monitor settings - always active and bigger
        self.virtual_monitor_active = True  # Always enabled by default
        self.virtual_monitor_width = 600
        self.virtual_monitor_height = 450
        self.virtual_monitor_x = 120
        self.virtual_monitor_y = 90
        self.in_virtual_monitor = False
        
        # Initialize Kalman filter for cursor position
        self.position_kalman = KalmanFilter(process_noise=0.001, measurement_noise=0.1)
        
        # Initialize Kalman filter for drawing
        self.drawing_kalman = KalmanFilter(process_noise=0.0005, measurement_noise=0.05)
        
        # Stroke management for better drawing
        self.stroke_manager = StrokeManager()
        
        # Hand visibility tracking
        self.hand_visible = False
        self.hand_lost_time = 0
        self.hand_visibility_threshold = 0.3  # seconds

        # Double click tracking
        self.double_click_active = False
        self.double_click_start_time = 0
        self.double_click_duration = 0.5  # Time window for double click gesture to be recognized

    def get_finger_states(self, hand_landmarks):
        """Get the state of each finger (up/down) with improved accuracy"""
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        finger_mcps = [
            self.mp_hands.HandLandmark.THUMB_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]

        # Get wrist position for reference
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Check thumb separately using angle
        thumb_tip = hand_landmarks.landmark[finger_tips[0]]
        thumb_mcp = hand_landmarks.landmark[finger_mcps[0]]
        thumb_extended = (thumb_tip.x - thumb_mcp.x) > 0.05  # Adjust threshold as needed

        # Check other fingers using height comparison
        fingers_extended = [thumb_extended]
        for i in range(1, 5):  # For index through pinky
            tip = hand_landmarks.landmark[finger_tips[i]]
            pip = hand_landmarks.landmark[finger_pips[i]]
            mcp = hand_landmarks.landmark[finger_mcps[i]]
            
            # Calculate vectors
            vec_base = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
            vec_finger = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])
            
            # Normalize vectors
            vec_base = vec_base / np.linalg.norm(vec_base)
            vec_finger = vec_finger / np.linalg.norm(vec_finger)
            
            # Calculate angle between vectors
            angle = np.arccos(np.clip(np.dot(vec_base, vec_finger), -1.0, 1.0))
            
            # Consider finger extended if angle is small (finger is straight)
            fingers_extended.append(angle < 0.7)  # Threshold in radians (about 40 degrees)

        return fingers_extended

    def detect_gestures(self, hand_landmarks):
        """Enhanced gesture detection with improved reliability"""
        fingers_extended = self.get_finger_states(hand_landmarks)
        
        # Get specific landmark positions
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        
        # Calculate thumb-index distance for pinch detection
        thumb_index_dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )

        # Define gestures with improved accuracy
        is_whiteboard_toggle = (not any(fingers_extended[:4]) and 
                              fingers_extended[4] and  # Only pinky raised
                              thumb_index_dist > 0.1)  # Ensure no pinch
                              
        # Mouse toggle gesture - pinky AND ring finger up, others down
        is_mouse_toggle = (not fingers_extended[0] and  # Thumb down
                          not fingers_extended[1] and  # Index down
                          not fingers_extended[2] and  # Middle down
                          fingers_extended[3] and      # Ring up
                          fingers_extended[4])         # Pinky up
        # Double click gesture - thumb, index, and pinky up, others down
        is_double_click = (fingers_extended[0] and     # Thumb up
                        fingers_extended[1] and     # Index up
                        not fingers_extended[2] and  # Middle down
                        not fingers_extended[3] and  # Ring down
                        fingers_extended[4])         # Pinky up
        gestures = {
            'pinch': thumb_index_dist < 0.05,
            'eraser': fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]),
            'copy': fingers_extended[1] and fingers_extended[2] and fingers_extended[3] and not fingers_extended[4],
            'paste': all(fingers_extended[1:]),
            'toggle_whiteboard': is_whiteboard_toggle,
            'toggle_mouse': is_mouse_toggle,
            'double_click': is_double_click 
        }
        
        return gestures

    def handle_toggle_gestures(self, gestures):
        """Handle toggle gestures with proper timing"""
        current_time = time.time()
        
        # Handle whiteboard toggle
        if gestures['toggle_whiteboard']:
            if self.toggle_wb_gesture_start is None:
                self.toggle_wb_gesture_start = current_time
            elif (current_time - self.toggle_wb_gesture_start) >= self.toggle_hold_duration:
                if (current_time - self.last_gesture_time) > self.gesture_cooldown:
                    self.whiteboard_active = not self.whiteboard_active
                    self.last_gesture_time = current_time
                    self.toggle_wb_gesture_start = None
                    print(f"Whiteboard {'activated' if self.whiteboard_active else 'deactivated'}!")
        else:
            self.toggle_wb_gesture_start = None
            
        # Handle mouse toggle
        if gestures['toggle_mouse']:
            if self.toggle_mouse_gesture_start is None:
                self.toggle_mouse_gesture_start = current_time
            elif (current_time - self.toggle_mouse_gesture_start) >= self.toggle_hold_duration:
                if (current_time - self.last_gesture_time) > self.gesture_cooldown:
                    self.mouse_control_active = not self.mouse_control_active
                    self.last_gesture_time = current_time
                    self.toggle_mouse_gesture_start = None
                    print(f"Mouse control {'activated' if self.mouse_control_active else 'deactivated'}!")
        else:
            self.toggle_mouse_gesture_start = None

    def handle_mouse_and_drawing(self, hand_landmarks, frame_width, frame_height, frame):
        """Enhanced handler for mouse, drawing, and gestures with Kalman filtering"""
        # Get index finger tip position
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Convert to pixel coordinates
        x = int(index_tip.x * frame_width)
        y = int(index_tip.y * frame_height)
        
        # Track that hand is visible
        self.hand_visible = True
        
        # Apply Kalman filter for smooth tracking
        filtered_x, filtered_y = self.position_kalman.filter(x, y)
        
        # Check if inside virtual monitor
        self.in_virtual_monitor = (self.virtual_monitor_x < filtered_x < (self.virtual_monitor_x + self.virtual_monitor_width) and 
                                  self.virtual_monitor_y < filtered_y < (self.virtual_monitor_y + self.virtual_monitor_height))
        
        # Convert to screen coordinates if in virtual monitor
        screen_x, screen_y = filtered_x, filtered_y
        if self.in_virtual_monitor:
            # Map virtual monitor to full screen
            screen_x = int((filtered_x - self.virtual_monitor_x) / self.virtual_monitor_width * self.screen_width)
            screen_y = int((filtered_y - self.virtual_monitor_y) / self.virtual_monitor_height * self.screen_height)
            
            # Ensure coordinates are within screen bounds
            screen_x = max(0, min(screen_x, self.screen_width - 1))
            screen_y = max(0, min(screen_y, self.screen_height - 1))
            
            # Move mouse cursor only if mouse control is active AND pointer is inside virtual monitor
            if self.mouse_control_active:
                win32api.SetCursorPos((screen_x, screen_y))
        
        # Detect gestures and handle toggles
        gestures = self.detect_gestures(hand_landmarks)
        self.handle_toggle_gestures(gestures)
        
        # Handle scroll if mouse control is active AND in virtual monitor
        if self.mouse_control_active and self.in_virtual_monitor:
            self.handle_scroll(hand_landmarks)
        
        current_time = time.time()
        
        # Handle copy/paste gestures
        if self.mouse_control_active and self.in_virtual_monitor and current_time - self.last_gesture_time > self.gesture_cooldown:
            if gestures['copy']:
                pyautogui.hotkey('ctrl', 'c')
                print("Copied")
                self.last_gesture_time = current_time
            elif gestures['paste']:
                pyautogui.hotkey('ctrl', 'v')
                print("Pasted")
                self.last_gesture_time = current_time

        # Handle double click gesture
        if gestures['double_click']:
            if not self.double_click_active:
                self.double_click_active = True
                self.double_click_start_time = current_time
                print("Double click gesture detected")
            elif current_time - self.double_click_start_time < self.double_click_duration:
                # Perform double click
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
                print("Double click performed")
                self.last_gesture_time = current_time
                self.double_click_active = False
        else:
            # Reset double click state if the gesture is no longer held
            self.double_click_active = False
        
        # Handle drawing and mouse states
        if self.mouse_control_active and self.in_virtual_monitor:
            # Pinch gesture handling
            if gestures['pinch'] and not self.mouse_pressed and (current_time - self.last_click_time) > self.click_cooldown:
                # Mouse down action
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
                self.mouse_pressed = True
                self.last_click_time = current_time
                
                # Start drawing if whiteboard is active
                if self.whiteboard_active:
                    self.drawing = True
                    # Use the drawing-specific Kalman filter for enhanced precision
                    draw_x, draw_y = self.drawing_kalman.filter(filtered_x, filtered_y)
                    self.stroke_manager.start_stroke((draw_x, draw_y))
                
            # Release pinch gesture handling
            elif not gestures['pinch'] and self.mouse_pressed:
                # Mouse up action
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
                self.mouse_pressed = False
                
                # End drawing if active
                if self.drawing:
                    self.drawing = False
                    self.stroke_manager.end_stroke()
            
            # Continue drawing if in drawing mode
            elif self.drawing and gestures['pinch']:
                # Use the drawing-specific Kalman filter for drawing movement
                draw_x, draw_y = self.drawing_kalman.filter(filtered_x, filtered_y)
                self.stroke_manager.add_point((draw_x, draw_y))
        
        # Handle whiteboard drawing and visualization
        if self.whiteboard_active:
            # Draw current stroke if in drawing mode
            if self.drawing:
                current_stroke = self.stroke_manager.get_current_stroke()
                if len(current_stroke) >= 2:
                    for i in range(len(current_stroke) - 1):
                        cv2.line(self.whiteboard, 
                                current_stroke[i], 
                                current_stroke[i + 1], 
                                self.draw_color, 
                                self.draw_thickness, 
                                cv2.LINE_AA)
            
            # Handle eraser
            if gestures['eraser']:
                cv2.circle(self.whiteboard, (filtered_x, filtered_y), self.eraser_size, self.eraser_color, -1)
        
        
        # Store last position
        self.last_actual_pos = (x, y)
        
        # Return filtered coordinates for visualization
        return filtered_x, filtered_y
    
    def handle_scroll(self, hand_landmarks):
        """Enhanced scroll handler with improved reliability"""
        fingers_extended = self.get_finger_states(hand_landmarks)
        is_scroll_gesture = (fingers_extended[1] and f
                           fingers_extended[2] and 
                           not any(fingers_extended[3:]) and 
                           not fingers_extended[0])
        
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        current_y = middle_tip.y
        
        if is_scroll_gesture:
            if not self.scroll_active:
                self.scroll_active = True
                self.last_scroll_y = current_y
                self.scroll_gesture_start_y = current_y
                self.scroll_velocity = 0
            else:
                if self.last_scroll_y is not None:
                    # Calculate raw y difference
                    y_diff = current_y - self.last_scroll_y
                    
                    # Apply threshold to avoid unintentional small movements
                    if abs(y_diff) > self.scroll_threshold:
                        # Use Kalman filter to smooth scroll behavior
                        self.scroll_velocity = self.scroll_velocity * 0.7 + y_diff * 0.3
                        scroll_amount = int(self.scroll_velocity * 1000)
                        scroll_amount = max(min(scroll_amount, 50), -50)  # Limit scroll amount
                        win32api.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -scroll_amount, 0)
                        
                    self.last_scroll_y = current_y
        else:
            # Reset scroll state when gesture ends
            self.scroll_active = False
            self.last_scroll_y = None
            self.scroll_velocity = 0

    def update_virtual_monitor(self, frame):
        """Update virtual monitor position based on face position"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detection.process(rgb_frame)
        
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                head_x = int(bboxC.xmin * w + bboxC.width * w / 2)
                head_y = int(bboxC.ymin * h + bboxC.height * h / 2)
                
                # Apply Kalman filtering to face position for smoother transitions
                # Only slightly adjust the monitor position each frame for smoother motion
                target_x = max(0, head_x - self.virtual_monitor_width // 2)
                target_y = max(0, head_y - self.virtual_monitor_height // 2)
                
                # Smooth transition
                self.virtual_monitor_x = int(self.virtual_monitor_x * 0.9 + target_x * 0.1)
                self.virtual_monitor_y = int(self.virtual_monitor_y * 0.9 + target_y * 0.1)
                
                # Ensure the virtual monitor stays within frame bounds
                self.virtual_monitor_x = min(self.virtual_monitor_x, frame.shape[1] - self.virtual_monitor_width)
                self.virtual_monitor_y = min(self.virtual_monitor_y, frame.shape[0] - self.virtual_monitor_height)
                break  # Process only the first detected face

    def check_hand_visibility(self):
        """Check if hand is visible and handle disappearance appropriately"""
        current_time = time.time()
        
        if not self.hand_visible:
            # Hand not visible in this frame
            if self.drawing:
                # If hand was lost while drawing, check if we should end the stroke
                if self.hand_lost_time == 0:
                    # Start timing when hand first disappears
                    self.hand_lost_time = current_time
                elif current_time - self.hand_lost_time > self.hand_visibility_threshold:
                    # Hand has been gone for too long, end the stroke
                    self.drawing = False
                    self.stroke_manager.end_stroke()
                    
                    # Release mouse if needed
                    if self.mouse_pressed:
                        screen_x, screen_y = win32api.GetCursorPos()
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
                        self.mouse_pressed = False
        else:
            # Hand is visible, reset the lost timer
            self.hand_lost_time = 0
        
        # Reset hand visibility for next frame
        self.hand_visible = False

    def process_frame(self):
        """Process each frame and update display with cursor overlay"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                return False

            frame = cv2.flip(frame, 1)
            
            # Update virtual monitor position based on face position
            self.update_virtual_monitor(frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Create a copy of whiteboard for display
            if self.whiteboard_active:
                display = self.whiteboard.copy()
            else:
                display = frame.copy()

            # Check for hand visibility and handle hand tracking
            cursor_pos = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    cursor_x, cursor_y = self.handle_mouse_and_drawing(hand_landmarks, frame.shape[1], frame.shape[0], frame)
                    cursor_pos = (cursor_x, cursor_y)
                    
                    # Only draw the cursor on the display copy, not the actual whiteboard
                    if self.whiteboard_active:
                        # Draw cursor with a more visible design
                        cv2.circle(display, (cursor_x, cursor_y), 5, (0, 0, 255), -1)  # Inner red dot
                        cv2.circle(display, (cursor_x, cursor_y), 6, (255, 255, 255), 1)  # White outline
            
            # Check hand visibility status
            self.check_hand_visibility()
            
            # Always draw virtual monitor rectangle
            monitor_color = (0, 255, 0) if self.in_virtual_monitor else (255, 0, 0)
            cv2.rectangle(frame, 
                        (self.virtual_monitor_x, self.virtual_monitor_y), 
                        (self.virtual_monitor_x + self.virtual_monitor_width, self.virtual_monitor_y + self.virtual_monitor_height), 
                        monitor_color, 2)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Display stats
            y_pos = 30  # Starting y position for text
            cv2.putText(frame, f'FPS: {int(avg_fps)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            
            # Status indicators
            cv2.putText(frame, f'Mouse: {"ON" if self.mouse_control_active else "OFF"}', (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.mouse_control_active else (0, 0, 255), 2)
            y_pos += 30
            
            cv2.putText(frame, f'Whiteboard: {"ON" if self.whiteboard_active else "OFF"}', (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.whiteboard_active else (0, 0, 255), 2)
            y_pos += 30
            
            cv2.putText(frame, f'Virtual Monitor: ON', (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            
            # Display whiteboard toggle progress if active
            if self.toggle_wb_gesture_start is not None:
                progress = min((current_time - self.toggle_wb_gesture_start) / self.toggle_hold_duration * 100, 100)
                cv2.putText(frame, f'Whiteboard Toggle: {int(progress)}%', (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
                
            # Display mouse toggle progress if active
            if self.toggle_mouse_gesture_start is not None:
                progress = min((current_time - self.toggle_mouse_gesture_start) / self.toggle_hold_duration * 100, 100)
                cv2.putText(frame, f'Mouse Toggle: {int(progress)}%', (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show/hide windows based on whiteboard state
            cv2.imshow("Hand Tracking", frame)
            if self.whiteboard_active:
                cv2.imshow("Whiteboard", display)
                cv2.namedWindow("Whiteboard", cv2.WINDOW_NORMAL)
                cv2.moveWindow("Whiteboard", int(self.screen_width/4), int(self.screen_height/4))
                cv2.resizeWindow("Whiteboard", self.whiteboard_width, self.whiteboard_height)
            else:
                try:
                    cv2.destroyWindow("Whiteboard")
                except:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
                self.stroke_manager.clear_strokes()
                print("Whiteboard cleared")
            elif key == ord('b'):
                self.draw_color = (0, 0, 0)  # Black
                print("Color set to black")
            elif key == ord('r'):
                self.draw_color = (0, 0, 255)  # Red
                print("Color set to red")
            elif key == ord('g'):
                self.draw_color = (0, 255, 0)  # Green
                print("Color set to green")
            elif key == ord('b'):
                self.draw_color = (255, 0, 0)  # Blue
                print("Color set to blue")
            elif key == ord('+') or key == ord('='):
                self.draw_thickness = min(10, self.draw_thickness + 1)
                print(f"Thickness increased to {self.draw_thickness}")
            elif key == ord('-'):
                self.draw_thickness = max(1, self.draw_thickness - 1)
                print(f"Thickness decreased to {self.draw_thickness}")
            elif key == ord('s'):
                # Save whiteboard to file
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"whiteboard_{timestamp}.png"
                cv2.imwrite(filename, self.whiteboard)
                print(f"Saved whiteboard as {filename}")
            
            return True
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return False

    def run(self):
        """Main loop for the gesture controller"""
        self.running = True
        try:
            while self.running:
                if not self.process_frame():
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Gesture controller stopped")

    def improved_draw_strokes(self, frame):
        """Improved drawing with stroke separation and smoothing"""
        # Draw completed strokes
        for stroke in self.stroke_manager.get_all_strokes():
            if len(stroke) >= 2:
                for i in range(len(stroke) - 1):
                    cv2.line(frame, 
                            stroke[i], 
                            stroke[i + 1], 
                            self.draw_color, 
                            self.draw_thickness, 
                            cv2.LINE_AA)
        
        # Draw current stroke
        current_stroke = self.stroke_manager.get_current_stroke()
        if len(current_stroke) >= 2:
            for i in range(len(current_stroke) - 1):
                cv2.line(frame, 
                        current_stroke[i], 
                        current_stroke[i + 1], 
                        self.draw_color, 
                        self.draw_thickness, 
                        cv2.LINE_AA)
                        
        return frame


class GestureControllerApp:
    """Main application for the gesture controller"""
    def __init__(self):
        self.controller = GestureController()
        
    def start(self):
        """Start the application"""
        print("Starting gesture controller...")
        print("\nKEYBOARD CONTROLS:")
        print("q - Quit")
        print("c - Clear whiteboard")
        print("r - Set color to red")
        print("g - Set color to green")
        print("b - Set color to blue")
        print("+ - Increase pen thickness")
        print("- - Decrease pen thickness")
        print("s - Save whiteboard to PNG file")
        print("\nGESTURE CONTROLS:")
        print("Pinch index finger and thumb - Mouse click / Draw")
        print("Index + Middle finger up - Eraser")
        print("Index + Middle + Ring up - Copy (Ctrl+C)")
        print("All fingers up - Paste (Ctrl+V)")
        print("Only Pinky up (hold) - Toggle whiteboard")
        print("Ring + Pinky up (hold) - Toggle mouse control")
        print("Index + Middle up, move up/down - Scroll")
        print("\nVirtual Monitor:")
        print("Green box when inside, Red when outside")
        print("\nStarting camera...")
        
        self.controller.run()


if __name__ == "__main__":
    app = GestureControllerApp()
    app.start()