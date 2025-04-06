import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict
import time

class GestureEvaluator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        
        # Metrics tracking
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.total_frames = 0
        self.detection_times = []
        self.correct_detections = 0  # Track total correct detections
        
        # Gesture ground truth states
        self.current_gesture = None
        self.gestures = ['pinch', 'eraser', 'copy', 'paste', 'toggle']
        
    def get_finger_states(self, hand_landmarks):
        """Get the state of each finger (up/down)"""
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

        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[finger_tips[0]]
        thumb_mcp = hand_landmarks.landmark[finger_mcps[0]]
        thumb_extended = (thumb_tip.x - thumb_mcp.x) > 0.05

        fingers_extended = [thumb_extended]
        for i in range(1, 5):
            tip = hand_landmarks.landmark[finger_tips[i]]
            pip = hand_landmarks.landmark[finger_pips[i]]
            mcp = hand_landmarks.landmark[finger_mcps[i]]
            
            vec_base = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
            vec_finger = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])
            
            vec_base = vec_base / np.linalg.norm(vec_base)
            vec_finger = vec_finger / np.linalg.norm(vec_finger)
            
            angle = np.arccos(np.clip(np.dot(vec_base, vec_finger), -1.0, 1.0))
            fingers_extended.append(angle < 0.7)

        return fingers_extended

    def detect_gestures(self, hand_landmarks):
        """Detect gestures and return all detected gestures"""
        fingers_extended = self.get_finger_states(hand_landmarks)
        
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )

        is_toggle = (
         not fingers_extended[1] and  # Index down
         not fingers_extended[2] and  # Middle down
         not fingers_extended[3] and  # Ring down
         fingers_extended[4] and      # Pinky up
         thumb_index_dist > 0.05      # Reduced distance threshold
    )
        detected_gestures = {
            'pinch': thumb_index_dist < 0.05,
            'eraser': fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]),
            'copy': fingers_extended[1] and fingers_extended[2] and fingers_extended[3] and not fingers_extended[4],
            'paste': all(fingers_extended[1:]),
            'toggle': is_toggle
        }
        
        return detected_gestures

    def evaluate_frame(self, frame):
        """Evaluate a single frame and update metrics"""
        start_time = time.time()
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        self.total_frames += 1
        frame_correct = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                detected_gestures = self.detect_gestures(hand_landmarks)
                
                # Update metrics for each gesture
                for gesture in self.gestures:
                    if detected_gestures[gesture]:
                        if gesture == self.current_gesture:
                            self.true_positives[gesture] += 1
                            frame_correct = True
                        else:
                            self.false_positives[gesture] += 1
                    elif gesture == self.current_gesture:
                        self.false_negatives[gesture] += 1
        
        if frame_correct:
            self.correct_detections += 1
            
        # Record detection time
        self.detection_times.append(time.time() - start_time)
        
        return results.multi_hand_landmarks is not None

    def calculate_metrics(self):
        """Calculate accuracy, precision, recall, and F1-score for each gesture"""
        metrics = {}
        
        # Calculate overall accuracy
        overall_accuracy = self.correct_detections / self.total_frames if self.total_frames > 0 else 0
        
        for gesture in self.gestures:
            tp = self.true_positives[gesture]
            fp = self.false_positives[gesture]
            fn = self.false_negatives[gesture]
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[gesture] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
        
        # Calculate average detection time
        avg_detection_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
        
        return metrics, avg_detection_time, overall_accuracy

    def run_evaluation(self, num_frames=1000):
        """Run evaluation for specified number of frames"""
        cap = cv2.VideoCapture(0)
        
        print("Starting evaluation...")
        print("Instructions:")
        print("1. Perform each gesture when prompted")
        print("2. Press 'q' to quit at any time")
        print("3. Hold each gesture steady for accurate evaluation")
        
        try:
            for gesture in self.gestures:
                print(f"\nPerforming evaluation for '{gesture}' gesture...")
                print("Get ready... (3 seconds)")
                time.sleep(3)
                
                self.current_gesture = gesture
                frames_evaluated = 0
                
                while frames_evaluated < num_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    hand_detected = self.evaluate_frame(frame)
                    
                    # Display frame count and gesture
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frames: {frames_evaluated}/{num_frames}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Evaluation", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                    
                    if hand_detected:
                        frames_evaluated += 1
        
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate and display metrics
        metrics, avg_detection_time, overall_accuracy = self.calculate_metrics()
        
        print("\nEvaluation Results:")
        print("-" * 50)
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        print(f"Average detection time: {avg_detection_time*1000:.2f}ms")
        print("\nPer-gesture metrics:")
        
        for gesture in self.gestures:
            print(f"\n{gesture.upper()}:")
            print(f"Precision: {metrics[gesture]['precision']:.3f}")
            print(f"Recall: {metrics[gesture]['recall']:.3f}")
            print(f"F1-Score: {metrics[gesture]['f1_score']:.3f}")

if __name__ == "__main__":
    evaluator = GestureEvaluator()
    evaluator.run_evaluation(num_frames=100)  # Evaluate 100 frames per gesture