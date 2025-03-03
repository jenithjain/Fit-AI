import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class BodyMeasurement:
    def _init_(self):
        print("Initializing system...")
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Size chart
        self.size_chart = {
            "S": {"shoulder": (38, 41), "chest": (86, 91), "waist": (71, 76)},
            "M": {"shoulder": (41, 44), "chest": (91, 97), "waist": (76, 81)},
            "L": {"shoulder": (44, 47), "chest": (97, 102), "waist": (81, 86)},
            "XL": {"shoulder": (47, 50), "chest": (102, 107), "waist": (86, 91)}
        }
        
        # Initialize measurement buffer for smoothing
        self.measurement_buffer = []
        self.buffer_size = 10

    def calculate_measurements(self, landmarks, frame_width, frame_height):
        """Calculate body measurements from landmarks"""
        
        # Get key points
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate pixel distances
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * frame_width
        hip_width = abs(right_hip.x - left_hip.x) * frame_width
        
        # Convert pixels to cm (approximate based on average human proportions)
        shoulder_cm = shoulder_width * 0.264583
        chest_cm = shoulder_cm * 2.2  # Chest is typically wider than shoulders
        waist_cm = hip_width * 0.264583 * 1.15
        
        return {
            "shoulder": shoulder_cm,
            "chest": chest_cm,
            "waist": waist_cm
        }

    def smooth_measurements(self, measurements):
        """Apply smoothing to measurements using moving average"""
        self.measurement_buffer.append(measurements)
        if len(self.measurement_buffer) > self.buffer_size:
            self.measurement_buffer.pop(0)
        
        smoothed = {}
        for key in measurements.keys():
            values = [m[key] for m in self.measurement_buffer]
            smoothed[key] = sum(values) / len(values)
        return smoothed

    def get_size_recommendation(self, measurements):
        """Determine size based on measurements"""
        size_scores = {size: 0 for size in self.size_chart.keys()}
        
        for size, ranges in self.size_chart.items():
            for measurement, value in measurements.items():
                if ranges[measurement][0] <= value <= ranges[measurement][1]:
                    size_scores[size] += 1
        
        # Get the size with highest score
        recommended_size = max(size_scores.items(), key=lambda x: x[1])[0]
        return recommended_size

    def start_measurement(self):
        print("Starting camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return
            
        print("\n=== Body Measurement System ===")
        print("Stand 2-3 feet from camera")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                    self.mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
                )
                
                # Calculate measurements
                measurements = self.calculate_measurements(
                    results.pose_landmarks.landmark,
                    frame.shape[1],
                    frame.shape[0]
                )
                
                # Smooth measurements
                smoothed = self.smooth_measurements(measurements)
                
                # Get size recommendation
                size = self.get_size_recommendation(smoothed)
                
                # Display measurements
                y_pos = 30
                for measure, value in smoothed.items():
                    cv2.putText(frame, f"{measure}: {value:.1f} cm",
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 255, 0), 2)
                    y_pos += 30
                
                # Display size recommendation
                cv2.putText(frame, f"Recommended Size: {size}",
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Body Measurement', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "_main_":
    measurer = BodyMeasurement()
    measurer.start_measurement()
    
# # Alternative OpenCV installation
# pip install opencv-contrib-python

# # Or try with --user flag if you have permission issues
# python -m pip install --user opencv-python
# python -m pip install --user mediapipe
# python -m pip install --user numpy
# python -m pip install --user scipy    