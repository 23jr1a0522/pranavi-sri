import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Function to detect hand gestures
def detect_gesture(frame):
    # Convert BGR frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Example: Using the position of the thumb and index finger for simple gesture detection
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate the distance between thumb and index finger to determine gesture
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            
            if distance < 0.05:
                gesture = "Thumbs Up"
            else:
                gesture = "Hand Open"
            
            # Return gesture
            return gesture
    return None

# Streamlit Interface
st.title("Gesture-Based Human-Computer Interaction")
st.write("Use your hand gestures for interaction!")

# Capture webcam feed
cap = cv2.VideoCapture(0)

# Create a placeholder for showing the video feed
frame_placeholder = st.empty()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect gestures in the frame
    gesture = detect_gesture(frame)
    
    # Display gesture on screen
    if gesture:
        cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Convert frame to RGB for Streamlit compatibility
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb)

# Release the webcam feed
cap.release()