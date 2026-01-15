import mediapipe as mp

if hasattr(mp, "Holistic"):
    print("MediaPipe Holistic found!")
else:
    print("MediaPipe Holistic NOT found. Using fallback...")
    Holistic = mp.solutions.holistic

print(f"Holistic class: {Holistic}")
