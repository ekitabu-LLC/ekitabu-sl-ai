import modal
import os
import numpy as np
import cv2

app = modal.App("ksl-extractor")
volume = modal.Volume.from_name("ksl-dataset-v2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libxext6")
    .pip_install("mediapipe==0.10.14", "opencv-python-headless", "numpy")
)


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def process_video(video_bytes, filename, class_name):
    import mediapipe as mp

    temp_path = f"/tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    mp_holistic = mp.solutions.holistic  # type: ignore
    holistic = mp_holistic.Holistic(
        static_image_mode=False, min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(temp_path)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_row = []

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_row.extend([lm.x, lm.y, lm.z])
        else:
            frame_row.extend([0] * 99)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_row.extend([lm.x, lm.y, lm.z])
        else:
            frame_row.extend([0] * 63)

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_row.extend([lm.x, lm.y, lm.z])
        else:
            frame_row.extend([0] * 63)

        frames_data.append(frame_row)

    cap.release()
    holistic.close()

    save_dir = f"/data/processed/{class_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{filename}.npy"
    np.save(save_path, np.array(frames_data))
    return f"Processed {filename}"


@app.local_entrypoint()
def main():
    video_tasks = []
    base_dir = "dataset"

    print("Scanning local dataset...")
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_path) and class_folder not in [
            "lstm_processed",
            "processed",
        ]:
            for video_file in os.listdir(class_path):
                if video_file.endswith((".mov", ".mp4")):
                    with open(os.path.join(class_path, video_file), "rb") as f:
                        content = f.read()
                    video_tasks.append((content, video_file, class_folder))

    print(f"Dispatching {len(video_tasks)} videos to Modal...")
    for res in process_video.starmap(video_tasks):
        print(res)

    print("Extraction Complete! Data saved to Volume 'ksl-dataset-v2'")
