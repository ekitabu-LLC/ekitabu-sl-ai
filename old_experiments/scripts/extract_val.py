import modal
import os
import numpy as np
import cv2 as cv2

app = modal.App("ksl-val-extractor")
volume = modal.Volume.from_name("ksl-dataset-v2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libegl1", "libxext6")
    .pip_install("mediapipe==0.10.14", "opencv-python-headless", "numpy")
)


def normalize_class_name(folder_name):
    parts = folder_name.split(". ")
    class_name = parts[1] if len(parts) > 1 else parts[0]
    if "." in class_name:
        sub_parts = class_name.split(". ")
        class_name = ".".join(sub_parts[1:])
    return class_name


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def process_video(video_bytes, filename, folder_path):
    import mediapipe as mp

    temp_path = f"/tmp/{os.path.basename(folder_path)}.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    # Use explicit Holistic variant
    Holistic = mp.solutions.Holistic
    cap = cv2.VideoCapture(temp_path)

    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = Holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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
    Holistic.close()

    class_folder = os.path.dirname(folder_path)
    class_name = normalize_class_name(os.path.basename(class_folder))

    save_dir = f"/data/val_processed/{class_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{filename}.npy"
    np.save(save_path, np.array(frames_data))

    return f"Processed {class_name} - {filename}"


@app.local_entrypoint()
def main():
    video_tasks = []
    base_dir = "validation-data"

    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        for video_file in os.listdir(class_path):
            if video_file.endswith(".mov") or video_file.endswith(".mp4"):
                video_path = os.path.join(class_path, video_file)
                folder_path = os.path.dirname(video_path)

                with open(video_path, "rb") as f:
                    video_bytes = f.read()

                filename = os.path.splitext(video_file)[0]

                video_tasks.append((video_bytes, filename, folder_path))

    print(f"Found {len(video_tasks)} videos. Processing...")

    for result in process_video.starmap(video_tasks):
        print(result)

    print("Validation processing complete!")
