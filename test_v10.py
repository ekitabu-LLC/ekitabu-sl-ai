"""
Test KSL v10 model on raw video files.
Extracts MediaPipe landmarks and runs inference.

Usage:
    modal run test_v10.py
"""

import modal
import os

app = modal.App("ksl-test-v10")
volume = modal.Volume.from_name("ksl-dataset-vol")

# Upload testing folder to Modal volume first
local_testing_path = "/Users/hassan/Documents/Hassan/ksl-dir-2/testing"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install("torch", "numpy", "opencv-python-headless", "mediapipe==0.10.9")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])
ALL_CLASSES = NUMBER_CLASSES + WORD_CLASSES

HAND_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
]

CONFIG = {
    "max_frames": 90, "num_nodes": 42, "in_channels": 3,
    "hidden_dim": 128, "num_layers": 6, "temporal_kernel": 9, "dropout": 0.3,
    "checkpoint_path": "/data/checkpoints_v10/best_model.pt",
}


@app.function(volumes={"/data": volume}, timeout=600, image=image)
def upload_test_videos(videos_data: list):
    """Upload test videos to Modal volume."""
    import os
    os.makedirs("/data/testing/Numbers", exist_ok=True)
    os.makedirs("/data/testing/Words - Left Side", exist_ok=True)

    for item in videos_data:
        path = f"/data/testing/{item['subdir']}/{item['filename']}"
        with open(path, "wb") as f:
            f.write(item["data"])
        print(f"Uploaded: {path}")

    volume.commit()
    return len(videos_data)


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def test_on_videos():
    import torch
    import torch.nn as nn
    import numpy as np
    import cv2
    import mediapipe as mp

    print("=" * 70)
    print("KSL v10 Testing on Raw Videos")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_adj(n=42):
        adj = np.zeros((n, n))
        for i, j in HAND_EDGES:
            adj[i, j] = adj[j, i] = 1
            adj[i+21, j+21] = adj[j+21, i+21] = 1
        adj[0, 21] = adj[21, 0] = 0.5
        adj += np.eye(n)
        d = np.sum(adj, axis=1)
        d_inv = np.power(d, -0.5)
        d_inv[np.isinf(d_inv)] = 0
        return torch.FloatTensor(np.diag(d_inv) @ adj @ np.diag(d_inv))

    class GConv(nn.Module):
        def __init__(self, inf, outf):
            super().__init__()
            self.fc = nn.Linear(inf, outf)
        def forward(self, x, adj):
            return self.fc(torch.matmul(adj, x))

    class STBlock(nn.Module):
        def __init__(self, ic, oc, adj, ks=9, st=1, dr=0.3):
            super().__init__()
            self.register_buffer('adj', adj)
            self.gcn = GConv(ic, oc)
            self.bn1 = nn.BatchNorm2d(oc)
            self.tcn = nn.Sequential(
                nn.Conv2d(oc, oc, (ks, 1), padding=(ks//2, 0), stride=(st, 1)),
                nn.BatchNorm2d(oc))
            self.residual = nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st, 1)), nn.BatchNorm2d(oc)) if ic != oc or st != 1 else nn.Identity()
            self.dropout = nn.Dropout(dr)
        def forward(self, x):
            r = self.residual(x)
            b, c, t, n = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b*t, n, c)
            x = self.gcn(x, self.adj)
            x = x.reshape(b, t, n, -1).permute(0, 3, 1, 2)
            x = torch.relu(self.bn1(x))
            x = self.dropout(self.tcn(x))
            return torch.relu(x + r)

    class STGCN(nn.Module):
        def __init__(self, nc, num_n=42, ic=3, hd=128, nl=6, tk=9, dr=0.3, adj=None):
            super().__init__()
            self.register_buffer('adj', adj)
            self.data_bn = nn.BatchNorm1d(num_n * ic)
            ch = [ic] + [hd]*2 + [hd*2]*2 + [hd*4]*2
            ch = ch[:nl+1]
            self.layers = nn.ModuleList([STBlock(ch[i], ch[i+1], adj, tk, 2 if i in [2,4] else 1, dr) for i in range(nl)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Linear(ch[-1], hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc))
        def forward(self, x):
            b, c, t, n = x.shape
            x = self.data_bn(x.permute(0, 1, 3, 2).reshape(b, c*n, t)).reshape(b, c, n, t).permute(0, 1, 3, 2)
            for l in self.layers:
                x = l(x)
            return self.classifier(self.pool(x).view(b, -1))

    def extract_landmarks(vpath):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        cap = cv2.VideoCapture(vpath)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            lh, rh = np.zeros((21, 3)), np.zeros((21, 3))
            if res.multi_hand_landmarks:
                for hlm, hnd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    lm = np.array([[l.x, l.y, l.z] for l in hlm.landmark])
                    if hnd.classification[0].label == "Left":
                        lh = lm
                    else:
                        rh = lm
            frames.append(np.concatenate([lh, rh], axis=0))
        cap.release()
        hands.close()
        return np.array(frames, dtype=np.float32) if frames else None

    def preprocess(data, mf=90):
        data[:, :21, :] -= data[:, 0:1, :]
        data[:, 21:, :] -= data[:, 21:22, :]
        data = data / (np.abs(data).max() + 1e-8)
        f = data.shape[0]
        if f >= mf:
            data = data[np.linspace(0, f-1, mf, dtype=int)]
        else:
            data = np.concatenate([data, np.zeros((mf-f, *data.shape[1:]))], axis=0)
        return torch.FloatTensor(data).permute(2, 0, 1).unsqueeze(0)

    adj = build_adj(42).to(device)
    model = STGCN(len(ALL_CLASSES), 42, 3, 128, 6, 9, 0.3, adj).to(device)
    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model - Val: {ckpt.get('val_acc', 0):.1f}%, 444: {ckpt.get('val_444', 0):.1f}%")

    idx2cls = {i: c for i, c in enumerate(ALL_CLASSES)}
    results = []

    for folder, prefix in [("/data/testing/Numbers", "No. "), ("/data/testing/Words - Left Side", "")]:
        if not os.path.exists(folder):
            continue
        print(f"\n--- {folder.split('/')[-1]} ---")
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".mov"):
                continue
            if prefix:
                gt = fn.replace(prefix, "").replace(".mov", "")
            else:
                gt = fn.split(". ", 1)[1].replace(".mov", "") if ". " in fn else fn.replace(".mov", "")

            print(f"\n{fn}")
            lm = extract_landmarks(os.path.join(folder, fn))
            if lm is None or len(lm) < 5:
                print("  FAILED: No landmarks")
                results.append({"gt": gt, "pred": "FAILED", "ok": False})
                continue

            print(f"  Frames: {len(lm)}")
            inp = preprocess(lm, 90).to(device)
            with torch.no_grad():
                out = model(inp)
                prob = torch.softmax(out, dim=1)
                pidx = out.argmax(1).item()
                pred = idx2cls[pidx]
                conf = prob[0, pidx].item()

            ok = pred == gt
            print(f"  GT: {gt} | Pred: {pred} ({conf*100:.1f}%) | {'OK' if ok else 'WRONG'}")
            t3 = prob[0].topk(3)
            print(f"  Top3: {', '.join([f'{idx2cls[i.item()]}:{v.item()*100:.0f}%' for v,i in zip(t3.values, t3.indices)])}")
            results.append({"gt": gt, "pred": pred, "conf": conf, "ok": ok})

    print("\n" + "=" * 70)
    tot = len(results)
    ok = sum(r["ok"] for r in results)
    fail = sum(r["pred"] == "FAILED" for r in results)
    print(f"Total: {tot} | Correct: {ok}/{tot-fail} ({100*ok/(tot-fail) if tot>fail else 0:.1f}%) | Failed: {fail}")

    nums = [r for r in results if r["gt"] in NUMBER_CLASSES]
    wrds = [r for r in results if r["gt"] in WORD_CLASSES]
    print(f"Numbers: {sum(r['ok'] for r in nums)}/{len(nums)} | Words: {sum(r['ok'] for r in wrds)}/{len(wrds)}")

    r444 = [r for r in results if r["gt"] == "444"]
    if r444:
        print(f"444: {r444[0]['pred']} ({'OK' if r444[0]['ok'] else 'WRONG'})")

    return {"acc": 100*ok/(tot-fail) if tot>fail else 0}


@app.local_entrypoint()
def main():
    import os

    # Check if we need to upload test videos
    print("Checking if test videos need uploading...")

    # Collect test videos from local folder
    local_path = "/Users/hassan/Documents/Hassan/ksl-dir-2/testing"
    videos_data = []

    for subdir in ["Numbers", "Words - Left Side"]:
        folder = os.path.join(local_path, subdir)
        if os.path.exists(folder):
            for fn in os.listdir(folder):
                if fn.endswith(".mov"):
                    fpath = os.path.join(folder, fn)
                    with open(fpath, "rb") as f:
                        videos_data.append({
                            "subdir": subdir,
                            "filename": fn,
                            "data": f.read()
                        })

    if videos_data:
        print(f"Uploading {len(videos_data)} test videos to Modal...")
        count = upload_test_videos.remote(videos_data)
        print(f"Uploaded {count} videos")

    print("\nRunning inference...")
    r = test_on_videos.remote()
    print(f"\nTest Accuracy: {r['acc']:.1f}%")
