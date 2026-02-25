"""
Test KSL v12 (Pose + Hands) on raw videos.

Usage:
    modal run test_v12.py
"""

import modal
import os

app = modal.App("ksl-test-v12")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install("torch", "numpy", "opencv-python-headless", "mediapipe==0.10.9")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])

HAND_EDGES = [(0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
              (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

POSE_EDGES = [
    (42, 43), (42, 44), (43, 45), (44, 46), (45, 47), (46, 0), (47, 21),
]

POSE_INDICES = [11, 12, 13, 14, 15, 16]


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def test_pose():
    import torch
    import torch.nn as nn
    import numpy as np
    import cv2
    import mediapipe as mp

    print("=" * 70)
    print("KSL v12 Testing (Pose + Hands)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_adj(n=48):
        adj = np.zeros((n, n))
        for i, j in HAND_EDGES:
            adj[i, j] = adj[j, i] = 1
            adj[i+21, j+21] = adj[j+21, i+21] = 1
        for i, j in POSE_EDGES:
            adj[i, j] = adj[j, i] = 1
        adj[0, 21] = adj[21, 0] = 0.3
        adj += np.eye(n)
        d = np.sum(adj, axis=1)
        d_inv = np.power(d, -0.5)
        d_inv[np.isinf(d_inv)] = 0
        return torch.FloatTensor(np.diag(d_inv) @ adj @ np.diag(d_inv))

    class GConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.fc = nn.Linear(i, o)
        def forward(self, x, adj):
            return self.fc(torch.matmul(adj, x))

    class Block(nn.Module):
        def __init__(self, ic, oc, adj, ks=9, st=1, dr=0.3):
            super().__init__()
            self.register_buffer('adj', adj)
            self.gcn = GConv(ic, oc)
            self.bn1 = nn.BatchNorm2d(oc)
            self.tcn = nn.Sequential(nn.Conv2d(oc, oc, (ks,1), padding=(ks//2,0), stride=(st,1)), nn.BatchNorm2d(oc))
            self.residual = nn.Sequential(nn.Conv2d(ic, oc, 1, stride=(st,1)), nn.BatchNorm2d(oc)) if ic != oc or st != 1 else nn.Identity()
            self.dropout = nn.Dropout(dr)
        def forward(self, x):
            r = self.residual(x)
            b, c, t, n = x.shape
            x = self.gcn(x.permute(0,2,3,1).reshape(b*t, n, c), self.adj)
            x = self.dropout(self.tcn(torch.relu(self.bn1(x.reshape(b,t,n,-1).permute(0,3,1,2)))))
            return torch.relu(x + r)

    class Model(nn.Module):
        def __init__(self, nc, num_n=48, ic=3, hd=128, nl=6, tk=9, dr=0.4, adj=None):
            super().__init__()
            self.register_buffer('adj', adj)
            self.data_bn = nn.BatchNorm1d(num_n * ic)
            ch = [ic] + [hd]*2 + [hd*2]*2 + [hd*4]*2
            ch = ch[:nl+1]
            self.layers = nn.ModuleList([Block(ch[i], ch[i+1], adj, tk, 2 if i in [2,4] else 1, dr) for i in range(nl)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Linear(ch[-1], hd), nn.ReLU(), nn.Dropout(dr), nn.Linear(hd, nc))
        def forward(self, x):
            b, c, t, n = x.shape
            x = self.data_bn(x.permute(0,1,3,2).reshape(b, c*n, t)).reshape(b, c, n, t).permute(0,1,3,2)
            for l in self.layers:
                x = l(x)
            return self.classifier(self.pool(x).view(b, -1))

    def get_landmarks(vpath):
        """Extract pose + hands using MediaPipe Holistic."""
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        cap = cv2.VideoCapture(vpath)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Extract hands
            lh = np.zeros((21, 3))
            rh = np.zeros((21, 3))
            if res.left_hand_landmarks:
                lh = np.array([[l.x, l.y, l.z] for l in res.left_hand_landmarks.landmark])
            if res.right_hand_landmarks:
                rh = np.array([[l.x, l.y, l.z] for l in res.right_hand_landmarks.landmark])

            # Extract upper body pose (6 landmarks)
            pose = np.zeros((6, 3))
            if res.pose_landmarks:
                for pi, idx in enumerate(POSE_INDICES):
                    lm = res.pose_landmarks.landmark[idx]
                    pose[pi] = [lm.x, lm.y, lm.z]

            # Combine: [left_hand(21), right_hand(21), pose(6)] = 48 nodes
            combined = np.concatenate([lh, rh, pose], axis=0)
            frames.append(combined)

        cap.release()
        holistic.close()
        return np.array(frames, dtype=np.float32) if frames else None

    def prep(data, mf=90):
        # Normalize hands relative to wrist
        data[:, :21, :] -= data[:, 0:1, :]
        data[:, 21:42, :] -= data[:, 21:22, :]

        # Normalize pose relative to mid-shoulder
        mid_shoulder = (data[:, 42:43, :] + data[:, 43:44, :]) / 2
        data[:, 42:48, :] -= mid_shoulder

        # Global normalization
        data = data / (np.abs(data).max() + 1e-8)

        f = data.shape[0]
        if f >= mf:
            data = data[np.linspace(0, f-1, mf, dtype=int)]
        else:
            data = np.concatenate([data, np.zeros((mf-f, 48, 3))], axis=0)
        return torch.FloatTensor(data).permute(2, 0, 1).unsqueeze(0)

    adj = build_adj(48).to(device)

    # Load models
    num_model = Model(len(NUMBER_CLASSES), 48, 3, 128, 6, 9, 0.4, adj).to(device)
    num_ckpt = torch.load("/data/checkpoints_v12_numbers/best_model.pt", map_location=device)
    num_model.load_state_dict(num_ckpt["model"])
    num_model.eval()
    print(f"Numbers model: {num_ckpt.get('val_acc', 0):.1f}%")

    word_model = Model(len(WORD_CLASSES), 48, 3, 128, 6, 9, 0.4, adj).to(device)
    word_ckpt = torch.load("/data/checkpoints_v12_words/best_model.pt", map_location=device)
    word_model.load_state_dict(word_ckpt["model"])
    word_model.eval()
    print(f"Words model: {word_ckpt.get('val_acc', 0):.1f}%")

    results = []

    print("\n--- NUMBERS ---")
    folder = "/data/testing/Numbers"
    if os.path.exists(folder):
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".mov"):
                continue
            gt = fn.replace("No. ", "").replace(".mov", "")
            print(f"\n{fn}")
            lm = get_landmarks(os.path.join(folder, fn))
            if lm is None or len(lm) < 5:
                print("  FAILED")
                results.append({"gt": gt, "ok": False, "t": "n"})
                continue
            inp = prep(lm, 90).to(device)
            with torch.no_grad():
                out = num_model(inp)
                prob = torch.softmax(out, dim=1)
                pidx = out.argmax(1).item()
                pred = NUMBER_CLASSES[pidx]
                conf = prob[0, pidx].item()
            ok = pred == gt
            print(f"  GT: {gt} | Pred: {pred} ({conf*100:.1f}%) | {'OK' if ok else 'WRONG'}")
            results.append({"gt": gt, "pred": pred, "ok": ok, "t": "n"})

    print("\n--- WORDS ---")
    folder = "/data/testing/Words - Left Side"
    if os.path.exists(folder):
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".mov"):
                continue
            gt = fn.split(". ", 1)[1].replace(".mov", "") if ". " in fn else fn.replace(".mov", "")
            print(f"\n{fn}")
            lm = get_landmarks(os.path.join(folder, fn))
            if lm is None or len(lm) < 5:
                print("  FAILED")
                results.append({"gt": gt, "ok": False, "t": "w"})
                continue
            inp = prep(lm, 90).to(device)
            with torch.no_grad():
                out = word_model(inp)
                prob = torch.softmax(out, dim=1)
                pidx = out.argmax(1).item()
                pred = WORD_CLASSES[pidx]
                conf = prob[0, pidx].item()
            ok = pred == gt
            print(f"  GT: {gt} | Pred: {pred} ({conf*100:.1f}%) | {'OK' if ok else 'WRONG'}")
            results.append({"gt": gt, "pred": pred, "ok": ok, "t": "w"})

    print("\n" + "=" * 70)
    nums = [r for r in results if r["t"] == "n"]
    wrds = [r for r in results if r["t"] == "w"]
    nok = sum(r["ok"] for r in nums)
    wok = sum(r["ok"] for r in wrds)
    print(f"Numbers: {nok}/{len(nums)} ({100*nok/len(nums):.1f}%)")
    print(f"Words: {wok}/{len(wrds)} ({100*wok/len(wrds):.1f}%)")
    print(f"Overall: {nok+wok}/{len(results)} ({100*(nok+wok)/len(results):.1f}%)")

    return {"numbers": 100*nok/len(nums), "words": 100*wok/len(wrds)}


@app.local_entrypoint()
def main():
    r = test_pose.remote()
    print(f"\nNumbers: {r['numbers']:.1f}% | Words: {r['words']:.1f}%")
