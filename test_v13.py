"""
Test KSL v13 using pre-extracted features.

Usage:
    modal run test_v13.py
"""

import modal
import os

app = modal.App("ksl-test-v13")
volume = modal.Volume.from_name("ksl-dataset-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
)

NUMBER_CLASSES = sorted(["9", "17", "22", "35", "48", "54", "66", "73", "89", "91", "100", "125", "268", "388", "444"])
WORD_CLASSES = sorted(["Agreement", "Apple", "Colour", "Friend", "Gift", "Market", "Monday",
                       "Picture", "Proud", "Sweater", "Teach", "Tomatoes", "Tortoise", "Twin", "Ugali"])

HAND_EDGES = [(0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
              (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

POSE_EDGES = [(42, 43), (42, 44), (43, 45), (44, 46), (45, 47), (46, 0), (47, 21)]
POSE_INDICES = [11, 12, 13, 14, 15, 16]


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=3600, image=image)
def test_v13():
    import torch
    import torch.nn as nn
    import numpy as np

    print("=" * 70)
    print("KSL v13 Testing (Robust to Sparse Detection)")
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
        def __init__(self, nc, num_n=48, ic=3, hd=128, nl=6, tk=9, dr=0.5, adj=None):
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

    def preprocess(data, mf=90):
        f = data.shape[0]

        if data.shape[1] >= 225:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            for pi, idx_pose in enumerate(POSE_INDICES):
                start = idx_pose * 3
                pose[:, pi, :] = data[:, start:start+3]
            lh = data[:, 99:162].reshape(f, 21, 3)
            rh = data[:, 162:225].reshape(f, 21, 3)
        else:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            lh = np.zeros((f, 21, 3), dtype=np.float32)
            rh = np.zeros((f, 21, 3), dtype=np.float32)

        h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

        lh_valid = np.abs(h[:, :21, :]).sum(axis=(1,2)) > 0.01
        rh_valid = np.abs(h[:, 21:42, :]).sum(axis=(1,2)) > 0.01

        if np.any(lh_valid):
            h[lh_valid, :21, :] -= h[lh_valid, 0:1, :]
        if np.any(rh_valid):
            h[rh_valid, 21:42, :] -= h[rh_valid, 21:22, :]

        mid_shoulder = (h[:, 42:43, :] + h[:, 43:44, :]) / 2
        h[:, 42:48, :] -= mid_shoulder

        max_val = np.abs(h).max()
        if max_val > 0.01:
            h = np.clip(h / max_val, -1, 1).astype(np.float32)

        if f >= mf:
            h = h[np.linspace(0, f-1, mf, dtype=int)]
        else:
            h = np.concatenate([h, np.zeros((mf-f, 48, 3), dtype=np.float32)])

        return torch.FloatTensor(h).permute(2, 0, 1).unsqueeze(0)

    adj = build_adj(48).to(device)

    num_ckpt_path = "/data/checkpoints_v13_numbers/best_model.pt"
    word_ckpt_path = "/data/checkpoints_v13_words/best_model.pt"

    if not os.path.exists(num_ckpt_path):
        print("v13 models not found! Run train_ksl_v13.py first.")
        return {"numbers": 0, "words": 0}

    num_model = Model(len(NUMBER_CLASSES), 48, 3, 128, 6, 9, 0.5, adj).to(device)
    num_ckpt = torch.load(num_ckpt_path, map_location=device)
    num_model.load_state_dict(num_ckpt["model"])
    num_model.eval()
    print(f"Numbers model: {num_ckpt.get('val_acc', 0):.1f}%")

    word_model = Model(len(WORD_CLASSES), 48, 3, 128, 6, 9, 0.5, adj).to(device)
    word_ckpt = torch.load(word_ckpt_path, map_location=device)
    word_model.load_state_dict(word_ckpt["model"])
    word_model.eval()
    print(f"Words model: {word_ckpt.get('val_acc', 0):.1f}%")

    results = []

    print("\n--- NUMBERS ---")
    test_dir = "/data/testing_v2/Numbers"
    if os.path.exists(test_dir):
        for class_name in sorted(os.listdir(test_dir)):
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fn in sorted(os.listdir(class_dir)):
                if not fn.endswith(".npy"):
                    continue
                data = np.load(os.path.join(class_dir, fn))
                print(f"\n{class_name}/{fn}: {data.shape}")

                if data.shape[0] < 5:
                    print("  FAILED: Too few frames")
                    results.append({"gt": class_name, "ok": False, "t": "n"})
                    continue

                inp = preprocess(data, 90).to(device)
                with torch.no_grad():
                    out = num_model(inp)
                    prob = torch.softmax(out, dim=1)
                    pidx = out.argmax(1).item()
                    pred = NUMBER_CLASSES[pidx]
                    conf = prob[0, pidx].item()

                ok = pred == class_name
                print(f"  GT: {class_name} | Pred: {pred} ({conf*100:.1f}%) | {'OK' if ok else 'WRONG'}")
                results.append({"gt": class_name, "pred": pred, "ok": ok, "t": "n"})
    else:
        print("  No test data found!")

    print("\n--- WORDS ---")
    test_dir = "/data/testing_v2/Words"
    if os.path.exists(test_dir):
        for class_name in sorted(os.listdir(test_dir)):
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fn in sorted(os.listdir(class_dir)):
                if not fn.endswith(".npy"):
                    continue
                data = np.load(os.path.join(class_dir, fn))
                print(f"\n{class_name}/{fn}: {data.shape}")

                if data.shape[0] < 5:
                    print("  FAILED: Too few frames")
                    results.append({"gt": class_name, "ok": False, "t": "w"})
                    continue

                inp = preprocess(data, 90).to(device)
                with torch.no_grad():
                    out = word_model(inp)
                    prob = torch.softmax(out, dim=1)
                    pidx = out.argmax(1).item()
                    pred = WORD_CLASSES[pidx]
                    conf = prob[0, pidx].item()

                ok = pred == class_name
                print(f"  GT: {class_name} | Pred: {pred} ({conf*100:.1f}%) | {'OK' if ok else 'WRONG'}")
                results.append({"gt": class_name, "pred": pred, "ok": ok, "t": "w"})
    else:
        print("  No test data found!")

    print("\n" + "=" * 70)
    nums = [r for r in results if r["t"] == "n"]
    wrds = [r for r in results if r["t"] == "w"]
    nok = sum(r["ok"] for r in nums) if nums else 0
    wok = sum(r["ok"] for r in wrds) if wrds else 0
    print(f"Numbers: {nok}/{len(nums)} ({100*nok/len(nums) if nums else 0:.1f}%)")
    print(f"Words: {wok}/{len(wrds)} ({100*wok/len(wrds) if wrds else 0:.1f}%)")
    print(f"Overall: {nok+wok}/{len(results)} ({100*(nok+wok)/len(results) if results else 0:.1f}%)")

    return {"numbers": 100*nok/len(nums) if nums else 0, "words": 100*wok/len(wrds) if wrds else 0}


@app.local_entrypoint()
def main():
    r = test_v13.remote()
    print(f"\nNumbers: {r['numbers']:.1f}% | Words: {r['words']:.1f}%")
