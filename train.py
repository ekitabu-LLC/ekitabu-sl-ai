import modal
import os

app = modal.App("ksl-trainer")
volume = modal.Volume.from_name("ksl-dataset-v2")
CHECKPOINT_DIR = "/data/checkpoints"

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")


@app.function(gpu="A10G", volumes={"/data": volume}, timeout=7200, image=image)
def train_task(args):
    name, classes = args
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import Dataset, DataLoader

    print(f"Starting Training: {name}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    class KSLTransformer(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.proj = nn.Linear(225, 128)
            self.pos = nn.Parameter(torch.randn(1, 150, 128))
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=128, nhead=4, dropout=0.3, batch_first=True
                ),
                num_layers=2,
            )
            self.head = nn.Linear(128, num_classes)

        def forward(self, x):
            b, f, _ = x.shape
            x = self.proj(x) + self.pos[:, :f, :]
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.head(x)

    class RemoteDataset(Dataset):
        def __init__(self, root, classes):
            self.samples = []
            self.class_map = {c: i for i, c in enumerate(classes)}

            for c in classes:
                c_dir = os.path.join(root, c)
                if os.path.exists(c_dir):
                    for f in os.listdir(c_dir):
                        self.samples.append((os.path.join(c_dir, f), self.class_map[c]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            data = np.load(path).astype(np.float32)

            if data.shape[0] > 120:
                data = data[:120]
            else:
                data = np.pad(data, ((0, 120 - data.shape[0]), (0, 0)))

            if np.random.rand() > 0.5:
                data += np.random.normal(0, 0.01, data.shape).astype(np.float32)

            return torch.from_numpy(data), label

    ds_train = RemoteDataset("/data/processed", classes)
    ds_val = RemoteDataset("/data/val_processed", classes)

    print(f"Train samples: {len(ds_train)}")
    print(f"Validation samples: {len(ds_val)}")

    train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=32)

    model = KSLTransformer(len(classes)).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0

    for epoch in range(200):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = crit(model(x.cuda()), y.cuda())
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        corr = 0
        total = 0
        per_class_corr = {i: 0 for i in range(len(classes))}
        per_class_total = {i: 0 for i in range(len(classes))}

        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.cuda())
                _, pred = torch.max(out, 1)
                corr += (pred == y.cuda()).sum().item()
                total += y.size(0)

                for i in range(len(classes)):
                    pred_mask = pred == i
                    true_mask = y.cuda() == i
                    per_class_corr[i] += (pred_mask & true_mask).sum().item()
                    per_class_total[i] += true_mask.sum().item()

        val_acc = corr / total if total > 0 else 0

        top_k = 3
        _, top_k_pred = torch.topk(out, top_k)
        top_k_acc = (
            (top_k_pred == y.cuda().unsqueeze(1)).any(dim=1).float().mean().item()
        )

        per_class_accs = []
        for i in range(len(classes)):
            c_acc = (
                per_class_corr[i] / per_class_total[i] if per_class_total[i] > 0 else 0
            )
            per_class_accs.append(f"{classes[i]}: {c_acc * 100:.1f}%")

        print(
            f"Epoch {epoch:3d} | Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc * 100:.2f}% | Top-3: {top_k_acc * 100:.1f}%"
        )

        if len(per_class_accs) <= 10:
            print("  | " + ", ".join(per_class_accs))
        else:
            print(f"  | {classes[0]}: {per_class_accs[0]}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = f"{CHECKPOINT_DIR}/{name}_best.pth"
            torch.save(model.state_dict(), path)
            print(f"  -> New Best Model Saved! (Acc: {val_acc * 100:.2f}%)")

    print(f"\nFinal Best Accuracy: {best_val_acc * 100:.2f}%")

    final_path = f"{CHECKPOINT_DIR}/{name}_final.pth"
    torch.save(model.state_dict(), final_path)
    volume.commit()
    return f"Finished {name}. Best Acc: {best_val_acc * 100:.2f}%"


@app.local_entrypoint()
def main():
    train_classes = [
        "100",
        "125",
        "17",
        "22",
        "268",
        "35",
        "388",
        "444",
        "48",
        "54",
        "66",
        "73",
        "89",
        "9",
        "91",
    ]
    word_classes = [
        "Agreement",
        "Apple",
        "Colour",
        "Friend",
        "Gift",
        "Market",
        "Monday",
        "Picture",
        "Proud",
        "Sweater",
        "Teach",
        "Tomatoes",
        "Tortoise",
        "Twin",
        "Ugali",
    ]

    print("Launching Dual Training with Validation (200 Epochs)...")
    results = list(
        train_task.map([("numbers", train_classes), ("words", word_classes)])
    )
    print(results)
    print("\nTraining Complete!")
    print("To download models:")
    print("  modal volume get ksl-dataset-v2 /data/checkpoints/numbers_best.pth .")
    print("  modal volume get ksl-dataset-v2 /data/checkpoints/words_best.pth .")
