
# Part 2 - appended to train_ksl_v29.py


# ---------------------------------------------------------------------------
# LOSO (Leave-One-Signer-Out) Cross-Validation
# ---------------------------------------------------------------------------

def collect_all_samples(data_dir, classes):
    """Collect all sample paths and labels from a data directory."""
    c2i = {c: i for i, c in enumerate(classes)}
    all_paths = []
    all_labels = []
    for cn in classes:
        cd = os.path.join(data_dir, cn)
        if os.path.exists(cd):
            for fn in sorted(os.listdir(cd)):
                if fn.endswith(".npy"):
                    all_paths.append(os.path.join(cd, fn))
                    all_labels.append(c2i[cn])
    return all_paths, all_labels


def run_loso(name, classes, config, train_dir, val_dir, ckpt_dir, results_dir, device):
    """
    Run leave-one-signer-out cross-validation.
    Combines train+val data, deduplicates, then holds out one signer per fold.
    After dedup, signers 2 & 3 collapse, giving ~4 unique signers for numbers.
    """
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] LOSO Cross-Validation: {name} ({len(classes)} classes) - v29")
    print(f"[{ts()}] {'=' * 70}")

    # Collect all data from both train and val dirs
    train_paths, train_labels = collect_all_samples(train_dir, classes)
    val_paths, val_labels = collect_all_samples(val_dir, classes)
    all_paths = train_paths + val_paths
    all_labels = train_labels + val_labels

    print(f"[{ts()}] Total samples before dedup: {len(all_paths)}")

    # Deduplicate
    unique_paths, removed = deduplicate_signer_groups(all_paths)
    if removed:
        print(f"[{ts()}] Deduplication: {len(all_paths)} -> {len(unique_paths)} "
              f"({len(removed)} removed)")

    unique_set = set(unique_paths)
    dedup_paths = []
    dedup_labels = []
    for path, label in zip(all_paths, all_labels):
        if path in unique_set:
            dedup_paths.append(path)
            dedup_labels.append(label)
            unique_set.discard(path)

    # Identify unique signers
    signer_ids = [extract_signer_id(os.path.basename(p)) for p in dedup_paths]
    unique_signers = sorted(set(signer_ids))
    print(f"[{ts()}] Unique signers after dedup: {unique_signers}")
    print(f"[{ts()}] Samples per signer: "
          f"{dict(Counter(signer_ids))}")

    fold_results = []
    for fold_idx, held_out_signer in enumerate(unique_signers):
        print(f"\n[{ts()}] --- LOSO Fold {fold_idx + 1}/{len(unique_signers)}: "
              f"Hold out signer {held_out_signer} ---")

        # Split into train/val for this fold
        fold_train_paths = []
        fold_train_labels = []
        fold_val_paths = []
        fold_val_labels = []

        for path, label, signer in zip(dedup_paths, dedup_labels, signer_ids):
            if signer == held_out_signer:
                fold_val_paths.append(path)
                fold_val_labels.append(label)
            else:
                fold_train_paths.append(path)
                fold_train_labels.append(label)

        print(f"[{ts()}] Train: {len(fold_train_paths)}, Val: {len(fold_val_paths)}")

        # Create datasets directly from paths (no dedup needed, already done)
        fold_ckpt_dir = os.path.join(ckpt_dir, f"loso_fold{fold_idx}")
        fold_result = train_split_from_paths(
            f"{name}_Fold{fold_idx+1}(held={held_out_signer})",
            classes, config,
            fold_train_paths, fold_train_labels,
            fold_val_paths, fold_val_labels,
            fold_ckpt_dir, device,
        )

        if fold_result:
            fold_result["held_out_signer"] = held_out_signer
            fold_results.append(fold_result)
            print(f"[{ts()}] Fold {fold_idx + 1} result: {fold_result['overall']:.1f}%")

    # Summary
    if fold_results:
        accs = [r["overall"] for r in fold_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"\n[{ts()}] LOSO Summary for {name}:")
        for r in fold_results:
            print(f"[{ts()}]   Signer {r['held_out_signer']} held out: {r['overall']:.1f}%")
        print(f"[{ts()}]   Mean: {mean_acc:.1f}% +/- {std_acc:.1f}%")
        return {
            "folds": fold_results,
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
        }
    return None


def train_split_from_paths(name, classes, config,
                            train_paths, train_labels,
                            val_paths, val_labels,
                            ckpt_dir, device):
    """
    Train one split from pre-split path lists (for LOSO).
    Similar to train_split but takes paths directly instead of directories.
    v29: Uses KSLGraphNetV29 with EMA support.
    """
    adj = build_adj(config["num_nodes"]).to(device)

    # Create datasets from paths
    train_ds = KSLGraphDatasetFromPaths(train_paths, train_labels, classes, config, aug=True)
    val_ds = KSLGraphDatasetFromPaths(val_paths, val_labels, classes, config, aug=False)

    if len(train_ds) == 0:
        print(f"[{ts()}] ERROR: No training samples")
        return None

    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {v: k for k, v in c2i.items()}
    num_signers = train_ds.num_signers

    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES

    train_sampler = SignerBalancedSampler(
        train_ds.labels, train_ds.signer_labels,
        batch_size=config["batch_size"], drop_last=True,
    )

    train_ld = DataLoader(
        train_ds, batch_size=config["batch_size"],
        sampler=train_sampler, num_workers=2, pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = KSLGraphNetV29(
        nc=len(classes),
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=config["num_nodes"],
        ic=config["in_channels"],
        adj=adj,
        dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1),
    ).to(device)

    # EMA
    ema = EMA(model, decay=config.get("ema_decay", 0.9998))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Parameters: {param_count:,}, signers: {num_signers}")

    supcon_loss_fn = SupConLoss(temperature=config["supcon_temperature"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config["epochs"], eta_min=config["min_lr"]
    )
    warmup_epochs = config["warmup_epochs"]

    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    best_ema_path = os.path.join(ckpt_dir, "best_model_ema.pt")

    best, best_ema, patience_counter = 0.0, 0.0, 0
    mixup_alpha = config.get("mixup_alpha", 0)
    cutmix_alpha = config.get("cutmix_alpha", 1.0)
    cutmix_prob = config.get("cutmix_prob", 0.3)
    supcon_weight = config["supcon_weight"]

    for ep in range(config["epochs"]):
        model.train()
        tl, tc, tt = 0.0, 0, 0

        if ep < config["grl_start_epoch"]:
            grl_lambda = 0.0
        else:
            progress = min(1.0, (ep - config["grl_start_epoch"]) / config["grl_ramp_epochs"])
            grl_lambda = config["grl_lambda_max"] * progress

        if ep < warmup_epochs:
            warmup_factor = (ep + 1) / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = config["learning_rate"] * warmup_factor

        for gcn_data, aux_data, targets, signer_targets in train_ld:
            gcn_data = gcn_data.to(device)
            aux_data = aux_data.to(device)
            targets = targets.to(device)
            signer_targets = signer_targets.to(device)
            opt.zero_grad()

            use_cutmix = np.random.random() < cutmix_prob
            use_mixup = (not use_cutmix) and (mixup_alpha > 0)

            if use_cutmix:
                gcn_mixed, aux_mixed, targets_a, targets_b, lam = temporal_cutmix(
                    gcn_data, aux_data, targets, signer_targets, alpha=cutmix_alpha,
                )
                logits, signer_logits, embedding = model(gcn_mixed, aux_mixed, grl_lambda=grl_lambda)
                cls_loss = (
                    lam * F.cross_entropy(logits, targets_a, label_smoothing=config["label_smoothing"])
                    + (1 - lam) * F.cross_entropy(logits, targets_b, label_smoothing=config["label_smoothing"])
                )
                sc_loss = supcon_loss_fn(embedding, targets_a)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += (lam * p.eq(targets_a).float() + (1 - lam) * p.eq(targets_b).float()).sum().item()
            elif use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(gcn_data.size(0), device=device)
                gcn_mixed = lam * gcn_data + (1 - lam) * gcn_data[perm]
                aux_mixed = lam * aux_data + (1 - lam) * aux_data[perm]
                t_perm = targets[perm]
                logits, signer_logits, embedding = model(gcn_mixed, aux_mixed, grl_lambda=grl_lambda)
                cls_loss = (
                    lam * F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                    + (1 - lam) * F.cross_entropy(logits, t_perm, label_smoothing=config["label_smoothing"])
                )
                sc_loss = supcon_loss_fn(embedding, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += (lam * p.eq(targets).float() + (1 - lam) * p.eq(t_perm).float()).sum().item()
            else:
                logits, signer_logits, embedding = model(gcn_data, aux_data, grl_lambda=grl_lambda)
                cls_loss = F.cross_entropy(logits, targets, label_smoothing=config["label_smoothing"])
                sc_loss = supcon_loss_fn(embedding, targets)
                signer_loss = F.cross_entropy(signer_logits, signer_targets)
                loss = cls_loss + supcon_weight * sc_loss + grl_lambda * signer_loss
                _, p = logits.max(1)
                tc += p.eq(targets).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update()
            tl += loss.item()
            tt += targets.size(0)

        if ep >= warmup_epochs:
            scheduler.step()

        # Validation (regular model)
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for gcn_data, aux_data, targets, _ in val_ld:
                gcn_data = gcn_data.to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)
                logits, _, _ = model(gcn_data, aux_data, grl_lambda=0.0)
                _, p = logits.max(1)
                vt += targets.size(0)
                vc += p.eq(targets).sum().item()

        va = 100.0 * vc / vt if vt > 0 else 0.0

        if va > best:
            best, patience_counter = va, 0
            torch.save({
                "model": model.state_dict(), "val_acc": va, "epoch": ep + 1,
                "classes": classes, "num_nodes": config["num_nodes"],
                "num_signers": num_signers, "aux_dim": aux_dim,
                "version": "v29", "config": config,
            }, best_path)
        else:
            patience_counter += 1

        # Validation (EMA model)
        ema.apply_shadow()
        vc_ema, vt_ema = 0, 0
        with torch.no_grad():
            for gcn_data, aux_data, targets, _ in val_ld:
                gcn_data = gcn_data.to(device)
                aux_data = aux_data.to(device)
                targets = targets.to(device)
                logits, _, _ = model(gcn_data, aux_data, grl_lambda=0.0)
                _, p = logits.max(1)
                vt_ema += targets.size(0)
                vc_ema += p.eq(targets).sum().item()

        va_ema = 100.0 * vc_ema / vt_ema if vt_ema > 0 else 0.0

        if va_ema > best_ema:
            best_ema = va_ema
            torch.save({
                "model": model.state_dict(), "val_acc": va_ema, "epoch": ep + 1,
                "classes": classes, "num_nodes": config["num_nodes"],
                "num_signers": num_signers, "aux_dim": aux_dim,
                "version": "v29_ema", "config": config,
            }, best_ema_path)

        ema.restore()

        if patience_counter >= config["patience"]:
            break

        if ep < 5 or (ep + 1) % 20 == 0:
            ta = 100.0 * tc / tt if tt > 0 else 0.0
            print(f"[{ts()}]   Ep {ep+1:3d} | Loss: {tl/max(len(train_ld),1):.4f} | "
                  f"Train: {ta:.1f}% | Val: {va:.1f}% | EMA: {va_ema:.1f}%")

    # Final evaluation with best model (use EMA if it's better)
    if best_ema >= best and os.path.isfile(best_ema_path):
        ckpt = torch.load(best_ema_path, map_location=device, weights_only=False)
        print(f"[{ts()}] Using EMA checkpoint (EMA {best_ema:.1f}% >= regular {best:.1f}%)")
    else:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        print(f"[{ts()}] Using regular checkpoint (regular {best:.1f}% > EMA {best_ema:.1f}%)")
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for gcn_data, aux_data, targets, _ in val_ld:
            logits, _, _ = model(gcn_data.to(device), aux_data.to(device), grl_lambda=0.0)
            _, p = logits.max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(targets.cpu().numpy())

    ov = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if len(tgts) > 0 else 0.0

    return {
        "overall": ov,
        "best_epoch": ckpt["epoch"],
        "params": param_count,
    }


class KSLGraphDatasetFromPaths(Dataset):
    """Dataset that takes pre-split paths directly (for LOSO mode)."""

    def __init__(self, sample_paths, labels, classes, config, aug=False):
        self.samples = sample_paths
        self.labels = labels
        self.mf = config["max_frames"]
        self.aug = aug
        self.config = config
        self.c2i = {c: i for i, c in enumerate(classes)}

        # Build signer label mapping
        self.signer_to_idx = {}
        self.signer_labels = []
        for path in self.samples:
            signer = extract_signer_id(os.path.basename(path))
            if signer not in self.signer_to_idx:
                self.signer_to_idx[signer] = len(self.signer_to_idx)
            self.signer_labels.append(self.signer_to_idx[signer])

        self.num_signers = len(self.signer_to_idx)
        print(f"[{ts()}]   LOSO dataset: {len(self.samples)} samples, "
              f"{self.num_signers} signers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = np.load(self.samples[idx])
        f = d.shape[0]

        if d.shape[1] >= 225:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            for pi, idx_pose in enumerate(POSE_INDICES):
                start = idx_pose * 3
                pose[:, pi, :] = d[:, start:start + 3]
            lh = d[:, 99:162].reshape(f, 21, 3)
            rh = d[:, 162:225].reshape(f, 21, 3)
        else:
            pose = np.zeros((f, 6, 3), dtype=np.float32)
            lh = np.zeros((f, 21, 3), dtype=np.float32)
            rh = np.zeros((f, 21, 3), dtype=np.float32)

        h = np.concatenate([lh, rh, pose], axis=1).astype(np.float32)

        if self.aug and np.random.random() < self.config["hand_dropout_prob"]:
            dropout_rate = np.random.uniform(
                self.config["hand_dropout_min"], self.config["hand_dropout_max"])
            if np.random.random() < 0.6:
                lh_mask = np.random.random(f) > dropout_rate
                h[~lh_mask, :21, :] = 0
            if np.random.random() < 0.6:
                rh_mask = np.random.random(f) > dropout_rate
                h[~rh_mask, 21:42, :] = 0

        if self.aug and np.random.random() < self.config["complete_hand_drop_prob"]:
            if np.random.random() < 0.5:
                h[:, :21, :] = 0
            else:
                h[:, 21:42, :] = 0

        if self.aug and np.random.random() < 0.3:
            h_copy = h.copy()
            h[:, :21, :] = h_copy[:, 21:42, :]
            h[:, 21:42, :] = h_copy[:, :21, :]
            h[:, :42, 0] = -h[:, :42, 0]
            h[:, 42:48, 0] = -h[:, 42:48, 0]

        hand_body_feats = compute_hand_body_features(h)
        h = normalize_wrist_palm(h)

        if self.aug and np.random.random() < self.config["bone_perturb_prob"]:
            h = augment_bone_length_perturbation(h, LH_CHAINS + RH_CHAINS,
                                                  scale_range=self.config["bone_perturb_range"])
        if self.aug and np.random.random() < self.config["hand_size_prob"]:
            h = augment_hand_size(h, scale_range=self.config["hand_size_range"])
        if self.aug and np.random.random() < self.config["rotation_prob"]:
            h = augment_rotation(h, max_deg=self.config["rotation_max_deg"])
        if self.aug and np.random.random() < self.config["shear_prob"]:
            h = augment_shear(h, max_shear=self.config["shear_max"])
        if self.aug and np.random.random() < self.config["joint_dropout_prob"]:
            h = augment_joint_dropout(h, dropout_rate=self.config["joint_dropout_rate"])
        if self.aug and np.random.random() < 0.5:
            h = h * np.random.uniform(0.8, 1.2)
        if self.aug and np.random.random() < self.config["noise_prob"]:
            noise = np.random.normal(0, self.config["noise_std"], h.shape).astype(np.float32)
            h = h + noise

        velocity = np.zeros_like(h)
        velocity[1:] = h[1:] - h[:-1]
        bones = compute_bones(h)
        joint_angles = compute_joint_angles(h)
        fingertip_dists = compute_fingertip_distances(h)

        if self.aug and np.random.random() < self.config["temporal_warp_prob"]:
            h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats = augment_temporal_warp(
                [h, velocity, bones, joint_angles, fingertip_dists, hand_body_feats],
                sigma=self.config["temporal_warp_sigma"])

        f = h.shape[0]
        if f >= self.mf:
            indices = np.linspace(0, f - 1, self.mf, dtype=int)
            h = h[indices]; velocity = velocity[indices]; bones = bones[indices]
            joint_angles = joint_angles[indices]; fingertip_dists = fingertip_dists[indices]
            hand_body_feats = hand_body_feats[indices]
        else:
            pad_len = self.mf - f
            h = np.concatenate([h, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            velocity = np.concatenate([velocity, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            bones = np.concatenate([bones, np.zeros((pad_len, 48, 3), dtype=np.float32)])
            joint_angles = np.concatenate([joint_angles, np.zeros((pad_len, NUM_ANGLE_FEATURES), dtype=np.float32)])
            fingertip_dists = np.concatenate([fingertip_dists, np.zeros((pad_len, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)])
            hand_body_feats = np.concatenate([hand_body_feats, np.zeros((pad_len, NUM_HAND_BODY_FEATURES), dtype=np.float32)])

        if self.aug and np.random.random() < 0.3:
            shift = np.random.randint(-8, 9)
            if shift > 0:
                z3 = np.zeros((shift, 48, 3), dtype=np.float32)
                h = np.concatenate([z3, h[:-shift]], axis=0)
                velocity = np.concatenate([z3, velocity[:-shift]], axis=0)
                bones = np.concatenate([z3, bones[:-shift]], axis=0)
                za = np.zeros((shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((shift, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                joint_angles = np.concatenate([za, joint_angles[:-shift]], axis=0)
                fingertip_dists = np.concatenate([zd, fingertip_dists[:-shift]], axis=0)
                hand_body_feats = np.concatenate([zh, hand_body_feats[:-shift]], axis=0)
            elif shift < 0:
                z3 = np.zeros((-shift, 48, 3), dtype=np.float32)
                h = np.concatenate([h[-shift:], z3], axis=0)
                velocity = np.concatenate([velocity[-shift:], z3], axis=0)
                bones = np.concatenate([bones[-shift:], z3], axis=0)
                za = np.zeros((-shift, NUM_ANGLE_FEATURES), dtype=np.float32)
                zd = np.zeros((-shift, 2 * NUM_FINGERTIP_PAIRS), dtype=np.float32)
                zh = np.zeros((-shift, NUM_HAND_BODY_FEATURES), dtype=np.float32)
                joint_angles = np.concatenate([joint_angles[-shift:], za], axis=0)
                fingertip_dists = np.concatenate([fingertip_dists[-shift:], zd], axis=0)
                hand_body_feats = np.concatenate([hand_body_feats[-shift:], zh], axis=0)

        gcn_features = np.concatenate([h, velocity, bones], axis=2)
        gcn_features = np.clip(gcn_features, -10, 10).astype(np.float32)
        aux_features = np.concatenate([joint_angles, fingertip_dists, hand_body_feats], axis=1)
        aux_features = np.clip(aux_features, -10, 10).astype(np.float32)

        gcn_tensor = torch.FloatTensor(gcn_features).permute(2, 0, 1)
        aux_tensor = torch.FloatTensor(aux_features)

        return gcn_tensor, aux_tensor, self.labels[idx], self.signer_labels[idx]


# ---------------------------------------------------------------------------
# Test Set Evaluation (held-out signer evaluation)
# ---------------------------------------------------------------------------

def evaluate_test_set(name, classes, config, test_dir, ckpt_path, device):
    """
    Evaluate a trained model on the held-out test set (ksl-alpha signers 13-15).
    Returns accuracy dict or None if test dir doesn't exist.
    """
    if not os.path.isdir(test_dir):
        print(f"[{ts()}] Test dir not found: {test_dir}, skipping test evaluation for {name}")
        return None

    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] Test Set Evaluation: {name} ({len(classes)} classes) - v29")
    print(f"[{ts()}] Test dir: {test_dir}")
    print(f"[{ts()}] {'=' * 70}")

    # Load test data (no augmentation)
    test_ds = KSLGraphDataset(test_dir, classes, config, aug=False)
    if len(test_ds) == 0:
        print(f"[{ts()}] WARNING: No test samples found in {test_dir}")
        return None

    test_ld = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load trained model
    if not os.path.isfile(ckpt_path):
        print(f"[{ts()}] WARNING: Checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    adj = build_adj(config["num_nodes"]).to(device)
    aux_dim = NUM_ANGLE_FEATURES + 2 * NUM_FINGERTIP_PAIRS + NUM_HAND_BODY_FEATURES
    num_signers = ckpt.get("num_signers", test_ds.num_signers)

    model = KSLGraphNetV29(
        nc=len(classes),
        num_signers=num_signers,
        aux_dim=aux_dim,
        nn_=config["num_nodes"],
        ic=config["in_channels"],
        adj=adj,
        dr=config["dropout"],
        spatial_dropout=config.get("spatial_dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {v: k for k, v in c2i.items()}

    preds, tgts = [], []
    with torch.no_grad():
        for gcn_data, aux_data, targets, _ in test_ld:
            logits, _, _ = model(
                gcn_data.to(device), aux_data.to(device), grl_lambda=0.0
            )
            _, p = logits.max(1)
            preds.extend(p.cpu().numpy())
            tgts.extend(targets.cpu().numpy())

    # Per-class results
    print(f"\n[{ts()}] {name} Test Set Per-Class Results:")
    per_class = {}
    for i in range(len(classes)):
        cn = i2c[i]
        tot_cls = sum(1 for t in tgts if t == i)
        cor = sum(1 for t, p in zip(tgts, preds) if t == i and t == p)
        per_class[cn] = 100.0 * cor / tot_cls if tot_cls > 0 else 0.0
        print(f"[{ts()}]   {cn:12s}: {per_class[cn]:5.1f}% ({cor}/{tot_cls})")

    overall = 100.0 * sum(1 for t, p in zip(tgts, preds) if t == p) / len(tgts) if len(tgts) > 0 else 0.0
    print(f"[{ts()}] {name} Test Overall: {overall:.1f}% ({sum(1 for t,p in zip(tgts,preds) if t==p)}/{len(tgts)})")

    # Per-signer accuracy
    signer_preds = defaultdict(list)
    signer_tgts = defaultdict(list)
    for idx, path in enumerate(test_ds.samples):
        signer = extract_signer_id(os.path.basename(path))
        if idx < len(preds):
            signer_preds[signer].append(preds[idx])
            signer_tgts[signer].append(tgts[idx])

    print(f"[{ts()}] {name} Test Per-Signer Results:")
    per_signer = {}
    for signer in sorted(signer_preds.keys()):
        sp = signer_preds[signer]
        st = signer_tgts[signer]
        acc = 100.0 * sum(1 for t, p in zip(st, sp) if t == p) / len(st) if len(st) > 0 else 0.0
        per_signer[signer] = acc
        print(f"[{ts()}]   Signer {signer}: {acc:.1f}% ({sum(1 for t,p in zip(st,sp) if t==p)}/{len(st)})")

    # Confusion matrix
    print(f"\n[{ts()}] {name} Test Confusion Matrix:")
    nc = len(classes)
    cm = [[0] * nc for _ in range(nc)]
    for t, p in zip(tgts, preds):
        cm[t][p] += 1

    hdr = "True\\Pred  " + " ".join(f"{i2c[j]:>5s}" for j in range(nc))
    print(f"[{ts()}] {hdr}")
    for i in range(nc):
        row_str = f"{i2c[i]:>9s}  " + " ".join(f"{cm[i][j]:5d}" for j in range(nc))
        print(f"[{ts()}] {row_str}")

    return {
        "overall": overall,
        "per_class": per_class,
        "per_signer": per_signer,
        "num_test_samples": len(tgts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KSL Training v29 - Deeper+Wider+EMA+FixedEpochs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type", type=str, default="both",
        choices=["numbers", "words", "both"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loso", action="store_true",
                        help="Run leave-one-signer-out cross-validation instead of normal training")

    base_data = "/scratch/alpine/hama5612/ksl-dir-2/data"
    parser.add_argument("--train-dir", type=str, default=os.path.join(base_data, "train_alpha"))
    parser.add_argument("--val-dir", type=str, default=os.path.join(base_data, "val_alpha"))
    parser.add_argument("--test-dir", type=str, default=os.path.join(base_data, "test_alpha"),
                        help="Held-out test set directory (ksl-alpha signers 13-15)")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(base_data, "checkpoints"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(base_data, "results"))

    args = parser.parse_args()

    set_seed(args.seed)

    mode_str = "LOSO" if args.loso else "Normal"
    print("=" * 70)
    print(f"KSL Training v29 - Deeper+Wider+EMA+FixedEpochs [{mode_str}]")
    print(f"Started: {ts()}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    check_mediapipe_version()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # v29: Train on 12 signers (train_alpha + val_alpha), validate on 3 (test_alpha)
    train_dirs = [args.train_dir, args.val_dir]  # signers 1-12
    val_dir = args.test_dir                       # signers 13-15

    print(f"\nv29 Data Split:")
    print(f"  Train dirs: {train_dirs}  (signers 1-12, ~895 samples)")
    print(f"  Val dir:    {val_dir}  (signers 13-15, ~225 samples)")
    print(f"  Ckpt dir:   {args.checkpoint_dir}")

    print(f"\n[{ts()}] v29 Config:")
    print(json.dumps(V29_CONFIG, indent=2, default=str))

    results = {}
    start_time = time.time()

    if args.loso:
        # LOSO cross-validation mode (uses original train/val dirs)
        if args.model_type in ("numbers", "both"):
            ckpt_dir = os.path.join(args.checkpoint_dir, "v29_loso_numbers")
            results["numbers_loso"] = run_loso(
                "Numbers", NUMBER_CLASSES, V29_CONFIG,
                args.train_dir, args.val_dir, ckpt_dir, args.results_dir, device,
            )

        if args.model_type in ("words", "both"):
            ckpt_dir = os.path.join(args.checkpoint_dir, "v29_loso_words")
            results["words_loso"] = run_loso(
                "Words", WORD_CLASSES, V29_CONFIG,
                args.train_dir, args.val_dir, ckpt_dir, args.results_dir, device,
            )
    else:
        # Normal training mode — v29: 12 signers train, 3 val, no focal loss
        if args.model_type in ("numbers", "both"):
            ckpt_dir = os.path.join(args.checkpoint_dir, "v29_numbers")
            results["numbers"] = train_split(
                "Numbers", NUMBER_CLASSES, V29_CONFIG,
                train_dirs, val_dir, ckpt_dir, device,
                use_focal=False,
            )
            # Evaluate on test set after training
            if results["numbers"]:
                best_path = os.path.join(ckpt_dir, "best_model.pt")
                best_ema_path = os.path.join(ckpt_dir, "best_model_ema.pt")
                # Prefer EMA checkpoint if it exists
                eval_path = best_ema_path if os.path.isfile(best_ema_path) else best_path
                results["numbers_test"] = evaluate_test_set(
                    "Numbers", NUMBER_CLASSES, V29_CONFIG,
                    args.test_dir, eval_path, device,
                )

        if args.model_type in ("words", "both"):
            ckpt_dir = os.path.join(args.checkpoint_dir, "v29_words")
            results["words"] = train_split(
                "Words", WORD_CLASSES, V29_CONFIG,
                train_dirs, val_dir, ckpt_dir, device,
                use_focal=False,
            )
            # Evaluate on test set after training
            if results["words"]:
                best_path = os.path.join(ckpt_dir, "best_model.pt")
                best_ema_path = os.path.join(ckpt_dir, "best_model_ema.pt")
                # Prefer EMA checkpoint if it exists
                eval_path = best_ema_path if os.path.isfile(best_ema_path) else best_path
                results["words_test"] = evaluate_test_set(
                    "Words", WORD_CLASSES, V29_CONFIG,
                    args.test_dir, eval_path, device,
                )

    total_time = time.time() - start_time

    # Summary
    print(f"\n[{ts()}] {'=' * 70}")
    print(f"[{ts()}] SUMMARY v29 [{mode_str}]")
    print(f"[{ts()}] {'=' * 70}")

    if args.loso:
        if results.get("numbers_loso"):
            r = results["numbers_loso"]
            print(f"[{ts()}] Numbers LOSO: {r['mean_accuracy']:.1f}% +/- {r['std_accuracy']:.1f}%")
        if results.get("words_loso"):
            r = results["words_loso"]
            print(f"[{ts()}] Words LOSO:   {r['mean_accuracy']:.1f}% +/- {r['std_accuracy']:.1f}%")
        if results.get("numbers_loso") and results.get("words_loso"):
            combined = (results["numbers_loso"]["mean_accuracy"] + results["words_loso"]["mean_accuracy"]) / 2
            print(f"[{ts()}] Combined LOSO: {combined:.1f}%")
    else:
        if results.get("numbers"):
            r = results["numbers"]
            print(f"[{ts()}] Numbers: {r['overall']:.1f}% "
                  f"(best epoch {r['best_epoch']}, params {r['params']:,})")
        if results.get("words"):
            r = results["words"]
            print(f"[{ts()}] Words:   {r['overall']:.1f}% "
                  f"(best epoch {r['best_epoch']}, params {r['params']:,})")
        if results.get("numbers") and results.get("words"):
            combined = (results["numbers"]["overall"] + results["words"]["overall"]) / 2
            print(f"[{ts()}] Combined: {combined:.1f}%")
        # Test set results
        if results.get("numbers_test"):
            r = results["numbers_test"]
            print(f"[{ts()}] Numbers Test: {r['overall']:.1f}% ({r['num_test_samples']} samples)")
        if results.get("words_test"):
            r = results["words_test"]
            print(f"[{ts()}] Words Test:   {r['overall']:.1f}% ({r['num_test_samples']} samples)")
        if results.get("numbers_test") and results.get("words_test"):
            combined_test = (results["numbers_test"]["overall"] + results["words_test"]["overall"]) / 2
            print(f"[{ts()}] Combined Test: {combined_test:.1f}%")

    print(f"[{ts()}] Total training time: {total_time / 60:.1f} minutes")

    # Save results JSON
    os.makedirs(args.results_dir, exist_ok=True)
    loso_tag = "_loso" if args.loso else ""
    results_path = os.path.join(
        args.results_dir,
        f"v29{loso_tag}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    results_out = {
        "version": "v29",
        "mode": mode_str,
        "model_type": args.model_type,
        "seed": args.seed,
        "config": V29_CONFIG,
        "results": results,
        "total_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": ts(),
        "changes_from_v27": [
            "1. 8 GCN LAYERS (up from 4): Channel progression ic->64->64->128->...->128.",
            "2. DILATED MULTI-SCALE TCN: kernel=3 with dilations=(1,2,4) replaces kernels=(3,5,7).",
            "3. 4 ATTENTION HEADS (up from 2): gcn_embed_dim = 4 * 128 = 512.",
            "4. WIDER AUX BRANCH: 128-dim output (was 64-dim).",
            "5. WIDER CLASSIFIER: 640->128->nc (was 320->64->nc).",
            "6. 300 FIXED EPOCHS with EMA, no early stopping.",
            "7. REDUCED REGULARIZATION: dropout 0.2, GRL 0.15, CutMix 0.15.",
            "8. EMA (decay=0.9998): Saves both regular and EMA checkpoints.",
            "9. AdamW with weight_decay=0.05.",
            "10. NO FOCAL LOSS for either numbers or words.",
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"[{ts()}] Results saved to {results_path}")

    print(f"[{ts()}] Done.")
    return results


if __name__ == "__main__":
    main()
