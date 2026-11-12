"""Public-only initialization and explicit held-out evaluation of frozen runs."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import ImageDataset, validate_manifest
from .io import read_json, atomic_json, atomic_torch, digest, file_hash, run_lock
from .model import CropCNN, evaluate
from .training import seed_everything


def public_warmup(config, output, epochs=100, patience=20):
    output = Path(output)
    with run_lock(output):
        if (output / "public_init.pt").exists():
            raise FileExistsError("Public initialization already exists; use a new output")
        manifest = read_json(config["manifest"])
        validate_manifest(manifest, verify_files=True)
        public = [r for r in manifest["rows"] if r["split"] == "public"]
        validation = ImageDataset([r for r in manifest["rows"] if r["split"] == "validation"])
        if not public or not len(validation):
            raise ValueError("Public training and validation sets are required")
        seed_everything(config["seed"])
        torch.set_num_threads(4)
        model = CropCNN(normalize=config.get("normalize", False)).to(config["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        loader = DataLoader(ImageDataset(public, augment=True), batch_size=32, shuffle=True)
        best_score, best, bad, history = -1, None, 0, []
        for epoch in range(epochs):
            model.train()
            for images, labels in loader:
                optimizer.zero_grad()
                logits = model(images.to(config["device"]))
                torch.nn.functional.cross_entropy(logits, labels.to(config["device"])).backward()
                optimizer.step()
            score = evaluate(model, validation, config["device"])["macro_f1"]
            history.append({"epoch": epoch + 1, "validation_f1": score})
            if score > best_score:
                best_score, bad = score, 0
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"Public epoch {epoch + 1}: validation F1={score:.4f}, best={best_score:.4f}", flush=True)
            if epoch + 1 >= 30 and bad >= patience:
                break
        provenance = {"manifest_hash": manifest["manifest_hash"], "seed": config["seed"],
                      "normalize": config.get("normalize", False), "public_samples": len(public),
                      "public_rows_hash": digest(public), "best_validation_f1": best_score,
                      "history": history, "test_evaluated": False,
                      "training_scope": "public split only; selection on validation; no private images"}
        atomic_torch(output / "public_init.pt", {"state_dict": best, "provenance": provenance})
        atomic_json(output / "summary.json", provenance)
        return provenance


def load_public_init(path, manifest, normalize):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    provenance = checkpoint["provenance"]
    public = [r for r in manifest["rows"] if r["split"] == "public"]
    if (provenance["manifest_hash"] != manifest["manifest_hash"] or
            provenance["public_rows_hash"] != digest(public) or provenance["normalize"] != normalize):
        raise ValueError("Public initialization does not match dataset or preprocessing")
    return checkpoint


def evaluate_run(run_dir):
    """Separate final test artifact; training config and privacy ledger stay immutable."""
    run_dir = Path(run_dir)
    with run_lock(run_dir):
        config = read_json(run_dir / "config.json")
        manifest = read_json(config["manifest"])
        validate_manifest(manifest, verify_files=True)
        state = torch.load(run_dir / "state.pt", map_location="cpu", weights_only=False)
        if not state["finished"] or state["config_hash"] != digest({"config": config, "manifest": manifest["manifest_hash"]}):
            raise ValueError("Final evaluation requires a completed, matching run")
        path = run_dir / "final_test.json"
        checkpoint_hash = file_hash(run_dir / "best.pt")
        if path.exists():
            existing = read_json(path)
            if existing["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Previously evaluated checkpoint has changed")
            return existing
        model = CropCNN(normalize=config.get("normalize", False))
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
        test = ImageDataset([r for r in manifest["rows"] if r["split"] == "test"])
        torch.set_num_threads(4)
        result = {"config_hash": state["config_hash"], "manifest_hash": manifest["manifest_hash"],
                  "checkpoint_sha256": checkpoint_hash, "test": evaluate(model, test, config["device"])}
        if config.get("public_checkpoint"):
            if state.get("public_checkpoint_sha256") is not None and state["public_checkpoint_sha256"] != file_hash(config["public_checkpoint"]):
                raise ValueError("Public initialization changed after training")
            initial = load_public_init(config["public_checkpoint"], manifest, config.get("normalize", False))
            model.load_state_dict(initial["state_dict"])
            result["public_only_test"] = evaluate(model, test, config["device"])
            result["public_checkpoint_sha256"] = file_hash(config["public_checkpoint"])
        atomic_json(path, result)
        return result
