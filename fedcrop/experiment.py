"""Checkpointed experiments with common data, traces, and acceptance evidence."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from .data import ImageDataset, partition, validate_manifest
from .federation import secure_round, plain_round
from .io import read_json, atomic_json, atomic_torch, digest, emit, host_info, run_lock, file_hash
from .model import CropCNN, get_weights, set_weights, evaluate, metrics_from_confusion
from .privacy import PrivacyLedger, calibrate_noise
from .resources import Device, make_devices, workload, select_clients, profile_host
from .training import train_local, seed_everything, sampling_rate


def load_config(path, overrides=None):
    config = read_json(Path(__file__).parent.parent / "configs/main.json")
    config.update(read_json(path))
    config.update(overrides or {})
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if config["mode"] not in {"federated", "centralized", "local"}:
        raise ValueError("Unknown training mode")
    if config["secure_aggregation"] and (not config["dp"] or config["mode"] != "federated"):
        raise ValueError("The protected mode requires federated DP training")
    if not 3 <= config["threshold"] < config["cohort_size"] <= config["num_clients"]:
        raise ValueError("Require 3 <= threshold < cohort_size <= num_clients")
    for key in ["rounds", "local_steps", "batch_size", "physical_batch_size", "patience", "learning_rate", "epsilon_cap", "compute_scale", "bandwidth_scale"]:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ["dropout", "availability", "battery_reserve"]:
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if config.get("freeze_features") and not config.get("public_checkpoint"):
        raise ValueError("Frozen features require a public initialization checkpoint")
    if not config.get("evaluate_test", True) and config["export_after_training"]:
        raise ValueError("Validation-only runs cannot export/evaluate test implicitly")
    return config


def test_partitions(training_rows, assignments, test_rows, seed):
    # Synthetic farm evaluation is benchmark data, outside the protected records.
    counts = np.ones((10, len(assignments))) * 0.1
    for client, indices in enumerate(assignments):
        for index in indices:
            counts[training_rows[index]["label"], client] += 1
    probabilities = counts / counts.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    group_owners, result = {}, [[] for _ in assignments]
    for i, row in enumerate(test_rows):
        group = row["group_id"]
        if group not in group_owners:
            group_owners[group] = int(rng.choice(len(assignments), p=probabilities[row["label"]]))
        result[group_owners[group]].append(i)
    return result


def run(config, resume=False):
    with run_lock(config["output"]):
        retries = config.get("max_retries", 1)
        if not isinstance(retries, int) or not 0 <= retries <= 3:
            raise ValueError("max_retries must be an integer between 0 and 3")
        for attempt in range(retries + 1):
            try:
                return _run(config, resume or attempt > 0)
            except RuntimeError as exc:
                if attempt >= retries or not (Path(config["output"]) / "state.pt").exists():
                    raise
                emit(Path(config["output"]) / "events.jsonl", event="recovery", status="retrying",
                     reason=type(exc).__name__)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def _run(config, resume=False):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = Path(config["output"])
    state_path = output / "state.pt"
    manifest = read_json(config["manifest"])
    validate_manifest(manifest, verify_files=True)
    if not manifest.get("all_physical_leaves_verified", False) and not config["allow_unverified_groups"]:
        raise ValueError("Dataset contains unverified physical leaf groups; explicitly enable audited mode or use strict data")
    rows = manifest["rows"]
    train_rows = [r for r in rows if r["split"] == "train"]
    val = ImageDataset([r for r in rows if r["split"] == "validation"])
    test_rows = [r for r in rows if r["split"] == "test"]
    assignments = partition(train_rows, config["num_clients"], config["alpha"], config["partition_seed"])
    client_data = [ImageDataset([train_rows[i] for i in indices], augment=config.get("augment", False)) for indices in assignments]
    test_assignment = test_partitions(train_rows, assignments, test_rows, config["partition_seed"])
    identity = digest({"config": config, "manifest": manifest["manifest_hash"]})
    public_hash = file_hash(config["public_checkpoint"]) if config.get("public_checkpoint") else None
    if resume:
        if not state_path.exists():
            raise FileNotFoundError("No checkpoint; refusing to initialize a resumed run")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["config_hash"] != identity:
            raise ValueError("Resume configuration or dataset differs from checkpoint")
        if state.get("public_checkpoint_sha256") != public_hash:
            raise ValueError("Public initialization changed during resumed run")
    else:
        if output.exists() and any(p.name != ".run.lock" for p in output.iterdir()):
            raise FileExistsError(f"Use a fresh output directory or --resume: {output}")
        output.mkdir(parents=True, exist_ok=True)
        seed_everything(config["seed"])
        model = CropCNN(normalize=config.get("normalize", False))
        initial_f1 = -1.0
        if config.get("public_checkpoint"):
            from .warmup import load_public_init
            initial = load_public_init(config["public_checkpoint"], manifest, config.get("normalize", False))
            model.load_state_dict(initial["state_dict"])
            initial_f1 = evaluate(model, val, config["device"])["macro_f1"]
        state = {"run_id": uuid.uuid4().hex, "config_hash": identity, "round": 0,
                 "version": 0, "weights": get_weights(model), "best_weights": get_weights(model),
                 "best_f1": initial_f1, "initial_validation_f1": initial_f1, "best_round": 0,
                 "public_checkpoint_sha256": public_hash,
                 "bad_rounds": 0, "aborts": 0, "history": [],
                 "devices": [asdict(d) for d in make_devices(config["num_clients"], config["partition_seed"])],
                 "simulated_seconds": 0.0, "wire_bytes": 0, "eligible_ever": [], "finished": False}
        atomic_json(output / "config.json", config)
        atomic_json(output / "host.json", host_info())
        atomic_json(output / "partitions.json", {"manifest_hash": manifest["manifest_hash"],
                    "train_indices": assignments, "test_indices": test_assignment,
                    "class_counts": [dict(Counter(train_rows[i]["label"] for i in indices)) for indices in assignments]})
    if config["mode"] != "federated":
        if not resume:
            atomic_torch(state_path, state)
        return run_anchor(config, state, train_rows, client_data, val, test_rows, test_assignment, output, resume)
    profile_path = Path(config["profile_path"])
    profile = read_json(profile_path) if profile_path.exists() else profile_host(profile_path)
    if "freeze_features" in profile and profile["freeze_features"] != config.get("freeze_features", False):
        raise ValueError("Resource profile measures a different trainable parameter set")
    if resume:
        if read_json(output / "profile.json") != profile:
            raise ValueError("Host resource profile changed during resumed run")
    else:
        atomic_json(output / "profile.json", profile)
    ledgers, sigmas = {}, {}
    if config["dp"]:
        for client, dataset in enumerate(client_data):
            ledgers[client] = PrivacyLedger(output / "private" / f"client_{client}.json", state["run_id"], client,
                                             config["epsilon_cap"], config["delta"], create=not resume)
            q = sampling_rate(len(dataset), config["batch_size"])
            sigmas[client] = config["noise_multiplier"] or calibrate_noise(config["epsilon_cap"], config["delta"], q,
                                                                           config["rounds"] * config["local_steps"])
    if not resume:
        atomic_torch(state_path, state)
    events = output / "events.jsonl"
    devices = [Device(**d) for d in state["devices"]]
    start = time.perf_counter()
    for round_index in range(state["round"], config["rounds"]):
        if state["finished"]:
            break
        round_start = time.perf_counter()
        # Exogenous traces are generated for ALL clients before scheduling.
        rng = np.random.default_rng(config["seed"] + round_index * 1009)
        online = rng.random(len(devices)) < config["availability"]
        drops = rng.random(len(devices)) < config["dropout"]
        jitter = rng.uniform(0.9, 1.1, len(devices))
        update_size = sum(w.nbytes for w in state["weights"])
        feasible = {}
        for d in devices:
            d.age += 1
            plan = workload(d, profile, config, update_size)
            if not plan or not online[d.client_id]:
                continue
            ledger = ledgers.get(d.client_id)
            if ledger and not ledger.can_step(sigmas[d.client_id], sampling_rate(len(client_data[d.client_id]), plan["batch_size"])):
                continue
            feasible[d.client_id] = plan
        state["eligible_ever"] = sorted(set(state["eligible_ever"]) | set(feasible))
        selected = select_clients(devices, feasible, config["cohort_size"], config["scheduler"], rng)
        local_stats = {}

        def train(client_id, weights):
            plan = feasible[client_id]
            ledger = ledgers.get(client_id)
            spent_steps = ledger.entries if ledger else 0
            result, stats = train_local(weights, client_data[client_id], config,
                                       seed=config["seed"] + round_index * 10000 + client_id * 101 + spent_steps,
                                       steps=plan["steps"], batch_size=plan["batch_size"],
                                       ledger=ledger, sigma=sigmas.get(client_id))
            local_stats[client_id] = stats
            if ledger:
                emit(events, event="privacy", run_id=state["run_id"], config_hash=identity,
                     round=round_index + 1, client_id=client_id, epsilon=ledger.epsilon,
                     epsilon_cap=ledger.cap, steps=ledger.entries,
                     sample_rate=stats["sample_rate"], noise_multiplier=sigmas[client_id])
            return result

        if len(selected) < config["cohort_size"]:
            aggregate, protocol = None, {"survivors": [], "wire_bytes": 0, "bytes_by_node": {}, "bytes_by_stage": {}}
            reason = "insufficient_eligible_clients"
        elif config["secure_aggregation"]:
            aggregate, protocol = secure_round(state["weights"], selected, train, state["version"],
                                                drop_ids=[i for i in selected if drops[i]],
                                                drop_stage=config["dropout_stage"], threshold=config["threshold"])
            reason = "secagg_threshold" if aggregate is None else "ok"
        else:
            aggregate, protocol = plain_round(state["weights"], selected, train,
                                              drop_ids=[i for i in selected if drops[i]], threshold=config["threshold"])
            reason = "survivor_threshold" if aggregate is None else "ok"
        if aggregate is None:
            state["aborts"] += 1
        else:
            state["weights"] = [x.astype(np.float32) for x in aggregate]
            state["version"] += 1
        elapsed = []
        for d in devices:
            cid = d.client_id
            if cid in local_stats:
                compute = feasible[cid]["step_seconds"] * local_stats[cid]["steps"]
                transfer = protocol["bytes_by_node"].get(cid, 0) * 8 / (d.uplink_mbps * config["bandwidth_scale"] * 1e6)
                stages = max(1, len(protocol["bytes_by_stage"]))
                seconds = (compute + transfer + stages * d.rtt_seconds) * jitter[cid]
                elapsed.append(seconds)
                if d.profile != "gateway":
                    d.battery = max(0.0, d.battery - (compute * 0.00002 + transfer * 0.00004))
            if cid in protocol["survivors"] and aggregate is not None:
                d.participations += 1
                d.age = 0
        simulated = max(elapsed, default=config["round_deadline"] if aggregate is None else 0)
        state["simulated_seconds"] += simulated
        state["wire_bytes"] += protocol["wire_bytes"]
        model = CropCNN(normalize=config.get("normalize", False))
        set_weights(model, state["weights"])
        metrics = evaluate(model, val, config["device"])
        if metrics["macro_f1"] > state["best_f1"]:
            state["best_f1"] = metrics["macro_f1"]
            state["best_weights"] = state["weights"]
            state["best_round"] = round_index + 1
            state["bad_rounds"] = 0
        else:
            state["bad_rounds"] += 1
        row = {"round": round_index + 1, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
               "selected": selected, "survivors": protocol["survivors"], "status": "committed" if aggregate is not None else "aborted",
               "reason": reason, "model_version": state["version"], "simulated_seconds": state["simulated_seconds"],
               "wall_seconds": time.perf_counter() - round_start, "wire_bytes": protocol["wire_bytes"],
               "update_bytes": update_size * len(protocol["survivors"]),
               "secagg_overhead_bytes": max(0, protocol["wire_bytes"] - update_size * (len(selected) + len(protocol["survivors"]))) if config["secure_aggregation"] else 0,
               "coverage": sum(d.participations > 0 for d in devices) / len(devices),
               "abort_rate": state["aborts"] / (round_index + 1)}
        state["history"].append(row)
        state["round"] = round_index + 1
        state["devices"] = [asdict(d) for d in devices]
        state["finished"] = (state["round"] >= config.get("min_rounds", 0) and state["bad_rounds"] >= config["patience"]) or state["round"] >= config["rounds"]
        atomic_torch(state_path, state)
        atomic_json(output / "rounds.json", state["history"])
        emit(events, event="round", run_id=state["run_id"], config_hash=identity, **row)
        print(f"Round {state['round']}: {row['status']}, validation F1={metrics['macro_f1']:.4f}, simulated={simulated:.1f}s", flush=True)
        if state["finished"]:
            break
    model = CropCNN(normalize=config.get("normalize", False))
    set_weights(model, state["best_weights"])
    atomic_torch(output / "best.pt", model.state_dict())
    test = evaluate(model, ImageDataset(test_rows), config["device"]) if config.get("evaluate_test", True) else None
    farm = [evaluate(model, ImageDataset([test_rows[i] for i in ids]), config["device"])["macro_f1"] if ids else None
            for ids in test_assignment] if config.get("evaluate_test", True) else []
    valid = sorted(x for x in farm if x is not None)
    reached = [r for r in state["history"] if config["target_f1"] is not None and r["macro_f1"] >= config["target_f1"]]
    summary = {"run_id": state["run_id"], "config_hash": identity, "manifest_hash": manifest["manifest_hash"],
               "grouping_policy": manifest.get("grouping_policy"),
               "all_physical_leaves_verified": manifest.get("all_physical_leaves_verified", False),
               "mode": config["mode"], "rounds": state["round"], "test": test,
               "best_validation_f1": state["best_f1"], "farm_test_f1": farm,
               "initial_validation_f1": state.get("initial_validation_f1", -1), "best_round": state.get("best_round"),
               "public_checkpoint_sha256": state.get("public_checkpoint_sha256"),
               "test_status": "evaluated" if test else "withheld",
               "worst_quartile_f1": float(np.mean(valid[:max(1, int(np.ceil(len(valid) / 4)))])) if valid else None,
               "epsilon_by_client": {str(k): v.epsilon for k, v in ledgers.items()},
               "budget_adherence": all(v.epsilon <= v.cap for v in ledgers.values()),
               "abort_rate": state["aborts"] / max(1, state["round"]),
               "simulated_seconds": state["simulated_seconds"], "wire_bytes": state["wire_bytes"],
               "wall_seconds": sum(r["wall_seconds"] for r in state["history"]),
               "target_f1": config["target_f1"], "time_to_target": reached[0]["simulated_seconds"] if reached else None,
               "bytes_to_target": sum(r["wire_bytes"] for r in state["history"][:reached[0]["round"]]) if reached else None,
               "target_status": "not_configured" if config["target_f1"] is None else "reached" if reached else "unreached",
               "devices": state["devices"], "eligible_ever": state["eligible_ever"],
               "all_eligible_participated": all(devices[i].participations > 0 for i in state["eligible_ever"])}
    atomic_json(output / "summary.json", summary)
    if config["export_after_training"]:
        from .deployment import export_models
        export_models(config, output)
    return summary


def run_anchor(config, state, train_rows, client_data, val, test_rows, test_assignment, output, resume):
    if config["dp"] or config["secure_aggregation"]:
        raise ValueError("Research anchors are nonprivate; set dp and secure_aggregation false")
    datasets = [ImageDataset(train_rows, augment=config.get("augment", False))] if config["mode"] == "centralized" else client_data
    results = []
    for client, dataset in enumerate(datasets):
        target = output / f"anchor_{client}.pt"
        done = output / f"anchor_{client}.json"
        if resume and done.exists():
            results.append(read_json(done))
            continue
        weights, best, best_f1, bad = state["weights"], state["weights"], state["best_f1"], 0
        history = []
        for round_index in range(config["rounds"]):
            weights, timing = train_local(weights, dataset, config, seed=config["seed"] + 1000 * client + round_index,
                                         steps=config["local_steps"], batch_size=config["batch_size"])
            model = CropCNN(normalize=config.get("normalize", False))
            set_weights(model, weights)
            metrics = evaluate(model, val, config["device"])
            history.append({"round": round_index + 1, "macro_f1": metrics["macro_f1"], "wall_seconds": timing["wall_seconds"]})
            if metrics["macro_f1"] > best_f1:
                best, best_f1, bad = weights, metrics["macro_f1"], 0
            else:
                bad += 1
            if round_index + 1 >= config.get("min_rounds", 0) and bad >= config["patience"]:
                break
        set_weights(model, best)
        atomic_torch(target, model.state_dict())
        if config["mode"] == "centralized":
            atomic_torch(output / "best.pt", model.state_dict())
        evaluation_rows = test_rows if config["mode"] == "centralized" else [test_rows[i] for i in test_assignment[client]]
        result = {"client": client, "test": evaluate(model, ImageDataset(evaluation_rows), config["device"]) if config.get("evaluate_test", True) else None,
                  "best_validation_f1": best_f1, "history": history}
        atomic_json(done, result)
        results.append(result)
    summary = {"mode": config["mode"], "config_hash": state["config_hash"], "run_id": state["run_id"],
               "manifest_hash": read_json(config["manifest"])["manifest_hash"],
               "test": metrics_from_confusion(sum(np.array(r["test"]["confusion_matrix"]) for r in results)) if config.get("evaluate_test", True) else None,
               "evaluation_scope": "pooled disjoint farm predictions" if config["mode"] == "local" else "global test set",
               "models": results, "mean_test_f1": float(np.mean([r["test"]["macro_f1"] for r in results])) if config.get("evaluate_test", True) else None}
    atomic_json(output / "summary.json", summary)
    state["finished"] = True
    atomic_torch(output / "state.pt", state)
    if config["mode"] == "centralized" and config["export_after_training"]:
        from .deployment import export_models
        export_models(config, output)
    return summary
