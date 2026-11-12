"""CLI entry point. Commands have no implicit external publication."""
import argparse
import json
import logging
from pathlib import Path


def main():
    logging.getLogger("flwr").setLevel(logging.WARNING)
    logging.getLogger("flwr").propagate = False
    parser = argparse.ArgumentParser(prog="fedcrop")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Download pinned PlantVillage tomato data")
    prepare.add_argument("--root", default="data")
    prepare.add_argument("--max-groups-per-class", type=int, default=0)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--policy", choices=["strict", "audited"], default="strict")
    prepare.add_argument("--train-images-per-class", type=int, default=0)
    prepare.add_argument("--test-images-per-class", type=int, default=0)
    prepare.add_argument("--preserve-test", action="store_true")
    prepare.add_argument("--manifest-name", default="manifest.json")
    profile = sub.add_parser("profile", help="Measure CPU/GPU DP training cost")
    profile.add_argument("--output", default="artifacts/profile.json")
    train = sub.add_parser("train", help="Run or resume a configured experiment")
    train.add_argument("--config", default="configs/main.json")
    train.add_argument("--output")
    train.add_argument("--resume", action="store_true")
    warmup = sub.add_parser("warmup", help="Pretrain on the explicit public pool only")
    warmup.add_argument("--config", required=True)
    warmup.add_argument("--output", required=True)
    warmup.add_argument("--epochs", type=int, default=100)
    final_test = sub.add_parser("evaluate-run", help="Final test evaluation after configuration selection")
    final_test.add_argument("run_dir")
    export = sub.add_parser("export", help="Export and benchmark a completed teacher")
    export.add_argument("run_dir")
    export.add_argument("--distill-epochs", type=int, help="Public-pool student training budget")
    export.add_argument("--distill-augment", action="store_true", help="Augment public student training images")
    assign = sub.add_parser("assign", help="Reassign inference models after resource changes")
    assign.add_argument("run_dir")
    assign.add_argument("--devices", help="JSON array of device profiles")
    assign.add_argument("--latency-budget-ms", type=float, default=100)
    predict = sub.add_parser("predict", help="Classify one local leaf image using an assigned ONNX model")
    predict.add_argument("run_dir")
    predict.add_argument("image")
    predict.add_argument("--client", type=int, default=0)
    suite = sub.add_parser("suite", help="Generate or run controlled experiment configurations")
    suite.add_argument("--config", default="configs/main.json")
    suite.add_argument("--output", default="runs/suite")
    suite.add_argument("--execute", action="store_true")
    suite.add_argument("--stress", action="store_true")
    report = sub.add_parser("report", help="Collect available results and generate charts")
    report.add_argument("root", nargs="?", default="runs")
    report.add_argument("--output", default="artifacts/analysis")
    sub.add_parser("doctor", help="Inspect installed dependencies and host")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare(args.root, args.max_groups_per_class, workers=args.workers, policy=args.policy,
                train_images_per_class=args.train_images_per_class,
                test_images_per_class=args.test_images_per_class, preserve_test=args.preserve_test,
                manifest_name=args.manifest_name)
    elif args.command == "profile":
        from .resources import profile_host
        print(json.dumps(profile_host(args.output), indent=2))
    elif args.command == "train":
        from .experiment import load_config, run
        config = load_config(args.config, {"output": args.output} if args.output else None)
        print(json.dumps(run(config, args.resume), indent=2))
    elif args.command == "warmup":
        from .experiment import load_config
        from .warmup import public_warmup
        public_warmup(load_config(args.config), args.output, args.epochs)
    elif args.command == "evaluate-run":
        from .warmup import evaluate_run
        print(json.dumps(evaluate_run(args.run_dir), indent=2))
    elif args.command == "export":
        from .deployment import export_models
        from .io import read_json
        config = read_json(Path(args.run_dir) / "config.json")
        if args.distill_epochs is not None:
            config["distill_epochs"] = args.distill_epochs
        if args.distill_augment:
            config["distill_augment"] = True
        export_models(config, args.run_dir)
    elif args.command == "assign":
        from .deployment import assign_models
        print(json.dumps(assign_models(args.run_dir, args.devices, args.latency_budget_ms), indent=2))
    elif args.command == "predict":
        from .deployment import predict
        print(json.dumps(predict(args.run_dir, args.image, args.client), indent=2))
    elif args.command == "suite":
        from .analysis import experiment_suite
        experiment_suite(args.config, args.output, args.execute, args.stress)
    elif args.command == "report":
        from .analysis import report
        report(args.root, args.output)
    elif args.command == "doctor":
        from .io import host_info
        from importlib.metadata import version
        packages = ["torch", "flwr", "opacus", "onnxruntime", "onnx", "streamlit", "numpy"]
        print(json.dumps({"host": host_info(), "packages": {p: version(p) for p in packages}}, indent=2))
