"""Command-line interface for ivdyn."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Sequence

from ivdyn.utils.paths import resolve_latest


def _to_path(v: str) -> Path:
    return Path(v).expanduser()


def _resolve_run_dir(raw: str | None) -> Path:
    if raw:
        return _to_path(raw).resolve()
    latest = resolve_latest(Path("outputs/runs"))
    if latest is None:
        raise RuntimeError("No run directory found. Provide --run-dir explicitly.")
    return latest


def _resolve_dataset(dataset_arg: str | None, run_dir: Path) -> Path:
    if dataset_arg:
        return _to_path(dataset_arg).resolve()
    train_summary = run_dir / "train_summary.json"
    if not train_summary.exists():
        raise RuntimeError("No dataset argument and no train_summary.json found in run dir.")
    import json

    payload = json.loads(train_summary.read_text(encoding="utf-8"))
    if not (dataset := payload.get("dataset_path")):
        raise RuntimeError("train_summary.json exists but missing dataset_path.")
    return _to_path(str(dataset)).resolve()


def _train_command(ns: Any) -> None:
    from ivdyn.training import TrainingConfig, train

    run_dir = train(
        _to_path(ns.dataset).resolve(),
        TrainingConfig(
            out_dir=_to_path(ns.out_dir),
            seed=ns.seed,
            train_frac=ns.train_frac,
            val_frac=ns.val_frac,
            latent_dim=ns.latent_dim,
            vae_epochs=ns.vae_epochs,
            vae_batch_size=ns.vae_batch_size,
            vae_lr=ns.vae_lr,
            vae_kl_beta=ns.vae_kl_beta,
            noarb_lambda=ns.noarb_lambda,
            head_epochs=ns.head_epochs,
            dyn_batch_size=ns.dyn_batch_size,
            contract_batch_size=ns.contract_batch_size,
            head_lr=ns.head_lr,
            joint_epochs=ns.joint_epochs,
            joint_lr=ns.joint_lr,
            joint_contract_batch_size=ns.joint_contract_batch_size,
            joint_dyn_lambda=ns.joint_dyn_lambda,
            joint_price_lambda=ns.joint_price_lambda,
            joint_exec_lambda=ns.joint_exec_lambda,
            weight_decay=ns.weight_decay,
            price_risk_weight=ns.price_risk_weight,
            exec_risk_weight=ns.exec_risk_weight,
            risk_focus_abs_x=ns.risk_focus_abs_x,
            risk_focus_tau_days=ns.risk_focus_tau_days,
            exec_label_smoothing=ns.exec_label_smoothing,
            exec_logit_l2=ns.exec_logit_l2,
        ),
    )
    print(run_dir)


def _build_dataset_command(ns: Any) -> None:
    from ivdyn.data import DatasetBuildConfig, build_dataset

    out = build_dataset(
        DatasetBuildConfig(
            data_root=_to_path(ns.data_root),
            out_dir=_to_path(ns.out_dir),
            symbol=ns.symbol,
            plugin=ns.plugin,
            api_key=ns.api_key,
            start_date=ns.start_date,
            end_date=ns.end_date,
            x_grid=tuple(ns.x_grid),
            tenor_days=tuple(ns.tenor_days),
            max_contracts_per_day=ns.max_contracts_per_day,
            random_seed=ns.random_seed,
            num_workers=ns.num_workers,
        )
    )
    print(out["dataset"])


def _evaluate_command(ns: Any) -> None:
    from ivdyn.eval import evaluate

    run_dir = _resolve_run_dir(ns.run_dir)
    dataset = _resolve_dataset(ns.dataset, run_dir)
    out_dir = evaluate(
        run_dir=run_dir,
        dataset_path=dataset,
        device=ns.device,
        num_workers=ns.num_workers,
    )
    print(out_dir)


def _backtest_command(ns: Any) -> None:
    try:
        from ivdyn.backtest import BacktestConfig, run_backtest
    except Exception as exc:
        raise RuntimeError(
            "Backtest support is not available in this checkout. Reinstall from a build "
            "that includes ivdyn.backtest."
        ) from exc

    run_dir = _resolve_run_dir(ns.run_dir)
    dataset = _resolve_dataset(ns.dataset, run_dir)
    out_dir = run_backtest(
        BacktestConfig(
            run_dir=run_dir,
            dataset_path=dataset,
            device=ns.device,
            num_workers=ns.num_workers,
            inference_batch_size=ns.inference_batch_size,
            fill_gate=ns.fill_gate,
            slippage_bps=ns.slippage_bps,
            max_trades_per_day=ns.max_trades_per_day,
            signal_abs_gate=ns.signal_abs_gate,
            min_dte=ns.min_dte,
            max_dte=ns.max_dte,
            min_moneyness=ns.min_moneyness,
            max_moneyness=ns.max_moneyness,
            max_rel_spread=ns.max_rel_spread,
        )
    )
    print(out_dir)


def _ui_command(ns: Any) -> None:
    env = os.environ.copy()
    if ns.run_dir:
        env["IVDYN_DEFAULT_RUN_DIR"] = _to_path(ns.run_dir).resolve().as_posix()
    app_path = Path(__file__).resolve().parent.parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    rc = subprocess.call(cmd, env=env)
    raise SystemExit(rc)


def _not_implemented(_: Any) -> None:
    raise RuntimeError(
        "This command is not implemented in the local checkout. "
        "Use external data-prep tooling and keep files in the expected locations."
    )


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", default="outputs/runs")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--vae-epochs", type=int, default=120)
    p.add_argument("--vae-batch-size", type=int, default=32)
    p.add_argument("--vae-lr", type=float, default=2e-3)
    p.add_argument("--vae-kl-beta", type=float, default=0.02)
    p.add_argument("--noarb-lambda", type=float, default=0.2)
    p.add_argument("--head-epochs", type=int, default=130)
    p.add_argument("--dyn-batch-size", type=int, default=64)
    p.add_argument("--contract-batch-size", type=int, default=2048)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--joint-epochs", type=int, default=30)
    p.add_argument("--joint-lr", type=float, default=5e-4)
    p.add_argument("--joint-contract-batch-size", type=int, default=4096)
    p.add_argument("--joint-dyn-lambda", type=float, default=1.0)
    p.add_argument("--joint-price-lambda", type=float, default=1.0)
    p.add_argument("--joint-exec-lambda", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--price-risk-weight", type=float, default=1.0)
    p.add_argument("--exec-risk-weight", type=float, default=0.5)
    p.add_argument("--risk-focus-abs-x", type=float, default=0.06)
    p.add_argument("--risk-focus-tau-days", type=float, default=20.0)
    p.add_argument("--exec-label-smoothing", type=float, default=0.03)
    p.add_argument("--exec-logit-l2", type=float, default=2e-4)
    p.set_defaults(func=_train_command)

    p = sub.add_parser("build-dataset")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--plugin", default="massive_raw_parquet")
    p.add_argument("--api-key")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument(
        "--x-grid",
        nargs="+",
        type=float,
        default=[-0.35, -0.30, -0.25, -0.20, -0.16, -0.12, -0.09, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35],
    )
    p.add_argument("--tenor-days", nargs="+", type=int, default=[7, 14, 30, 60, 90, 180])
    p.add_argument("--max-contracts-per-day", type=int, default=900)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)
    p.set_defaults(func=_build_dataset_command)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--device")
    p.add_argument("--num-workers", type=int, default=0)
    p.set_defaults(func=_evaluate_command)

    p = sub.add_parser("backtest")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--inference-batch-size", type=int, default=65536)
    p.add_argument("--fill-gate", type=float, default=0.65)
    p.add_argument("--slippage-bps", type=float, default=7.5)
    p.add_argument("--max-trades-per-day", type=int, default=5)
    p.add_argument(
        "--signal-abs-gate",
        type=float,
        default=0.04,
        help="Require |signal| >= threshold (signal is relative edge: (mid_now - pred_next) / mid_now) before a contract is eligible for trading.",
    )
    p.add_argument("--min-dte", type=int, default=7)
    p.add_argument("--max-dte", type=int, default=75)
    p.add_argument("--min-moneyness", type=float, default=0.88)
    p.add_argument("--max-moneyness", type=float, default=1.12)
    p.add_argument("--max-rel-spread", type=float, default=0.10)
    p.set_defaults(func=_backtest_command)

    p = sub.add_parser("ui")
    p.add_argument("--run-dir", default=None)
    p.set_defaults(func=_ui_command)

    p = sub.add_parser("pull-underlying-massive")
    p.set_defaults(func=_not_implemented)

    p = sub.add_parser("pull-massive")
    p.set_defaults(func=_not_implemented)

    p = sub.add_parser("pull-flatfiles")
    p.set_defaults(func=_not_implemented)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
