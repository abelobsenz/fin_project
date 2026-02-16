from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class MassiveFlatFilesConfig:
    endpoint_url: str = os.getenv("MASSIVE_FILES_ENDPOINT_URL", "https://files.massive.com")
    bucket: str = os.getenv("MASSIVE_FILES_BUCKET", "flatfiles")
    prefix: str = os.getenv("MASSIVE_FILES_PREFIX", "us_options_opra/day_aggs_v1")
    aws_access_key_id_env_var: str = "MASSIVE_FILES_ACCESS_KEY_ID"
    aws_secret_access_key_env_var: str = "MASSIVE_FILES_SECRET_ACCESS_KEY"
    cache_enabled: bool = True
    cache_dir: Path = Path("data/massive_cache/flatfiles")
    read_chunksize: int = 250_000


_OSI_PATTERN = re.compile(
    r"^(?:O:)?(?P<underlying>[A-Z]{1,6})(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$"
)


def parse_option_symbol_osi(ticker: str) -> dict[str, Any] | None:
    m = _OSI_PATTERN.match(ticker.strip().upper())
    if m is None:
        return None
    yy = int(m.group("yy"))
    expiry = date(2000 + yy, int(m.group("mm")), int(m.group("dd")))
    strike = int(m.group("strike")) / 1000.0
    return {
        "underlying": m.group("underlying"),
        "expiry": expiry,
        "call_put": m.group("cp"),
        "strike": float(strike),
    }


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for c in candidates:
        got = lowered.get(c.lower())
        if got is not None:
            return got
    return None


class MassiveFlatFileClient:
    def __init__(self, config: MassiveFlatFilesConfig | None = None) -> None:
        self.config = config or MassiveFlatFilesConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _credentials(self) -> tuple[str, str]:
        access = os.getenv(self.config.aws_access_key_id_env_var)
        secret = os.getenv(self.config.aws_secret_access_key_env_var)
        if not access or not secret:
            raise RuntimeError(
                "Missing Massive flat-files credentials. "
                f"Set {self.config.aws_access_key_id_env_var} and "
                f"{self.config.aws_secret_access_key_env_var}."
            )
        return access, secret

    def _s3_client(self):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for Massive flat-files. Install with `pip install boto3`."
            ) from exc

        access, secret = self._credentials()
        session = boto3.Session(
            aws_access_key_id=access,
            aws_secret_access_key=secret,
        )
        return session.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            config=Config(signature_version="s3v4"),
        )

    def object_key_for_date(self, asof: date) -> str:
        return f"{self.config.prefix}/{asof:%Y/%m/%Y-%m-%d}.csv.gz"

    def download_day_file(self, asof: date) -> Path:
        key = self.object_key_for_date(asof)
        local = self.config.cache_dir / key
        if self.config.cache_enabled and local.exists():
            return local

        local.parent.mkdir(parents=True, exist_ok=True)
        s3 = self._s3_client()
        tmp = local.with_suffix(local.suffix + ".part")
        try:
            s3.download_file(self.config.bucket, key, str(tmp))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Massive flat file for {asof.isoformat()} at key={key}: {exc}"
            ) from exc
        tmp.replace(local)
        return local

    def read_day_aggs_for_underlying(self, asof: date, underlying: str) -> pd.DataFrame:
        path = self.download_day_file(asof)
        target_prefix = f"O:{underlying.upper()}"
        chunks = pd.read_csv(path, compression="gzip", chunksize=self.config.read_chunksize)
        frames: list[pd.DataFrame] = []

        for chunk in chunks:
            ticker_col = _find_col(
                list(chunk.columns),
                ["ticker", "symbol", "sym", "options_ticker"],
            )
            if ticker_col is None:
                raise ValueError("Could not find options ticker column in Massive flat file")

            tickers = chunk[ticker_col].astype(str)
            mask = tickers.str.startswith(target_prefix)
            if not mask.any():
                continue
            slim = chunk.loc[mask].copy()
            slim["ticker"] = tickers.loc[mask].values
            frames.append(slim)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
