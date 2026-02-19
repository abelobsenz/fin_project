"""Learned underlying hedging policies.

This module adds a lightweight, ML-driven hedge policy that can be trained
*after* an options strategy backtest has been run.

It is intentionally independent from the main IV dynamics model training:

- The IV model learns the state of the implied-vol surface and price/fill heads.
- The hedge policy learns how much underlying delta to neutralize given the
  current market state (latent IV surface + context) and the strategy's
  portfolio delta exposure.

The hedge policy is trained offline on historical backtest episodes by
optimizing a mean-vs-risk objective (maximize mean daily PnL, penalize daily
PnL volatility) with realistic underlying execution costs.
"""

from ivdyn.hedge_policy.policy import HedgePolicyBundle
from ivdyn.hedge_policy.train import HedgePolicyTrainConfig, train_hedge_policy

__all__ = ["HedgePolicyBundle", "HedgePolicyTrainConfig", "train_hedge_policy"]
