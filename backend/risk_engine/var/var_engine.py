"""Value at Risk (VaR) calculation engine.

This module provides a high-performance VaR engine that calculates:
    - VaR: Value at Risk at specified confidence levels (e.g., 95%, 99%)
    - ES: Expected Shortfall (Conditional VaR) - average loss beyond VaR
    - Marginal VaR: Risk contribution from removing each asset
    - Incremental VaR: Each asset's contribution to portfolio VaR

The engine uses vectorized NumPy operations for efficient computation
on large portfolios and long time series.
"""

from pandas import read_parquet
import numpy as np
import json
from core.timer import timed


class VaR:
    """Container for Value at Risk calculation results.

    Stores VaR metrics at a specific confidence level including the VaR value,
    expected shortfall, and per-asset risk contributions.

    Attributes:
        ci: Confidence interval (e.g., 0.95 for 95% VaR).
        var: Value at Risk as a percentage of portfolio value.
        es: Expected Shortfall (average loss in tail).
        var_index: Index of the observation at the VaR threshold.
        var_date: Date corresponding to the VaR observation.
        tail_indexes: Indices of observations in the tail.
        marginal_var: List of marginal VaR for each asset.
        incremental_var: List of incremental VaR for each asset.
    """

    def __init__(self, *, ci, var, k, var_date, es, idx,
                 marginal_var=[], incremental_var=[]):
        """Initialize VaR results container.

        Args:
            ci: Confidence interval level.
            var: Calculated VaR value.
            k: Index of VaR observation.
            var_date: Date of VaR observation.
            es: Expected shortfall value.
            idx: Tail observation indices.
            marginal_var: Per-asset marginal VaR values.
            incremental_var: Per-asset incremental VaR values.
        """
        self.ci = float(ci)
        self.var = float(var)
        self.es = float(es)
        self.var_index = int(k)
        self.var_date = var_date
        self.tail_indexes = list(idx)
        self.marginal_var = marginal_var
        self.incremental_var = incremental_var

    def __repr__(self):
        """Return string representation of VaR results."""
        return self.__dict__.__repr__()

    def to_json(self):
        """Serialize VaR results to JSON string."""
        return json.dumps(self.__dict__)

def calc_var_core(P, cis):
    """Calculate VaR values for multiple confidence intervals.

    Uses partial sorting for efficient VaR threshold identification.

    Args:
        P: 1D array of portfolio returns (T,).
        cis: 1D array of confidence intervals (n_cis,).

    Returns:
        Tuple of (vars_, ks, idxs):
            vars_: VaR values for each CI (n_cis,).
            ks: Threshold indices for each CI (n_cis,).
            idxs: Sorted indices for each CI (n_cis, T).
    """
    T = P.shape[0]
    n_cis = cis.shape[0]

    vars_ = np.empty(n_cis, dtype=np.float64)
    ks = np.empty(n_cis, dtype=np.int64)
    idxs = np.empty((n_cis, T), dtype=np.int64)

    for i in range(n_cis):
        ci = cis[i]
        k = int((1.0 - ci) * T)
        ks[i] = k

        idx = np.argpartition(P, k)
        idxs[i, :] = idx
        vars_[i] = -P[idx[k]]

    return vars_, ks, idxs


def calc_expected_shortfall(P, idx, k):
    """Calculate Expected Shortfall (average loss in tail).

    Args:
        P: 1D array of portfolio returns.
        idx: Sorted indices from VaR calculation.
        k: Number of observations in the tail.

    Returns:
        Expected shortfall value (mean of tail losses).
    """
    return np.mean(P[idx[:k]])


def calc_marginal_var_batch(P, C, k):
    """Calculate marginal VaR for all assets in batch.

    Computes VaR without each asset to determine marginal contribution.

    Args:
        P: 1D array (T,) of portfolio P&L.
        C: 2D array (T, N) of component contributions (R * W).
        k: Index for VaR threshold.

    Returns:
        1D array (N,) of VaR values without each asset.
    """
    T, N = C.shape
    var_wo = np.empty(N, dtype=np.float64)
    return var_wo


class VarEngine:
    """High-performance Value at Risk calculation engine.

    Computes VaR, Expected Shortfall, and risk attribution metrics
    using historical simulation methodology.

    Attributes:
        df_time_series: Original price time series DataFrame.
        df_returns: Calculated returns DataFrame.
        R: NumPy array of returns (T x N).
        W: NumPy array of portfolio weights (N,).
        CR: Correlation matrix of returns.

    Example:
        engine = VarEngine(price_df, weights)
        var_results = engine.calc_var(cis=[0.95, 0.99])
        for var in var_results:
            print(f"{var.ci:.0%} VaR: {var.var:.2%}")
    """

    def __init__(self, df_time_series, W):
        """Initialize VaR engine with time series and weights.

        Args:
            df_time_series: DataFrame of historical prices with dates as index.
            W: List or array of portfolio weights for each asset.
        """
        self.df_time_series = df_time_series
        self.df_returns = self.df_time_series.pct_change(1)
        self.R = self.df_returns.fillna(0).values.astype(np.float64)
        self.W = np.asarray(W, dtype=np.float64)
        self._C = (self.R * self.W).astype(np.float64)
        self.CR = self.df_returns.corr()

    def calc_proforma(self):
        """Calculate portfolio P&L time series.

        Returns:
            1D array of portfolio returns (R @ W).
        """
        return self.R @ self.W

    def calc_var(self, cis=[0.95]):
        """Calculate VaR at specified confidence intervals.

        Computes Value at Risk, Expected Shortfall, and per-asset
        risk contributions for each confidence level.

        Args:
            cis: List of confidence intervals (default [0.95]).

        Returns:
            List of VaR objects, one per confidence interval.
        """
        P = self.R @ self.W
        cis_arr = np.asarray(cis, dtype=np.float64)

        vars_, ks, idxs = calc_var_core(P, cis_arr)

        C = self._C

        results = []
        for i, ci in enumerate(cis):
            k = ks[i]
            idx = idxs[i, :]

            es = calc_expected_shortfall(P, idx, k)
            var_idx = int(idx[k])
            var = -1 * float(vars_[i])
            P_wo = P[:, None] - self.R * self.W
            var_wo = np.partition(P_wo, k, axis=0)[k, :]
            mar_var = var - var_wo
            inc_var = C[var_idx, :]

            results.append(
                VaR(
                    ci=float(ci),
                    var=var,
                    k=var_idx,
                    var_date=self.df_time_series.index[k],
                    es=float(es),
                    idx=[int(j) for j in idx[:k]],
                    marginal_var=[float(mv) for mv in mar_var],
                    incremental_var=[float(iv) for iv in inc_var]
                )
            )
        return results


if __name__ == "__main__":
    df_ts = read_parquet("s3://pswn-test/all_time_series.parquet")
    print("have ts")
    N = len(df_ts.columns)
    weights = [1.0 / N] * N
    var_e = VarEngine(df_ts, weights)


    @timed
    def run_var():
        for _ in range (1000):
            res = var_e.calc_var([0.95])
        return res[0]

    res = run_var()
    var = res.var
    ivars = res.incremental_var
    tot_ivar = sum(ivars)
    print (f"VaR: {var} Tot IvaR: {tot_ivar}")
    print(res)
