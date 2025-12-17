from numpy import argpartition
from pandas import read_parquet
from numba import njit
import numpy as np
from purple_swan.core.timer import timed


class VaR:
    def __init__(self, ci, var, k, idx):
        self.CI = ci
        self.VaR = var
        self.Idx = k
        self.Index = idx


# ---------- NUMERIC CORE: nopython ----------

@njit(cache=True, fastmath=True)
def calc_var_core(P, cis):
    """
    P   : 1D array (T,)
    cis : 1D array (n_cis,)

    Returns:
        vars_ : (n_cis,)
        ks    : (n_cis,)
        idxs  : (n_cis, T)
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

        idx = argpartition(P, k)
        idxs[i, :] = idx
        vars_[i] = -P[idx[k]]

    return vars_, ks, idxs


class VarEngine:

    def __init__(self, df_time_series, W):
        self.df_time_series = df_time_series
        self.df_returns = self.df_time_series.pct_change(1)
        self.R = self.df_returns.values.astype(np.float64)
        self.W = np.asarray(W, dtype=np.float64)

    def calc_proforma(self):
        return self.R @ self.W

    def calc_var(self, cis):
        P = self.R @ self.W
        cis_arr = np.asarray(cis, dtype=np.float64)

        vars_, ks, idxs = calc_var_core(P, cis_arr)

        results = []
        for i, ci in enumerate(cis):
            k = ks[i]
            idx = idxs[i, :]
            results.append(
                VaR(
                    ci = float(ci),
                    var = float(vars_[i]),
                    k = int(idx[k]),
                    idx = idx[:k].copy()
                )
            )
        return results



    import numpy as np

    def hist_var_from_pnl(pnl: np.ndarray, alpha: float = 0.99) -> float:
        """
        Historical VaR (>0) at confidence alpha from scenario P&L vector.
        """
        T = pnl.shape[0]
        k = int((1 - alpha) * T)
        kth = np.partition(pnl, k)[k]  # k-th smallest (loss quantile)
        return -kth

    def marginal_var(self,
            alpha: float = 0.99,
    ):

        # Base portfolio P&L and VaR
        pnl = self.R @ self.W  # (T,)
        base_var = hist_var_from_pnl(pnl, alpha)

        T = returns.shape[0]
        k = int((1 - alpha) * T)

        # pnl_wo[:, i] = pnl - w_i * r_i, all i at once:
        # returns * weights -> (T, N) by broadcasting
        pnl_wo = pnl[:, None] - returns * weights  # (T, N)

        # For each column i, we want the k-th smallest element across T
        kth_vals = np.partition(pnl_wo, k, axis=0)[k, :]  # (N,)
        var_wo = -kth_vals  # VaR without each name

        # Incremental / marginal-by-removal VaR
        inc_var = base_var - var_wo  # (N,)

        return base_var, inc_var

    def warmup(self):
        # one dummy call to trigger compilation
        P = self.R @ self.W
        cis_arr = np.array([0.99, 0.97, 0.95], dtype=np.float64)
        _ = calc_var_core(P, cis_arr)


if __name__ == "__main__":
    df_ts = read_parquet("s3://pswn-test/all_time_series.parquet")
    print("have ts")
    N = len(df_ts.columns)
    weights = [1.0 / N] * N
    var_e = VarEngine(df_ts, weights)

    # 1) Warm up Numba (compilation happens here)
    var_e.warmup()

    # 2) Time the Python wrapper (still builds VaR objects)
    @timed
    def test_var_python_wrapper():
        for _ in range(100000):
            _ = var_e.calc_var([0.99, 0.97, 0.95])

    test_var_python_wrapper()

    # 3) (Optional) Time just the Numba core to see real gain
    P = var_e.R @ var_e.W
    cis_arr = np.array([0.99, 0.97, 0.95], dtype=np.float64)

    @timed
    def test_numba_core():
        for _ in range(100000):
            _ = calc_var_core(P, cis_arr)

    test_numba_core()

    print("done")
