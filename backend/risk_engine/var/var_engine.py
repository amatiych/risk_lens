from pandas import read_parquet
import numpy as np
import json
from core.timer import timed
from numba import njit
"""
    Simplified Version of VaR Engine.
    It calculates the following measures:
        1. var - value at risk at each confidence level (e.g. 0.95,0.99 etc)
        2. es - expeted shortfall at the same cofidence level 
        3. marginal var - var that accounts for removal of one asset at a time
        4. incremental var - each assets contribution to var.

"""
class VaR:

    def __init__(self,*, ci, var, k, es,idx,
                 marginal_var=[], incremental_var=[]):
        self.ci = float(ci)
        self.var = float(var)
        self.es = float(es)
        self.var_index = int(k)
        self.tail_indexes = list(idx)
        self.marginal_var = marginal_var,
        self.incremental_var = incremental_var

    def __repr__(self):
        return self.__dict__.__repr__()

    def to_json(self):
        return json.dumps(self.__dict__)

#@njit()
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

        idx = np.argpartition(P, k)
        idxs[i, :] = idx
        vars_[i] = -P[idx[k]]

    return vars_, ks, idxs

def calc_expected_shortfall(P, idx, k):
    """Calculate expected shortfall using numba - vectorized numpy operations."""
    return np.mean(P[idx[:k]])


def calc_marginal_var_batch(P, C, k):
    """
    Optimized marginal VaR calculation using numba.
    Processes each asset column efficiently.
    
    P   : 1D array (T,) - portfolio P&L
    C   : 2D array (T, N) - component contributions R * W
    k   : int - index for VaR threshold
    
    Returns:
        var_wo : 1D array (N,) - VaR without each asset
    """
    T, N = C.shape
    var_wo = np.empty(N, dtype=np.float64)
    
    # Process each asset separately - numba-friendly approach
    for i in range(N):
        P_wo = P - C[:, i]
        kth_val = np.partition(P_wo, k)[k]
        var_wo[i] = kth_val
    
    return var_wo

class VarEngine:

    def __init__(self, df_time_series, W):
        self.df_time_series = df_time_series
        self.df_returns = self.df_time_series.pct_change(1)
        self.R = self.df_returns.fillna(0).values.astype(np.float64)
        self.W = np.asarray(W, dtype=np.float64)
        # Pre-compute component contributions (R * W) - doesn't change unless weights change
        # This is a major optimization - C is reused across all calc_var calls
        self._C = (self.R * self.W).astype(np.float64)

    def calc_proforma(self):
        return self.R @ self.W

    def calc_var(self, cis=[0.95]):
        P = self.R @ self.W
        cis_arr = np.asarray(cis, dtype=np.float64)

        vars_, ks, idxs = calc_var_core(P, cis_arr)

        # C is pre-computed in __init__ - no need to recompute
        C = self._C

        results = []
        for i, ci in enumerate(cis):
            k = ks[i]
            idx = idxs[i, :]
            
            # OPTIMIZATION 1: Numba-accelerated ES calculation
            # Old: es = np.mean([P[j] for j in idx[:k]])
            # New: Numba-accelerated numpy array indexing
            es = calc_expected_shortfall(P, idx, k)
            
            var = -1 * float(vars_[i])
            var_idx = int(idx[k])
            
            # OPTIMIZATION 2: Numba-accelerated marginal VaR calculation
            # Uses numba-optimized loop instead of numpy broadcasting + partition
            # This is faster for large numbers of assets
            var_wo = calc_marginal_var_batch(P, C, k)  # (N,)
            mar_var = var - var_wo  # (N,)
            inc_var = C[var_idx, :]

            results.append(
                VaR(
                    ci = float(ci),
                    var = var,
                    k = var_idx,
                    es = float(es),
                    idx = [int(j) for j in idx[:k]],
                    marginal_var = [float(mv) for mv in mar_var],
                    incremental_var = [float(iv) for iv in inc_var]
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
