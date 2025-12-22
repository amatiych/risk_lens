from dataclasses import dataclass
from typing import List
from numpy import zeros
from pandas import DataFrame, read_csv
from datetime import datetime

class TransitionMatrix:

    def __init__(self,regime_ts:DataFrame):
        regimes = regime_ts['regime'].values
        regime_list = list(set(regimes))

        n = len(regime_list)
        txn_counts = zeros((n,n),dtype=int)
        for i in range(len(regimes)-1):
            from_regime_id = regimes[i]
            to_regime_id = regimes[i+1]
            txn_counts[from_regime_id,to_regime_id] += 1
        row_sums = txn_counts.sum(axis=1,keepdims=True)
        self.txn_probs = txn_counts/row_sums / row_sums



@dataclass
class RegimeModel:

    factors: List[str]
    mean_returns : DataFrame
    covariances: DataFrame
    regime_dates: DataFrame
    regime_info: DataFrame
    @classmethod
    def load(cls) -> "RegimeModel":
        means = read_csv("data/regime/means.csv")
        covs = read_csv("data/regime/covs.csv")
        regime_info = read_csv("data/regime/regime_desc.csv")
        regime_dates = read_csv("data/regime/regimes.csv")
        regime_dates['date'] = regime_dates['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
        regime_dates.set_index('date', inplace=True)
        factors = means.columns.values[1:]
        return RegimeModel(factors,means, covs, regime_dates,regime_info)



if __name__ == "__main__":
    regime = RegimeModel.load()
    print(regime.mean_returns)