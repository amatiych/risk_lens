from models.portfolio import Portfolio
from models.regime_model import RegimeModel
from pandas import DataFrame

class RegimeAnalysis:
    def __init__(self,portfolio:Portfolio, regime_model:RegimeModel):

        self.regime_model = regime_model
        self.all_ts = portfolio.time_series.pct_change(1).dropna()
        self.proforma = self.all_ts.values  @ portfolio.W
        self.port_ts = DataFrame(self.proforma,index=self.all_ts.index)
        self.reg_data = self.port_ts.merge(self.regime_model.regime_dates, left_index=True, right_on="date")
        self.reg_stats = self.reg_data.groupby("regime").mean()
        self.reg_stats = self.reg_stats.merge(regime_model.regime_info, left_on="regime", right_on="regime")
        self.all_ts  = self.all_ts.merge(regime_model.regime_dates, left_index=True,right_on='date')
