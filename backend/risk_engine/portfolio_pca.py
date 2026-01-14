import numpy as np

from models.portfolio import Portfolio
from sklearn.decomposition import PCA
from pandas import DataFrame
class PortfolioPCA:
    def __init__(self, portfolio:Portfolio):
        self.portfolio = portfolio
        self.ts = portfolio.time_series
        self.ts = self.ts.pct_change(1).dropna()
        self.pca = PCA()
        self.comp = self.pca.fit_transform(self.ts)
        self.var_pct = self.pca.explained_variance_ratio_
        self.cum_var_pct = np.cumsum(self.var_pct)
