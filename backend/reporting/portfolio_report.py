from dataclasses import dataclass

from backend.risk_engine.factor_analysis import FactorResult
from backend.risk_engine.regime_analysis import RegimeAnalysis
from backend.risk_engine.var.var_engine import VaR
from models.portfolio import Portfolio
import json

class PortfolioReport(object):

     def __init__(self,portfolio:Portfolio, var:VaR, cr=None,factor_res:FactorResult=None,regime_anlaysis: RegimeAnalysis=None):
        self.portfolio = portfolio
        self.var = var
        self.cr = cr
        self.factor_res = factor_res
        self.holdings_json = self.portfolio.holdings.reset_index().to_json(orient='records')

        mvs = self.portfolio.holdings.market_value.values
        net_value = sum(mvs)
        gross_value = sum([abs(w) for w in mvs])
        var_results = []
        marginal_var = []

        tickers = self.portfolio.holdings.index.tolist()
        for var in self.var:
            var_results.append({'CI':var.ci,'Date': var.var_date.strftime("%Y-%m-%d"),'VaR':var.var,'ES':var.es })
            for ticker,mv,iv in zip(tickers,var.marginal_var,var.incremental_var):
                marginal_var.append({'CI': var.ci,'Ticker':ticker,'Marginal VaR':mv,'Incremental VaR':iv})
        self.var_json = json.dumps(var_results)
        self.mvar_json = json.dumps(marginal_var)

        self.factor_data =  {factor:{'beta' : float(beta),'% risk':float(mr)} for factor,beta,mr in
                             zip(self.factor_res.factors, self.factor_res.betas,self.factor_res.marginal_risk)}



        self.report = f"""
        Holdings:
        {self.holdings_json}
        
        NAV: {portfolio.nav}  Net Market Value: {net_value} Gross Market Value: {gross_value}
        NAV:  Net Exposure : {net_value/portfolio.nav} Gross Exposure: {gross_value/portfolio.nav}
        
        Value at Risk Report 
        {self.var_json}
    
        Marginal and Incremental VaR Report:
        {self.mvar_json}    
        
        Correlation Matrix
        {self.cr.reset_index().to_json(orient='records',double_precision=2)}
        
        Factor Analysis
        Factor Model: {self.factor_res.factor_model.model_name}
        Factors Betas and Risk Contribution:     {self.factor_data }
        
        Regime Analysis: 
        Portfolio Regime Stats: {regime_anlaysis.reg_stats.to_json(orient='records',double_precision=2)}
                  
        Holdings Time Series with Regime Data: 
        {regime_anlaysis.all_ts.to_json(orient='records',double_precision=2)}  
        """



