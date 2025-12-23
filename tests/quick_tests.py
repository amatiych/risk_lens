from backend.risk_engine.factor_analysis import FactorAnalysis
from backend.risk_engine.regime_analysis import RegimeAnalysis
from models.factor_model import FactorModel
from models.portfolio import Portfolio
from models.enrichment.time_series_enricher import YahooTimeSeriesEnricher
from models.enrichment.price_enricher import YahooFinancePriceEnricher
from backend.llm.var_analyzer import VaRAnalyzer
from models.regime_model import RegimeModel

if __name__ == "__main__":



    print("-----------------------------------------")
    portfolio = Portfolio.load(101)

    enrichers = [YahooFinancePriceEnricher(),YahooTimeSeriesEnricher()]
    for enricher in enrichers:
        enricher.enrich_portfolio(portfolio)
    print("-----------------------------------------")
    print(portfolio.holdings)

    print("-----------------------------------------")
    print(portfolio.time_series)
    portfolio.time_series.to_csv("data/time_series.csv")


    tickers = list(portfolio.time_series.columns)


    weights = list(portfolio.holdings.loc[tickers,'weight'].values)

    from backend.risk_engine.var.var_engine import VarEngine
    var_engine = VarEngine(portfolio.time_series,weights)
    var = var_engine.calc_var(cis=[0.95])

    model = FactorModel.load('fama_french')
    fa = FactorAnalysis(model)
    res = fa.analyze(portfolio)

    regime_model = RegimeModel.load()
    regime_analysis = RegimeAnalysis(portfolio, regime_model)


    from backend.reporting.portfolio_report import PortfolioReport
    pr = PortfolioReport(portfolio,var,var_engine.CR,res,regime_analysis)
    open("/tmp/portfolio_report.txt","w").write(pr.report)
    print(pr.report)
    claude = VaRAnalyzer(pr)

    analysis = claude.analyze()

    print(analysis)