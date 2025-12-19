from models.portfolio import Portfolio
from models.enrichment.time_series_enricher import YahooTimeSeriesEnricher
from models.enrichment.price_enricher import YahooFinancePriceEnricher
from backend.llm.var_analyzer import VaRAnalyzer
if __name__ == "__main__":



    print("-----------------------------------------")
    portfolio = Portfolio.load(104)

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


    from backend.reporting.portfolio_report import PortfolioReport
    pr = PortfolioReport(portfolio,var,var_engine.CR)
    claude = VaRAnalyzer(pr)
    analysis = claude.analyze()
    print(analysis)