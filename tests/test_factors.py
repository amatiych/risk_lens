
from models.portfolio import Portfolio
from models.factor_model import FactorModel
from models.enrichment.time_series_enricher import YahooTimeSeriesEnricher
from models.enrichment.price_enricher import YahooFinancePriceEnricher

from backend.llm.var_analyzer import VaRAnalyzer
from backend.risk_engine.factor_analysis import FactorAnalysis

if __name__ == "__main__":
    portfolio = Portfolio.load(104)
    print("-----------------------------------------")
    from models.enrichment.price_enricher import YahooFinancePriceEnricher
    from models.enrichment.time_series_enricher import YahooTimeSeriesEnricher

    enrichers = [YahooFinancePriceEnricher(),YahooTimeSeriesEnricher()]
    for enricher in enrichers:
        enricher.enrich_portfolio(portfolio)

    model = FactorModel.load('fama_french')
    fa = FactorAnalysis(model)
    res = fa.analyze(portfolio)
    print("=====================================")
    print(res)


