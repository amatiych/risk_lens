from backend.risk_engine.regime_analysis import RegimeAnalysis
from models.portfolio import Portfolio
from models.regime_model import RegimeModel

if __name__ == "__main__":

    portfolio = Portfolio.load(104)
    print("-----------------------------------------")
    from models.enrichment.price_enricher import YahooFinancePriceEnricher
    from models.enrichment.time_series_enricher import YahooTimeSeriesEnricher

    regime_model = RegimeModel.load()

    enrichers = [YahooFinancePriceEnricher(),YahooTimeSeriesEnricher()]
    for enricher in enrichers:
        enricher.enrich_portfolio(portfolio)

    regime_analysis = RegimeAnalysis(portfolio,  regime_model)
    print(regime_analysis.reg_stats)
    regime_analysis.reg_stats.to_csv("/tmp/regime_stats.csv")