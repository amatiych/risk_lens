from pandas import DataFrame, read_csv
from dataclasses import dataclass

@dataclass
class Portfolio:
    portfolio_id: int
    name: str
    holdings: DataFrame

    @classmethod
    def get_portfolios(self)->DataFrame:
        return read_csv("data/portfolios.csv").set_index("portfolio_id")

    @classmethod
    def load(cls, portfolio_id: str) -> "Portfolio":
        meta = read_csv("data/portfolios.csv").set_index("portfolio_id")
        row = meta.loc[portfolio_id]
        holdings = read_csv(f"data/holdings/port_{portfolio_id}.csv").set_index('ticker')
        return cls(
            portfolio_id=portfolio_id,
            name=row["name"],
            holdings=holdings
        )

if __name__ == "__main__":

    print("-----------------------------------------")
    all_portfolios = Portfolio.get_portfolios()
    print(all_portfolios)

    print("-----------------------------------------")
    portfolio = Portfolio.load(101)
    from enrichment.price_enricher import YahooFinancePriceEnricher
    from enrichment.time_series_enricher import YahooTimeSeriesEnricher

    enrichers = [YahooFinancePriceEnricher(),YahooTimeSeriesEnricher()]
    for enricher in enrichers:
        enricher.enrich_portfolio(portfolio)
    print("-----------------------------------------")
    print(portfolio.holdings)

    print("-----------------------------------------")
    print(portfolio.time_series)

    print("------------------------------------------")
    print("TICKERS")
    tickers = list(portfolio.time_series.columns)

    print(tickers)

    weights = list(portfolio.holdings.loc[tickers,'weight'].values)
    print("WEIGHTS")
    print(weights)
    from backend.risk_engine.var.var_engine import VarEngine
    var_engine = VarEngine(portfolio.time_series,weights)
    var = var_engine.calc_var()
    print(var)