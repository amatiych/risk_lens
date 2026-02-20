"""Portfolio upload and analysis endpoints."""

import json
import uuid
import io
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from backend.api.schemas import (
    PortfolioResponse,
    AnalysisResponse,
    HoldingResponse,
    VarResultResponse,
    CorrelationResponse,
    FactorExposureResponse,
    RegimeStatResponse,
    PcaResponse,
)
from backend.api.dependencies import store, PortfolioEntry, get_provider
from frontend.services.portfolio_service import (
    parse_uploaded_csv,
    enrich_portfolio,
    run_analysis,
)
from backend.reporting.portfolio_report import PortfolioReport
from backend.llm.tools import TOOLS, execute_tool, set_portfolio_context
from backend.llm.providers import LLMConfig, get_provider as get_llm_provider

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])


def _get_ai_summary(report: PortfolioReport, provider_name: str) -> str:
    """Generate AI summary without depending on Streamlit."""
    set_portfolio_context(report.portfolio)
    config = LLMConfig(provider=provider_name)
    provider = get_llm_provider(config)

    system_prompt = f"""You are a senior risk analyst. Today is {datetime.today().strftime('%Y-%m-%d')}.
Analyze portfolios and provide concise executive summaries."""

    messages = [{
        "role": "user",
        "content": f"""Analyze this portfolio and provide a brief executive summary covering:
1. Overall risk profile (VaR interpretation)
2. Top 3 risk contributors
3. Key factor exposures
4. Regime sensitivity

Keep your response concise (under 300 words).

Portfolio Data:
{report.report}"""
    }]

    for _ in range(3):
        response = provider.create_message(
            system_prompt=system_prompt,
            messages=messages,
            tools=TOOLS,
            max_tokens=1000,
        )
        if response.stop_reason == "tool_use":
            tool_results = []
            for tool_call in response.tool_calls:
                try:
                    result = execute_tool(tool_call.name, tool_call.arguments)
                    tool_results.append(
                        provider.format_tool_result(tool_call.id, json.dumps(result))
                    )
                except Exception as e:
                    tool_results.append(
                        provider.format_tool_result(
                            tool_call.id, json.dumps({"error": str(e)}), is_error=True
                        )
                    )
            messages.append(
                provider.format_assistant_message(response.content, response.tool_calls)
            )
            messages.append({"role": "user", "content": tool_results})
        else:
            if response.content:
                return response.content.strip()
            break
    return "Unable to generate analysis."


def _portfolio_to_response(entry: PortfolioEntry) -> PortfolioResponse:
    p = entry.portfolio
    holdings = []
    for ticker, row in p.holdings.iterrows():
        holdings.append(HoldingResponse(
            ticker=ticker,
            shares=float(row["shares"]),
            price=float(row["price"]) if "price" in row.index else None,
            market_value=float(row["market_value"]) if "market_value" in row.index else None,
            weight=float(row["weight"]) if "weight" in row.index else None,
        ))
    status = "analyzed" if entry.results else ("enriched" if p.time_series is not None else "uploaded")
    pid = [k for k, v in store._data.items() if v is entry][0]
    return PortfolioResponse(id=pid, name=p.name, nav=p.nav, holdings=holdings, status=status)


def _build_analysis_response(pid: str, entry: PortfolioEntry) -> AnalysisResponse:
    results = entry.results
    portfolio_resp = _portfolio_to_response(entry)
    tickers = list(results.portfolio.time_series.columns)

    var_results = []
    for v in results.var_results:
        marginal = [{"ticker": t, "value": mv} for t, mv in zip(tickers, v.marginal_var)]
        incremental = [{"ticker": t, "value": iv} for t, iv in zip(tickers, v.incremental_var)]
        var_results.append(VarResultResponse(
            ci=v.ci,
            var=v.var,
            es=v.es,
            var_date=v.var_date.strftime("%Y-%m-%d") if hasattr(v.var_date, "strftime") else str(v.var_date),
            marginal_var=marginal,
            incremental_var=incremental,
        ))

    cr = results.correlation_matrix
    correlation = CorrelationResponse(
        tickers=list(cr.columns),
        matrix=cr.values.tolist(),
    )

    fr = results.factor_result
    factor_exposures = [
        FactorExposureResponse(factor=f, beta=float(b), risk_contribution=float(r))
        for f, b, r in zip(fr.factors, fr.betas, fr.marginal_risk)
    ]

    ra = results.regime_analysis
    regime_stats = []
    for _, row in ra.reg_stats.iterrows():
        regime_stats.append(RegimeStatResponse(
            regime=int(row["regime"]) if "regime" in row.index else 0,
            label=str(row.get("label", "")),
            description=str(row.get("description", "")),
            performance=float(row["performance"]),
        ))

    pca_obj = entry.report.pca
    pca = PcaResponse(
        variance_explained=[float(x) for x in pca_obj.var_pct],
        cumulative_variance=[float(x) for x in pca_obj.cum_var_pct],
    )

    return AnalysisResponse(
        portfolio=portfolio_resp,
        var_results=var_results,
        correlation=correlation,
        factor_exposures=factor_exposures,
        regime_stats=regime_stats,
        pca=pca,
        ai_summary=entry.ai_summary,
    )


@router.post("/upload", response_model=PortfolioResponse)
async def upload_portfolio(
    file: UploadFile = File(...),
    nav: Optional[float] = Form(None),
):
    content = await file.read()
    csv_file = io.BytesIO(content)

    try:
        portfolio = parse_uploaded_csv(csv_file, nav=nav)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    pid = str(uuid.uuid4())
    entry = PortfolioEntry(portfolio=portfolio)
    store.set(pid, entry)

    return _portfolio_to_response(entry)


@router.post("/{portfolio_id}/analyze", response_model=AnalysisResponse)
async def analyze_portfolio(portfolio_id: str):
    entry = store.get(portfolio_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    try:
        enrich_portfolio(entry.portfolio)
        results = run_analysis(entry.portfolio)
        entry.results = results
        entry.report = results.report

        try:
            entry.ai_summary = _get_ai_summary(results.report, get_provider())
        except Exception:
            entry.ai_summary = None

        return _build_analysis_response(portfolio_id, entry)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}: {str(e)}")


@router.get("/{portfolio_id}", response_model=AnalysisResponse)
async def get_portfolio(portfolio_id: str):
    entry = store.get(portfolio_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if not entry.results:
        raise HTTPException(status_code=400, detail="Portfolio not yet analyzed")

    return _build_analysis_response(portfolio_id, entry)
