export interface MarginalVarEntry {
  ticker: string;
  value: number;
}

export interface VarResult {
  ci: number;
  var: number;
  es: number;
  var_date: string;
  marginal_var: MarginalVarEntry[];
  incremental_var: MarginalVarEntry[];
}

export interface Correlation {
  tickers: string[];
  matrix: number[][];
}

export interface FactorExposure {
  factor: string;
  beta: number;
  risk_contribution: number;
}

export interface RegimeStat {
  regime: number;
  label: string;
  description: string;
  performance: number;
}

export interface PcaData {
  variance_explained: number[];
  cumulative_variance: number[];
}

export interface AnalysisData {
  portfolio: import("./portfolio").Portfolio;
  var_results: VarResult[];
  correlation: Correlation;
  factor_exposures: FactorExposure[];
  regime_stats: RegimeStat[];
  pca: PcaData;
  ai_summary: string | null;
}
