import { Navigate } from "react-router-dom";
import { PageLayout } from "@/components/layout/PageLayout";
import { useApp } from "@/context/AppContext";
import { HoldingsTable } from "./HoldingsTable";
import { VarCard } from "./VarCard";
import { VarTable } from "./VarTable";
import { CorrelationHeatmap } from "./CorrelationHeatmap";
import { FactorExposureTable } from "./FactorExposureTable";
import { RegimeTable } from "./RegimeTable";
import { PcaChart } from "./PcaChart";
import { AiSummaryCard } from "./AiSummaryCard";

export function DashboardPage() {
  const { analysis } = useApp();

  if (!analysis) return <Navigate to="/" replace />;

  return (
    <PageLayout title="Risk Analysis Dashboard">
      <div className="space-y-6">
        <AiSummaryCard summary={analysis.ai_summary} />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {analysis.var_results.map((v) => (
            <VarCard key={v.ci} data={v} />
          ))}
        </div>

        <HoldingsTable holdings={analysis.portfolio.holdings} />

        <VarTable data={analysis.var_results} />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <CorrelationHeatmap data={analysis.correlation} />
          <PcaChart data={analysis.pca} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <FactorExposureTable data={analysis.factor_exposures} />
          <RegimeTable data={analysis.regime_stats} />
        </div>
      </div>
    </PageLayout>
  );
}
