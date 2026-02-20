import type { VarResult } from "@/types/analysis";
import { formatPercent } from "@/lib/formatters";

export function VarCard({ data }: { data: VarResult }) {
  const ciLabel = `${(data.ci * 100).toFixed(0)}%`;

  return (
    <div className="bg-surface border border-border rounded-xl p-5">
      <h3 className="text-sm text-text-muted mb-3">
        {ciLabel} Confidence VaR
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-2xl font-bold text-danger">
            {formatPercent(data.var)}
          </p>
          <p className="text-xs text-text-muted">Value at Risk</p>
        </div>
        <div>
          <p className="text-2xl font-bold text-warning">
            {formatPercent(data.es)}
          </p>
          <p className="text-xs text-text-muted">Expected Shortfall</p>
        </div>
      </div>
      <p className="text-xs text-text-muted mt-3">
        VaR Date: {data.var_date}
      </p>
    </div>
  );
}
