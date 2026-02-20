import type { FactorExposure } from "@/types/analysis";
import { formatPercent, formatNumber } from "@/lib/formatters";

export function FactorExposureTable({ data }: { data: FactorExposure[] }) {
  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-sm">Factor Exposures</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-text-muted text-left">
              <th className="px-4 py-2">Factor</th>
              <th className="px-4 py-2 text-right">Beta</th>
              <th className="px-4 py-2 text-right">Risk Contribution</th>
            </tr>
          </thead>
          <tbody>
            {data.map((f) => (
              <tr key={f.factor} className="border-b border-border/50 hover:bg-surface-2">
                <td className="px-4 py-2 font-medium">{f.factor}</td>
                <td className="px-4 py-2 text-right font-mono">
                  {formatNumber(f.beta, 4)}
                </td>
                <td className="px-4 py-2 text-right font-mono">
                  {formatPercent(f.risk_contribution)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
