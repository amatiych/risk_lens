import type { RegimeStat } from "@/types/analysis";
import { formatPercent } from "@/lib/formatters";

export function RegimeTable({ data }: { data: RegimeStat[] }) {
  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-sm">Regime Analysis</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-text-muted text-left">
              <th className="px-4 py-2">Regime</th>
              <th className="px-4 py-2">Description</th>
              <th className="px-4 py-2 text-right">Avg Performance</th>
            </tr>
          </thead>
          <tbody>
            {data.map((r) => (
              <tr key={r.regime} className="border-b border-border/50 hover:bg-surface-2">
                <td className="px-4 py-2 font-medium">{r.label}</td>
                <td className="px-4 py-2 text-text-muted text-xs">
                  {r.description}
                </td>
                <td
                  className={`px-4 py-2 text-right font-mono ${
                    r.performance >= 0 ? "text-accent" : "text-danger"
                  }`}
                >
                  {formatPercent(r.performance, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
