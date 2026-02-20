import type { VarResult } from "@/types/analysis";
import { formatPercent } from "@/lib/formatters";

export function VarTable({ data }: { data: VarResult[] }) {
  if (!data.length) return null;
  const primary = data[0];
  const tickers = primary.marginal_var.map((m) => m.ticker);

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-sm">Marginal & Incremental VaR</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-text-muted text-left">
              <th className="px-4 py-2">Ticker</th>
              {data.map((v) => (
                <th key={`m-${v.ci}`} className="px-4 py-2 text-right">
                  Marginal ({(v.ci * 100).toFixed(0)}%)
                </th>
              ))}
              {data.map((v) => (
                <th key={`i-${v.ci}`} className="px-4 py-2 text-right">
                  Incremental ({(v.ci * 100).toFixed(0)}%)
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((ticker, idx) => (
              <tr key={ticker} className="border-b border-border/50 hover:bg-surface-2">
                <td className="px-4 py-2 font-mono">{ticker}</td>
                {data.map((v) => (
                  <td key={`m-${v.ci}-${ticker}`} className="px-4 py-2 text-right">
                    {formatPercent(v.marginal_var[idx].value)}
                  </td>
                ))}
                {data.map((v) => (
                  <td key={`i-${v.ci}-${ticker}`} className="px-4 py-2 text-right">
                    {formatPercent(v.incremental_var[idx].value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
