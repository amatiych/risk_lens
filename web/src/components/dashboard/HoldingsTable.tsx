import type { Holding } from "@/types/portfolio";
import { formatCurrency, formatPercent } from "@/lib/formatters";

export function HoldingsTable({ holdings }: { holdings: Holding[] }) {
  const sorted = [...holdings].sort(
    (a, b) => (b.market_value ?? 0) - (a.market_value ?? 0)
  );

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-sm">Holdings</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-text-muted text-left">
              <th className="px-4 py-2">Ticker</th>
              <th className="px-4 py-2 text-right">Shares</th>
              <th className="px-4 py-2 text-right">Price</th>
              <th className="px-4 py-2 text-right">Market Value</th>
              <th className="px-4 py-2 text-right">Weight</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((h) => (
              <tr key={h.ticker} className="border-b border-border/50 hover:bg-surface-2">
                <td className="px-4 py-2 font-mono font-medium">{h.ticker}</td>
                <td className="px-4 py-2 text-right">{h.shares.toLocaleString()}</td>
                <td className="px-4 py-2 text-right">
                  {h.price != null ? `$${h.price.toFixed(2)}` : "-"}
                </td>
                <td className="px-4 py-2 text-right">
                  {h.market_value != null ? formatCurrency(h.market_value) : "-"}
                </td>
                <td className="px-4 py-2 text-right">
                  {h.weight != null ? formatPercent(h.weight) : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
