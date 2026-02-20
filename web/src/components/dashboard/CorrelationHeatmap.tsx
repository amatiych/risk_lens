import type { Correlation } from "@/types/analysis";

function colorForValue(v: number): string {
  if (v >= 0.8) return "#dc2626";
  if (v >= 0.5) return "#ef4444";
  if (v >= 0.2) return "#f97316";
  if (v >= -0.2) return "#6b7280";
  if (v >= -0.5) return "#3b82f6";
  return "#2563eb";
}

export function CorrelationHeatmap({ data }: { data: Correlation }) {
  const { tickers, matrix } = data;
  const size = Math.min(40, Math.floor(600 / tickers.length));

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-sm">Correlation Matrix</h3>
      </div>
      <div className="p-4 overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th />
              {tickers.map((t) => (
                <th
                  key={t}
                  className="font-mono px-1 text-text-muted"
                  style={{ width: size, minWidth: size }}
                >
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((row, i) => (
              <tr key={row}>
                <td className="font-mono text-text-muted pr-2 text-right">
                  {row}
                </td>
                {matrix[i].map((val, j) => (
                  <td
                    key={j}
                    style={{
                      backgroundColor: colorForValue(val),
                      width: size,
                      height: size,
                    }}
                    className="text-center text-white/80"
                    title={`${row}/${tickers[j]}: ${val.toFixed(2)}`}
                  >
                    {tickers.length <= 12 ? val.toFixed(1) : ""}
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
