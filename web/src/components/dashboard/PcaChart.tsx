import type { PcaData } from "@/types/analysis";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

export function PcaChart({ data }: { data: PcaData }) {
  const chartData = data.variance_explained.map((v, i) => ({
    component: `PC${i + 1}`,
    variance: +(v * 100).toFixed(2),
    cumulative: +(data.cumulative_variance[i] * 100).toFixed(2),
  }));

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-sm">PCA Variance Explained</h3>
      </div>
      <div className="p-4 h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis dataKey="component" tick={{ fill: "#a3a3a3", fontSize: 12 }} />
            <YAxis tick={{ fill: "#a3a3a3", fontSize: 12 }} unit="%" />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e1e1e",
                border: "1px solid #2a2a2a",
                borderRadius: 8,
                color: "#e5e5e5",
              }}
            />
            <Legend />
            <Bar dataKey="variance" name="Variance %" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Line
              dataKey="cumulative"
              name="Cumulative %"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ fill: "#10b981", r: 3 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
