export function correlationColor(value: number): string {
  if (value >= 0.7) return "#ef4444";
  if (value >= 0.3) return "#f59e0b";
  if (value >= -0.3) return "#a3a3a3";
  if (value >= -0.7) return "#3b82f6";
  return "#2563eb";
}

export const CHART_COLORS = [
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
];
