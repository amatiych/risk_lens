import type { GuardrailsSummary } from "@/types/chat";
import { ShieldAlert, ShieldCheck } from "lucide-react";

export function GuardrailsBadge({ data }: { data: GuardrailsSummary }) {
  const Icon =
    data.status === "blocked"
      ? ShieldAlert
      : data.status === "warnings"
        ? ShieldAlert
        : ShieldCheck;

  const color =
    data.status === "blocked"
      ? "text-danger"
      : data.status === "warnings"
        ? "text-warning"
        : "text-accent";

  return (
    <div className="flex items-center gap-2 text-xs text-text-muted">
      <Icon size={14} className={color} />
      <span>
        {data.total_checks} checks &middot; {data.status}
      </span>
      {data.warnings.length > 0 && (
        <span className="text-warning">
          ({data.warnings.length} warning{data.warnings.length > 1 ? "s" : ""})
        </span>
      )}
    </div>
  );
}
