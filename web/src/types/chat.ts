export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface GuardrailsSummary {
  status: "passed" | "warnings" | "blocked";
  total_checks: number;
  warnings: { guard: string; message: string }[];
  errors: { guard: string; message: string }[];
}

export interface ChatSSEEvent {
  type: "text" | "tool_status" | "done";
  content?: string;
  guardrails?: GuardrailsSummary;
}
