import type { ChatMessage, ChatSSEEvent } from "@/types/chat";

export async function* streamChat(
  portfolioId: string,
  message: string,
  history: ChatMessage[]
): AsyncGenerator<ChatSSEEvent> {
  const res = await fetch(`/api/v1/chat/${portfolioId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? "Chat request failed");
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const json = line.slice(6).trim();
        if (json) {
          yield JSON.parse(json) as ChatSSEEvent;
        }
      }
    }
  }
}
