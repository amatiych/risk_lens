import { useState, useRef, useEffect } from "react";
import { Navigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useApp } from "@/context/AppContext";
import { streamChat } from "@/api/chat";
import type { ChatMessage as ChatMessageType, GuardrailsSummary } from "@/types/chat";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { SuggestedQuestions } from "./SuggestedQuestions";
import { ProviderToggle } from "./ProviderToggle";
import { GuardrailsBadge } from "./GuardrailsBadge";

export function ChatPage() {
  const { portfolio, analysis } = useApp();
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [guardrails, setGuardrails] = useState<GuardrailsSummary | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  if (!analysis || !portfolio) return <Navigate to="/" replace />;

  async function sendMessage(text: string) {
    const userMsg: ChatMessageType = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setStreaming(true);
    setGuardrails(null);

    let assistantContent = "";
    const history = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      for await (const event of streamChat(portfolio!.id, text, history)) {
        if (event.type === "text" && event.content) {
          assistantContent += event.content;
          setMessages((prev) => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last?.role === "assistant") {
              copy[copy.length - 1] = { ...last, content: assistantContent };
            } else {
              copy.push({ role: "assistant", content: assistantContent });
            }
            return copy;
          });
        } else if (event.type === "done" && event.guardrails) {
          setGuardrails(event.guardrails);
        }
      }
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : "Chat failed";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errMsg}` },
      ]);
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="flex h-screen">
      <div className="flex-1 flex flex-col">
        <header className="border-b border-border px-6 py-4">
          <h2 className="text-xl font-semibold">AI Chat</h2>
          <p className="text-xs text-text-muted">
            Ask questions about your portfolio risk analysis
          </p>
        </header>

        <div className="flex-1 overflow-auto p-6 space-y-4">
          {messages.length === 0 && (
            <p className="text-text-muted text-center py-12">
              Start a conversation about your portfolio
            </p>
          )}
          {messages.map((m, i) => (
            <ChatMessage key={i} msg={m} />
          ))}
          {streaming && (
            <div className="flex items-center gap-2 text-text-muted text-sm">
              <Loader2 size={14} className="animate-spin" />
              Thinking...
            </div>
          )}
          {guardrails && <GuardrailsBadge data={guardrails} />}
          <div ref={bottomRef} />
        </div>

        <div className="border-t border-border p-4">
          <ChatInput onSend={sendMessage} disabled={streaming} />
        </div>
      </div>

      <aside className="w-64 border-l border-border bg-surface p-4 space-y-6 hidden lg:block">
        <ProviderToggle />
        <SuggestedQuestions onSelect={sendMessage} disabled={streaming} />
      </aside>
    </div>
  );
}
