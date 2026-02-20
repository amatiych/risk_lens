import Markdown from "react-markdown";
import { User, Bot } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/types/chat";

export function ChatMessage({ msg }: { msg: ChatMessageType }) {
  const isUser = msg.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : ""}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center shrink-0">
          <Bot size={16} className="text-primary" />
        </div>
      )}
      <div
        className={`max-w-[80%] rounded-xl px-4 py-3 text-sm ${
          isUser
            ? "bg-primary text-white"
            : "bg-surface border border-border"
        }`}
      >
        {isUser ? (
          <p>{msg.content}</p>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none">
            <Markdown>{msg.content}</Markdown>
          </div>
        )}
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-surface-2 flex items-center justify-center shrink-0">
          <User size={16} className="text-text-muted" />
        </div>
      )}
    </div>
  );
}
