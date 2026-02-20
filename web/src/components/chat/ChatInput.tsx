import { useState } from "react";
import { Send } from "lucide-react";

interface Props {
  onSend: (message: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setValue("");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask about your portfolio risk..."
        disabled={disabled}
        className="flex-1 bg-surface border border-border rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-primary disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="bg-primary hover:bg-primary-hover text-white rounded-lg px-4 py-2.5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <Send size={18} />
      </button>
    </form>
  );
}
