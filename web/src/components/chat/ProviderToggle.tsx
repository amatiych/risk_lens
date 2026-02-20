import { useApp } from "@/context/AppContext";
import { setProvider as apiSetProvider } from "@/api/config";

export function ProviderToggle() {
  const { provider, setProvider } = useApp();

  async function toggle() {
    const next = provider === "claude" ? "openai" : "claude";
    await apiSetProvider(next);
    setProvider(next);
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-text-muted font-medium uppercase tracking-wider">
        LLM Provider
      </p>
      <button
        onClick={toggle}
        className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-2 border border-border text-sm hover:border-primary/50 transition-colors"
      >
        <span
          className={`w-2 h-2 rounded-full ${
            provider === "claude" ? "bg-primary" : "bg-accent"
          }`}
        />
        {provider === "claude" ? "Claude" : "OpenAI"}
      </button>
    </div>
  );
}
