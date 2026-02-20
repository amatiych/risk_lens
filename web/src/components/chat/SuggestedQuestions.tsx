const QUESTIONS = [
  "What are my top risk contributors?",
  "How diversified is my portfolio?",
  "What happens in a bear market?",
  "Which factors drive my risk?",
  "What stocks should I add for diversification?",
  "Explain my VaR results",
];

interface Props {
  onSelect: (q: string) => void;
  disabled: boolean;
}

export function SuggestedQuestions({ onSelect, disabled }: Props) {
  return (
    <div className="space-y-2">
      <p className="text-xs text-text-muted font-medium uppercase tracking-wider">
        Suggested Questions
      </p>
      <div className="space-y-1">
        {QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => !disabled && onSelect(q)}
            disabled={disabled}
            className="w-full text-left text-sm px-3 py-2 rounded-lg text-text-muted hover:text-text hover:bg-surface-2 transition-colors disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
