import Markdown from "react-markdown";

export function AiSummaryCard({ summary }: { summary: string | null }) {
  if (!summary) return null;

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="font-semibold text-sm">AI Executive Summary</h3>
      </div>
      <div className="p-4 prose prose-invert prose-sm max-w-none text-text-muted">
        <Markdown>{summary}</Markdown>
      </div>
    </div>
  );
}
