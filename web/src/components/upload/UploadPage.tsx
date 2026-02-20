import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { PageLayout } from "@/components/layout/PageLayout";
import { CsvDropzone } from "./CsvDropzone";
import { useApp } from "@/context/AppContext";
import { uploadPortfolio, analyzePortfolio } from "@/api/portfolio";

export function UploadPage() {
  const navigate = useNavigate();
  const { setPortfolio, setAnalysis } = useApp();
  const [nav, setNav] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  async function handleFile(file: File) {
    setLoading(true);
    setError("");
    try {
      setStatus("Uploading portfolio...");
      const navVal = nav ? parseFloat(nav) : undefined;
      const portfolio = await uploadPortfolio(file, navVal);
      setPortfolio(portfolio);

      setStatus("Running risk analysis (this may take 10-15 seconds)...");
      const analysis = await analyzePortfolio(portfolio.id);
      setAnalysis(analysis);
      setPortfolio(analysis.portfolio);

      navigate("/dashboard");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
      setStatus("");
    }
  }

  return (
    <PageLayout title="Upload Portfolio">
      <div className="max-w-2xl mx-auto space-y-6">
        <div className="space-y-2">
          <label className="text-sm text-text-muted block">
            Net Asset Value (optional)
          </label>
          <input
            type="number"
            placeholder="e.g. 10000000"
            value={nav}
            onChange={(e) => setNav(e.target.value)}
            disabled={loading}
            className="w-full bg-surface border border-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-primary"
          />
        </div>

        {loading ? (
          <div className="flex flex-col items-center gap-3 py-12">
            <Loader2 className="animate-spin text-primary" size={40} />
            <p className="text-text-muted">{status}</p>
          </div>
        ) : (
          <CsvDropzone onFile={handleFile} />
        )}

        {error && (
          <div className="bg-danger/10 border border-danger/30 text-danger rounded-lg p-4 text-sm">
            {error}
          </div>
        )}

        <div className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-medium mb-2">Sample CSV Format</h3>
          <pre className="text-xs text-text-muted font-mono">
{`ticker,shares
AAPL,100
MSFT,50
GOOGL,30
AMZN,20`}
          </pre>
        </div>
      </div>
    </PageLayout>
  );
}
