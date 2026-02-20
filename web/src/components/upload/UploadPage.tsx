import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2, FileDown } from "lucide-react";
import { PageLayout } from "@/components/layout/PageLayout";
import { CsvDropzone } from "./CsvDropzone";
import { useApp } from "@/context/AppContext";
import {
  uploadPortfolio,
  analyzePortfolio,
  listSamplePortfolios,
  downloadSamplePortfolio,
  type SamplePortfolio,
} from "@/api/portfolio";

export function UploadPage() {
  const navigate = useNavigate();
  const { setPortfolio, setAnalysis } = useApp();
  const [nav, setNav] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [samples, setSamples] = useState<SamplePortfolio[]>([]);

  useEffect(() => {
    listSamplePortfolios().then(setSamples).catch(() => {});
  }, []);

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

  async function handleSampleClick(filename: string) {
    const file = await downloadSamplePortfolio(filename);
    handleFile(file);
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

        {samples.length > 0 && !loading && (
          <div className="bg-surface border border-border rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-border">
              <h3 className="font-semibold text-sm">Sample Portfolios</h3>
              <p className="text-xs text-text-muted mt-1">
                Click to load and analyze a sample portfolio
              </p>
            </div>
            <div className="divide-y divide-border/50">
              {samples.map((s) => (
                <button
                  key={s.filename}
                  onClick={() => handleSampleClick(s.filename)}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-surface-2 transition-colors"
                >
                  <FileDown size={16} className="text-primary shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">{s.filename}</p>
                    <p className="text-xs text-text-muted truncate">
                      {s.num_holdings} holdings: {s.tickers.join(", ")}
                    </p>
                  </div>
                </button>
              ))}
            </div>
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
