import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";
import type { Portfolio } from "@/types/portfolio";
import type { AnalysisData } from "@/types/analysis";

interface AppState {
  portfolio: Portfolio | null;
  analysis: AnalysisData | null;
  provider: string;
  setPortfolio: (p: Portfolio | null) => void;
  setAnalysis: (a: AnalysisData | null) => void;
  setProvider: (p: string) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [provider, setProvider] = useState("claude");

  return (
    <AppContext.Provider
      value={{
        portfolio,
        analysis,
        provider,
        setPortfolio,
        setAnalysis,
        setProvider,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
