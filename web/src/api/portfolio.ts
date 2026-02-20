import type { Portfolio } from "@/types/portfolio";
import type { AnalysisData } from "@/types/analysis";
import { apiFetch } from "./client";

export async function uploadPortfolio(
  file: File,
  nav?: number
): Promise<Portfolio> {
  const form = new FormData();
  form.append("file", file);
  if (nav != null) form.append("nav", String(nav));
  return apiFetch<Portfolio>("/portfolio/upload", {
    method: "POST",
    body: form,
  });
}

export async function analyzePortfolio(id: string): Promise<AnalysisData> {
  return apiFetch<AnalysisData>(`/portfolio/${id}/analyze`, {
    method: "POST",
  });
}

export async function getPortfolio(id: string): Promise<AnalysisData> {
  return apiFetch<AnalysisData>(`/portfolio/${id}`);
}
