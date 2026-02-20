import { apiFetch } from "./client";

export async function getProvider(): Promise<string> {
  const data = await apiFetch<{ provider: string }>("/config/provider");
  return data.provider;
}

export async function setProvider(provider: string): Promise<void> {
  await apiFetch("/config/provider", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
}
