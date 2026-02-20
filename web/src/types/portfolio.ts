export interface Holding {
  ticker: string;
  shares: number;
  price: number | null;
  market_value: number | null;
  weight: number | null;
}

export interface Portfolio {
  id: string;
  name: string;
  nav: number;
  holdings: Holding[];
  status: "uploaded" | "enriched" | "analyzed";
}
