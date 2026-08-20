/** Types mirroring understudy/api.py, and the fetchers that use them. */

export type Engine = string;

export interface Health {
  ok: boolean;
  artifacts: boolean;
  calls_enabled: boolean;
  claude_available: boolean;
  engines: Record<string, string>;
  engines_available: Record<string, boolean>;
  max_claude_runs: number;
  listings: number;
  skus: string[];
  strategies: string[];
}

export interface Report {
  listings_total: number;
  listings_real: number;
  sold_real: number;
  sold_synthetic: number;
  synthetic_sold_data: boolean;
  calibration: string;
  eval: string;
}

export interface ListingRow {
  id: string;
  title: string;
  url: string;
  ask_price: number;
  sku_id: string;
  condition: string;
  seller_type: string;
  accepts_offers: boolean;
  condition_flags: string[];
  days_listed: number;
  price_cuts: number;
  sold_median: number;
  over_median: number;
  why: string[];
  rank: number;
  most_negotiable: boolean;
}

export interface Packet {
  listing_id: string;
  headline: string;
  facts: string[];
  ask_price: number;
  target: number;
  opening: number;
  walk_away: number;
}

export interface Turn {
  speaker: "buyer" | "seller";
  text: string;
}

/** The vocabulary a live Vapi call and a replayed simulation both speak. */
export type CallEvent =
  | { type: "transcript"; speaker: "buyer" | "seller"; text: string; final?: boolean }
  | { type: "offer"; price: number }
  | { type: "deal"; price: number }
  | { type: "call-ended"; outcome: string; reason?: string }
  | { type: "status"; status: string }
  | { type: "call-started"; listing?: string; ask_price?: number; target?: number }
  | { type: "hello"; clients: number };

export interface SimResponse {
  /** What actually generated the transcript. Displayed verbatim; never inferred. */
  engine: string;
  requested_runs: number;
  runs: number;
  listing: { id: string; title: string; ask_price: number; sku_id: string };
  packet: Packet;
  sold_prices: number[];
  sold_median: number;
  p_deal: number;
  settles: number[];
  results: {
    outcome: string;
    final_price: number | null;
    turns: number;
    reservation_ratio: number | null;
    floor: number | null;
  }[];
  transcript: Turn[];
  replay: CallEvent[];
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw Object.assign(new Error(`${url} → ${res.status}`), { res });
  return res.json() as Promise<T>;
}

export const getHealth = () => json<Health>("/health");
export const getReport = () => json<Report>("/report");
export const getListings = (limit = 200) => json<ListingRow[]>(`/listings?limit=${limit}`);

export async function simulate(
  listing_id: string, n: number, strategy: string, llm: Engine,
): Promise<SimResponse> {
  const res = await fetch("/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ listing_id, n, strategy, llm }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `simulate → ${res.status}` }));
    throw new Error(body.detail);
  }
  return res.json() as Promise<SimResponse>;
}

export async function negotiate(listing_id: string, strategy: string): Promise<string | null> {
  const res = await fetch("/negotiate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ listing_id, strategy }),
  });
  if (res.ok) return null;
  const body = await res.json().catch(() => ({ detail: "Call failed." }));
  return body.detail as string;
}

export const usd = (n: number | null | undefined) =>
  n == null ? "$—" : "$" + Math.round(n).toLocaleString();
