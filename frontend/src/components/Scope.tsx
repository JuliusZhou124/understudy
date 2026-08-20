import type { SimResponse } from "../api";
import { usd } from "../api";

const BINS = 36;

function histogram(values: number[], lo: number, hi: number) {
  const out = new Array(BINS).fill(0);
  const width = (hi - lo) / BINS || 1;
  for (const v of values) {
    out[Math.min(BINS - 1, Math.max(0, Math.floor((v - lo) / width)))] += 1;
  }
  return out;
}

function Lane({
  label, values, kind, lo, hi, hatched,
}: {
  label: string; values: number[]; kind: "observed" | "simulated";
  lo: number; hi: number; hatched: boolean;
}) {
  const bins = histogram(values, lo, hi);
  const peak = Math.max(1, ...bins);
  return (
    <div className="lane">
      <span className="lane__label">
        {label}
        {values.length === 0 && <em style={{ textTransform: "none" }}> no deals closed</em>}
      </span>
      {bins.map((count, i) =>
        count === 0 ? null : (
          <div
            key={i}
            className={`bar bar--${kind}${hatched ? " bar--synthetic" : ""}`}
            style={{
              left: `${(i / BINS) * 100}%`,
              width: `${100 / BINS}%`,
              height: `${(count / peak) * 76}%`,
            }}
          />
        ),
      )}
    </div>
  );
}

/**
 * The signature: observed sold prices and simulated settles on one shared
 * price axis, so "does the simulated world match the real one" is something
 * you look at rather than a number you take on faith.
 */
export function Scope({ data, synthetic }: { data: SimResponse; synthetic: boolean }) {
  const all = [...data.sold_prices, ...data.settles, data.listing.ask_price];
  if (all.length === 0) return null;

  const lo = Math.min(...all) * 0.95;
  const hi = Math.max(...all) * 1.05;
  const pos = (v: number) => ((v - lo) / (hi - lo)) * 100;

  return (
    <figure className="scope">
      <figcaption>
        <span className="key"><i className="sw sw--observed" />Observed sold</span>
        <span className="key"><i className="sw sw--simulated" />Simulated settles</span>
      </figcaption>
      <div id="plot">
        <Lane label={synthetic ? "Observed sold (synthetic)" : "Observed sold"}
              values={data.sold_prices} kind="observed" lo={lo} hi={hi} hatched={synthetic} />
        <Lane label="Simulated settles" values={data.settles} kind="simulated"
              lo={lo} hi={hi} hatched={false} />
        <div className="lane lane--axis">
          <div className="tick" style={{ left: `${pos(data.sold_median)}%` }}>
            <span>median {usd(data.sold_median)}</span>
          </div>
          {/* Second row: ask and median sit close together whenever a listing is
              priced near the market, and the labels would otherwise overlap. */}
          <div className="tick" style={{ left: `${pos(data.listing.ask_price)}%` }}>
            <span className="tick__low">ask {usd(data.listing.ask_price)}</span>
          </div>
        </div>
        <div className="axis"><span>{usd(lo)}</span><span>{usd(hi)}</span></div>
      </div>
    </figure>
  );
}
