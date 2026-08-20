import type { Health, Report } from "../api";

export function Masthead({ health, report }: { health: Health; report: Report }) {
  const stats: [string, string | number][] = [
    ["Listings", report.listings_real],
    ["SKUs", health.skus.length],
    ["Sold rows", report.sold_real + report.sold_synthetic],
    ["Calls", health.calls_enabled ? "live" : "off"],
  ];
  return (
    <header className="masthead">
      <div className="wordmark">
        Understudy <span>· Market sheet</span>
      </div>
      <dl className="readout">
        {stats.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </header>
  );
}

export function ProvenanceNote() {
  return (
    <div className="stamp-note">
      <span className="hatch-key" aria-hidden="true" />
      <p>
        <strong>Sold prices on this sheet are synthetic.</strong> eBay CAPTCHA-walls
        sold-listing search, so settle prices come from a documented process in{" "}
        <code>understudy/synthetic.py</code>. Anything hatched is synthetic. Calibration here
        measures whether the estimator recovers a known truth — not the market.
      </p>
    </div>
  );
}
