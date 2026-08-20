import { usd, type Engine, type Health, type SimResponse } from "../api";
import { Scope } from "./Scope";

export function Rig({
  data, title, synthetic, health, strategy, runs, engine, busy, error,
  onStrategy, onRuns, onEngine, onRun, onReplay, onCall,
}: {
  data: SimResponse | null;
  title: string;
  synthetic: boolean;
  health: Health;
  strategy: string;
  runs: number;
  engine: Engine;
  busy: boolean;
  error: string | null;
  onStrategy: (s: string) => void;
  onRuns: (n: number) => void;
  onEngine: (e: Engine) => void;
  onRun: () => void;
  onReplay: () => void;
  onCall: () => void;
}) {
  // Providers come from the server, so adding one needs no frontend change.
  const picker = (
    <select value={engine} onChange={(e) => onEngine(e.target.value as Engine)}
            title="Which model negotiates">
      {Object.entries(health.engines).map(([key, label]) => {
        const usable = health.engines_available[key];
        return (
          <option key={key} value={key} disabled={!usable}>
            {label}{usable ? "" : " (no API key)"}
          </option>
        );
      })}
    </select>
  );

  if (!data) {
    return (
      <section className="block">
        <div className="block__head"><h2>{title}</h2></div>
        <p className="empty">{error ?? "Simulating negotiations…"}</p>
      </section>
    );
  }
  const p = data.packet;
  const ladder: [string, string][] = [
    ["Ask", usd(data.listing.ask_price)],
    ["Opening", usd(p.opening)],
    ["Target", usd(p.target)],
    ["Walk away", usd(p.walk_away)],
    ["P(deal)", `${Math.round(data.p_deal * 100)}%`],
  ];

  return (
    <section className="block">
      <div className="block__head">
        <h2>{data.listing.title}</h2>
        <div className="fbar">
          <select value={strategy} onChange={(e) => onStrategy(e.target.value)}>
            {health.strategies.map((s) => <option key={s}>{s}</option>)}
          </select>
          {picker}
          <input id="runs" type="number" min={1} max={60} value={runs}
                 onChange={(e) => onRuns(Number(e.target.value))} title="runs" />
          <button className="btn" onClick={onRun} disabled={busy}>
            {busy ? "Running…" : "Run simulation"}
          </button>
          <button className="btn btn--stamp" onClick={onReplay}>Replay as call</button>
          <button className="btn" onClick={onCall}>Call seller</button>
        </div>
      </div>

      <Scope data={data} synthetic={synthetic} />

      <div className="cols">
        <div>
          <h3>Briefing packet <em>the agent's entire fact base</em></h3>
          <ul className="facts">
            {p.facts.map((f) => <li key={f}>{f}</li>)}
          </ul>
          <div className="dfacts">
            {ladder.map(([label, value]) => (
              <div className="dfact" key={label}>
                <div className="lbl">{label}</div>
                <div className="val">{value}</div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h3>
            Transcript
            <em>first run</em>
            <span className={`engine-tag${engine === "claude" ? " engine-tag--live" : ""}`}>
              {data.engine}
            </span>
          </h3>
          {data.runs < data.requested_runs && (
            <p className="empty">
              Capped to {data.runs} runs (asked for {data.requested_runs}) — each Claude run is a
              full multi-turn call.
            </p>
          )}
          <ol className="feed feed--static">
            {data.transcript.map((t, i) => (
              <li className={`msg ${t.speaker}`} key={i}>
                <span className="who">{t.speaker}</span>
                {t.text}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
