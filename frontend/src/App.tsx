import { useCallback, useEffect, useRef, useState } from "react";

import {
  getHealth, getListings, getReport, negotiate, simulate,
  type Engine, type Health, type ListingRow, type Report, type SimResponse,
} from "./api";
import { Inventory } from "./components/Inventory";
import { Masthead, ProvenanceNote } from "./components/Masthead";
import { Rig } from "./components/Rig";
import { CallCard } from "./components/CallCard";
import { useCall, useLiveEvents } from "./hooks/useCall";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [rows, setRows] = useState<ListingRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [sim, setSim] = useState<SimResponse | null>(null);
  const [strategy, setStrategy] = useState("two_offer_close");
  const [engine, setEngine] = useState<Engine>("stub");
  const [runs, setRuns] = useState(16);
  const [simError, setSimError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const call = useCall();
  useLiveEvents(call.event);

  // The inventory grid is long; without this the rig opens far below the fold
  // and clicking a card looks like it did nothing at all.
  const rigRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!selected) return;
    rigRef.current?.scrollIntoView({
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      block: "start",
    });
  }, [selected]);

  useEffect(() => {
    Promise.all([getHealth(), getReport(), getListings()])
      .then(([h, r, l]) => { setHealth(h); setReport(r); setRows(l); })
      .catch((e: Error) => setError(e.message));
  }, []);

  const run = useCallback(async (listingId: string, strat: string, n: number, eng: Engine) => {
    setBusy(true);
    setSimError(null);
    try {
      setSim(await simulate(listingId, n, strat, eng));
    } catch (e) {
      // A failed run must not leave a stale transcript on screen labelled with
      // the engine the user just asked for.
      setSim(null);
      setSimError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  const select = (id: string) => { setSelected(id); void run(id, strategy, runs, engine); };
  const changeStrategy = (s: string) => {
    setStrategy(s);
    if (selected) void run(selected, s, runs, engine);
  };
  const changeEngine = (e: Engine) => {
    setEngine(e);
    if (selected) void run(selected, strategy, runs, e);
  };

  const replay = () => {
    if (!sim) return;
    call.open(
      `Rehearsing — ${sim.listing.title}`.slice(0, 70),
      `Replayed simulation · ${strategy} · ${sim.engine}`,
      sim.listing.ask_price, sim.sold_median,
    );
    call.play(sim.replay);
  };

  const placeCall = async () => {
    if (!selected || !sim) return;
    call.open("Calling the seller…", sim.listing.title.slice(0, 60),
              sim.listing.ask_price, sim.sold_median);
    const refusal = await negotiate(selected, strategy);
    if (refusal) {
      call.say("system", refusal);
      call.say("system",
        "Tip: “Replay as call” drives this same card from a simulation — no phone needed.");
    }
  };

  if (error) return <div className="sheet"><section className="block"><h2>Error</h2><pre>{error}</pre></section></div>;
  if (!health || !report) return <div className="sheet"><section className="block"><p className="empty">Loading…</p></section></div>;

  return (
    <>
      <div className="sheet">
        <Masthead health={health} report={report} />
        {report.synthetic_sold_data && <ProvenanceNote />}

        <Inventory rows={rows} selected={selected} onSelect={select} />

        <div ref={rigRef}>
          {selected && (
            <Rig
              data={sim}
              title={rows.find((r) => r.id === selected)?.title ?? "Rehearsal rig"}
              synthetic={report.synthetic_sold_data}
              health={health}
              strategy={strategy} runs={runs} engine={engine}
              busy={busy} error={simError}
              onStrategy={changeStrategy} onRuns={setRuns} onEngine={changeEngine}
              onRun={() => run(selected, strategy, runs, engine)}
              onReplay={replay} onCall={placeCall}
            />
          )}
        </div>

        <section className="block">
          <div className="block__head">
            <h2>Measurements</h2>
            <span className="panel__hint">
              measured inside the simulator — no human seller has been involved
            </span>
          </div>
          <div className="cols">
            <div><h3>Calibration</h3><pre>{report.calibration}</pre></div>
            <div><h3>Strategy A/B</h3><pre>{report.eval}</pre></div>
          </div>
        </section>
      </div>

      <CallCard state={call.state} onClose={call.close} />
    </>
  );
}
