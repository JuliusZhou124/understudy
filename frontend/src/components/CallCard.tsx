import { useEffect, useRef } from "react";

import { usd } from "../api";
import type { CallState } from "../hooks/useCall";
import { Confetti } from "./Confetti";

function Receipt({ ask, median, final }: { ask: number; median: number; final: number }) {
  const saved = ask - final;
  const vsMedian = median - final;
  return (
    <div className="receipt">
      <div className="receipt__row"><span>Asking</span><b>{usd(ask)}</b></div>
      <div className="receipt__row"><span>Agreed</span><b>{usd(final)}</b></div>
      <div className="receipt__row receipt__row--big">
        <span>Saved</span>
        <b>{usd(saved)} ({Math.round((saved / ask) * 100)}%)</b>
      </div>
      <div className="receipt__row">
        <span>vs sold median</span>
        <b>{vsMedian >= 0 ? `${usd(vsMedian)} under` : `${usd(-vsMedian)} over`}</b>
      </div>
    </div>
  );
}

export function CallCard({ state, onClose }: { state: CallState; onClose: () => void }) {
  const feed = useRef<HTMLOListElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (feed.current) feed.current.scrollTop = feed.current.scrollHeight;
  }, [state.feed.length]);

  useEffect(() => {
    if (state.open) closeBtn.current?.focus();
  }, [state.open]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!state.open) return null;

  const saved = state.offer == null ? 0 : state.ask - state.offer;
  const savedLine = state.settled != null
    ? `Deal — ${usd(state.ask - state.settled)} off asking`
    : state.ended
      ? "No deal"
      : saved > 0 ? `▼ ${usd(saved)} off asking` : "";

  return (
    <div id="call-modal" className="show" role="dialog" aria-modal="true" aria-label="Call">
      <div className="callcard">
        <div className="top">
          <div className="pulse" />
          <div>
            <h3>{state.title}</h3>
            <div className="sub">{state.subtitle}</div>
          </div>
        </div>

        <div className="ticker">
          <div className="lbl">Live offer</div>
          <div className={`amt${state.dropped ? " drop" : ""}`}>{usd(state.offer)}</div>
          <div className="save">{savedLine}</div>
        </div>

        <ol className="feed" ref={feed}>
          {state.feed.map((line) => (
            <li className={`msg ${line.speaker}`} key={line.id}>
              <span className="who">{line.speaker}</span>
              {line.text}
            </li>
          ))}
        </ol>

        {state.settled != null && (
          <Receipt ask={state.ask} median={state.median} final={state.settled} />
        )}

        <button className="btn btn--wide" ref={closeBtn} onClick={onClose}>Close</button>
      </div>
      {state.settled != null && <Confetti />}
    </div>
  );
}
