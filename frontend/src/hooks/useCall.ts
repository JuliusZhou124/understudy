import { useCallback, useEffect, useRef, useState } from "react";

import type { CallEvent } from "../api";

export interface FeedLine {
  id: number;
  speaker: "buyer" | "seller" | "system";
  text: string;
}

export interface CallState {
  open: boolean;
  title: string;
  subtitle: string;
  ask: number;
  median: number;
  offer: number | null;
  dropped: boolean;
  feed: FeedLine[];
  settled: number | null;
  ended: string | null;
}

const CLOSED: CallState = {
  open: false, title: "", subtitle: "", ask: 0, median: 0,
  offer: null, dropped: false, feed: [], settled: null, ended: null,
};

/**
 * The call card's state machine.
 *
 * Deliberately source-agnostic: it consumes `CallEvent`s and does not care
 * whether they arrived from a live Vapi call over the WebSocket or from a
 * replayed simulation. One renderer, two sources.
 */
export function useCall() {
  const [state, setState] = useState<CallState>(CLOSED);
  const lineId = useRef(0);
  const replay = useRef<number | undefined>(undefined);
  const dropTimer = useRef<number | undefined>(undefined);

  const stopReplay = useCallback(() => window.clearTimeout(replay.current), []);

  const open = useCallback((title: string, subtitle: string, ask: number, median: number) => {
    stopReplay();
    setState({ ...CLOSED, open: true, title, subtitle, ask, median, offer: ask });
  }, [stopReplay]);

  const close = useCallback(() => {
    stopReplay();
    setState(CLOSED);
  }, [stopReplay]);

  const say = useCallback((speaker: FeedLine["speaker"], text: string) => {
    setState((s) => ({ ...s, feed: [...s.feed, { id: lineId.current++, speaker, text }] }));
  }, []);

  const event = useCallback((ev: CallEvent) => {
    switch (ev.type) {
      case "transcript":
        if (ev.final !== false) say(ev.speaker, ev.text);
        break;
      case "offer":
        setState((s) => ({
          ...s,
          offer: ev.price,
          dropped: s.offer == null || ev.price < s.offer,
        }));
        window.clearTimeout(dropTimer.current);
        dropTimer.current = window.setTimeout(
          () => setState((s) => ({ ...s, dropped: false })), 450);
        break;
      case "deal":
        setState((s) => ({ ...s, offer: ev.price, settled: ev.price }));
        say("system", `Deal at $${Math.round(ev.price).toLocaleString()}`);
        break;
      case "call-ended":
        setState((s) => ({ ...s, ended: ev.outcome }));
        say("system", `Call ended — ${ev.outcome}${ev.reason ? `: ${ev.reason}` : ""}`);
        break;
      case "status":
        say("system", `Call ${ev.status}`);
        break;
    }
  }, [say]);

  /** Play a recorded event list back at conversational pace. */
  const play = useCallback((events: CallEvent[]) => {
    const queue = [...events];
    const step = () => {
      const ev = queue.shift();
      if (!ev) return;
      event(ev);
      replay.current = window.setTimeout(step, ev.type === "transcript" ? 900 : 220);
    };
    replay.current = window.setTimeout(step, 350);
  }, [event]);

  useEffect(() => stopReplay, [stopReplay]);

  return { state, open, close, say, event, play };
}

/** Live events from the server. Same shape the replay produces. */
export function useLiveEvents(onEvent: (ev: CallEvent) => void) {
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/dashboard`);
    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data) as CallEvent;
      if (ev.type !== "hello") handler.current(ev);
    };
    return () => ws.close();
  }, []);
}
