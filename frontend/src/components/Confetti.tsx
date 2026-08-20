import { useEffect, useRef } from "react";

const COLORS = ["#f2c230", "#c22f1d", "#16140f", "#4ade80"];
const FRAMES = 150;

/** Fires once on mount. Respects prefers-reduced-motion. */
export function Confetti() {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const cv = canvas.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;

    cv.width = innerWidth;
    cv.height = innerHeight;
    const bits = Array.from({ length: 130 }, () => ({
      x: innerWidth / 2 + (Math.random() - 0.5) * 260,
      y: innerHeight / 2,
      vx: (Math.random() - 0.5) * 9,
      vy: -Math.random() * 13 - 4,
      s: 4 + Math.random() * 6,
      c: COLORS[(Math.random() * COLORS.length) | 0],
      r: Math.random() * Math.PI,
    }));

    let frame = 0;
    let raf = 0;
    const tick = () => {
      ctx.clearRect(0, 0, cv.width, cv.height);
      for (const b of bits) {
        b.vy += 0.36; b.x += b.vx; b.y += b.vy; b.r += 0.1;
        ctx.save();
        ctx.translate(b.x, b.y);
        ctx.rotate(b.r);
        ctx.fillStyle = b.c;
        ctx.fillRect(-b.s / 2, -b.s / 2, b.s, b.s * 0.6);
        ctx.restore();
      }
      if (++frame < FRAMES) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas id="confetti" className="show" ref={canvas} />;
}
