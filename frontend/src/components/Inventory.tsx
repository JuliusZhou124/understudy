import { useMemo, useState } from "react";

import { usd, type ListingRow } from "../api";

type Sort = "rank" | "over" | "price-asc" | "price-desc";

const SORTS: Record<Sort, (a: ListingRow, b: ListingRow) => number> = {
  rank: (a, b) => b.rank - a.rank,
  over: (a, b) => b.over_median - a.over_median,
  "price-asc": (a, b) => a.ask_price - b.ask_price,
  "price-desc": (a, b) => b.ask_price - a.ask_price,
};

function ListingCard({
  row, selected, onSelect,
}: { row: ListingRow; selected: boolean; onSelect: (id: string) => void }) {
  const why = row.why.slice(0, 3).join(" · ") || "no negotiability signals";
  return (
    <button className="card" aria-pressed={selected} onClick={() => onSelect(row.id)}>
      {row.most_negotiable && <span className="card__badge">Most negotiable</span>}
      <span className="card__plate">
        <span className="card__ask">{usd(row.ask_price)}</span>
        <span className="card__median">median {usd(row.sold_median)}</span>
      </span>
      <span className="card__body">
        <span className="card__title">{row.title}</span>
        <span className={`card__why${row.over_median > 0 ? "" : " cool"}`}>{why}</span>
      </span>
    </button>
  );
}

export function Inventory({
  rows, selected, onSelect,
}: { rows: ListingRow[]; selected: string | null; onSelect: (id: string) => void }) {
  const [text, setText] = useState("");
  const [max, setMax] = useState("");
  const [sort, setSort] = useState<Sort>("rank");

  const visible = useMemo(() => {
    const q = text.trim().toLowerCase();
    const cap = Number(max) || Infinity;
    return rows
      .filter((r) => r.ask_price <= cap)
      .filter((r) => !q ||
        `${r.title} ${r.condition} ${r.seller_type} ${r.why.join(" ")}`.toLowerCase().includes(q))
      .sort(SORTS[sort]);
  }, [rows, text, max, sort]);

  return (
    <section className="block">
      <div className="block__head">
        <h2>Inventory</h2>
        <div className="fbar">
          <input id="ftext" placeholder="Filter: model, seller, condition…" value={text}
                 autoComplete="off" onChange={(e) => setText(e.target.value)} />
          <input id="fmax" type="number" min="0" step="25" placeholder="Max $" value={max}
                 onChange={(e) => setMax(e.target.value)} />
          <select value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
            <option value="rank">Sort: negotiability</option>
            <option value="over">Most over median</option>
            <option value="price-asc">Price: low → high</option>
            <option value="price-desc">Price: high → low</option>
          </select>
          <span className="fcount">{visible.length} of {rows.length}</span>
        </div>
      </div>

      <div className="grid">
        {visible.length === 0 ? (
          <p className="empty">
            Nothing matches. Clear the filter, or run <code>understudy ingest</code> then{" "}
            <code>build</code>.
          </p>
        ) : (
          visible.map((row) => (
            <ListingCard key={row.id} row={row} selected={row.id === selected} onSelect={onSelect} />
          ))
        )}
      </div>
    </section>
  );
}
