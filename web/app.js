"use strict";

const SEATS = ["You", "West", "Partner", "East"];
const SEAT_LONG = ["You (South)", "West", "Partner (North)", "East"];
const SUIT_SYM = { S: "♠", H: "♥", C: "♣", D: "♦" };
const SAME_COLOR = { S: "C", C: "S", H: "D", D: "H" };
const RANKS = "9TJQKA";
const AI_DELAY = 650;
const AUTO_DELAY = 300;
const AUTO_LABEL = { forced: "only legal move", equivalent: "all options equal", mastered: "mastered" };

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

const S = {
  deal: null, // {hands, kitty, dealer, stick}
  actions: [],
  resp: null,
  paused: false,
  analysis: null, // {at, data}
  analyzing: false,
  feedback: null,
  timer: null,
  seq: 0,
  autoLog: {}, // action index -> why it was auto-played
  retryAt: null, // decision index being retried (not recorded again)
  progress: null,
  session: { right: 0, total: 0 },
};
const opts = {
  reveal: false, train: true, all: false,
};
const OPT_DEFAULTS = { reveal: false, train: true, all: false };

// ------------------------------------------------------------------ utils
async function api(path, body) {
  const r = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}

const actKey = (a) => [a.type, a.card || "", a.suit || "", a.alone ? 1 : 0].join(":");
const fmt = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const isRed = (code) => code[1] === "H" || code[1] === "D";
const rankLabel = (code) => (code[0] === "T" ? "10" : code[0]);
const human = (seat) => opts.all || seat === 0;
// Other seats' hands, discards and evaluations are private unless revealed.
const canSee = (seat) => seat === 0 || opts.reveal || opts.all || (S.resp && S.resp.state.done);

function effSuit(code, trump) {
  if (trump && code[0] === "J" && code[1] === SAME_COLOR[trump]) return trump;
  return code[1];
}

const SUIT_ORDER = { S: "SHCD", H: "HSDC", C: "CHSD", D: "DSHC" }; // reference suit first, colours alternate

function sortHand(cards, trump, upSuit) {
  const order = SUIT_ORDER[trump || upSuit || "S"];
  const power = (c) => {
    if (trump && c[0] === "J" && c[1] === trump) return 100;
    if (trump && c[0] === "J" && c[1] === SAME_COLOR[trump]) return 99;
    return RANKS.indexOf(c[0]);
  };
  return [...cards].sort((a, b) => {
    const sa = order.indexOf(effSuit(a, trump)), sb = order.indexOf(effSuit(b, trump));
    return sa !== sb ? sa - sb : power(b) - power(a);
  });
}

function cardEl(code, o = {}) {
  if (o.back) return el("div", "card back" + (o.small ? " small" : ""));
  const c = el("div", "card" + (isRed(code) ? " red" : "") + (o.small ? " small" : ""));
  c.append(el("span", "r", rankLabel(code)), el("span", "s", SUIT_SYM[code[1]]));
  c.title = rankLabel(code) + SUIT_SYM[code[1]];
  if (o.playable) {
    c.classList.add("playable");
    c.addEventListener("click", o.onClick);
  }
  if (o.illegal) c.classList.add("illegal");
  if (o.best) c.classList.add("best");
  if (o.win) c.classList.add("win");
  if (o.qtag !== undefined) c.append(el("span", "qtag", o.qtag));
  return c;
}

// ------------------------------------------------------------------ url state
function encodeHash() {
  const d = S.deal;
  const tok = (a) => {
    if (a.type === "pass") return "p";
    if (a.type === "order") return a.alone ? "O" : "o";
    if (a.type === "call") return (a.alone ? "C" : "c") + a.suit;
    return a.card;
  };
  return "#d=" + d.hands.flat().join("") + d.kitty.join("") + "&dl=" + d.dealer + "&st=" + (d.stick ? 1 : 0) +
    "&a=" + S.actions.map(tok).join(".");
}

function decodeHash(h) {
  const p = new URLSearchParams(h.replace(/^#/, ""));
  const cards = (p.get("d") || "").match(/../g);
  if (!cards || cards.length !== 24) return null;
  const hands = [0, 1, 2, 3].map((i) => cards.slice(i * 5, i * 5 + 5));
  const actions = (p.get("a") || "").split(".").filter(Boolean).map((t) => {
    if (t === "p") return { type: "pass" };
    if (t === "o" || t === "O") return { type: "order", alone: t === "O" };
    if (t[0] === "c" || t[0] === "C") return { type: "call", suit: t[1], alone: t[0] === "C" };
    return { type: "card", card: t };
  });
  return {
    deal: { hands, kitty: cards.slice(20), dealer: +p.get("dl") || 0, stick: p.get("st") === "1" },
    actions,
  };
}

// ------------------------------------------------------------------ flow
async function newDeal(dealer) {
  const q = dealer === undefined ? "" : `?dealer=${dealer}`;
  const d = await api("/api/random_deal" + q);
  startDeal({ ...d, stick: $("#opt-stick").checked });
}

function startDeal(deal, actions = [], paused = false) {
  S.deal = deal;
  S.actions = actions;
  S.paused = paused;
  S.feedback = null;
  S.autoLog = {};
  S.retryAt = null;
  $("#opt-stick").checked = !!deal.stick;
  refresh();
}

async function refresh() {
  clearTimeout(S.timer);
  const seq = ++S.seq;
  try {
    const resp = await api("/api/state", { deal: S.deal, actions: S.actions });
    if (seq !== S.seq) return;
    S.resp = resp;
  } catch (e) {
    toast("Error: " + e.message);
    return;
  }
  history.replaceState(null, "", encodeHash());
  render();
  scheduleNext();
}

// Why the current decision can be played without asking (null = ask the human).
function autoReason(ev) {
  if (!ev || !human(ev.seat)) return null;
  if (ev.auto === "forced") return "forced";
  return opts.train ? ev.auto : null;
}

function scheduleNext() {
  clearTimeout(S.timer);
  const st = S.resp.state, ev = S.resp.eval;
  if (st.done || S.paused) return;
  const auto = autoReason(ev);
  if (human(st.turn) && !auto) return;
  S.timer = setTimeout(() => act(ev.ai_action, true, auto), auto ? AUTO_DELAY : AI_DELAY);
}

function act(action, byAI = false, auto = null) {
  const ev = S.resp.eval;
  const index = S.actions.length;
  if (auto) S.autoLog[index] = auto;
  if (!byAI) S.paused = false;
  if (!byAI && ev) {
    const i = ev.actions.findIndex((x) => actKey(x.action) === actKey(action));
    const best = ev.actions[ev.best];
    const mine = ev.actions[i];
    const loss = best.q - mine.q;
    S.feedback = { index, seat: ev.seat, label: mine.label, bestLabel: best.label, q: mine.q, bestQ: best.q, loss,
                   situation: ev.situation, retry: S.retryAt === index };
    if (opts.train && ev.situation) {
      const tolerance = (S.progress && S.progress.settings.tolerance) || 0.03;
      if (loss > tolerance) S.paused = true; // stop and let the mistake sink in
      if (S.retryAt !== index) recordResult(ev.situation, loss, S.feedback);
    }
  }
  S.retryAt = null;
  S.actions = [...S.actions, action];
  S.analysis = null;
  refresh();
}

async function recordResult(situation, loss, fb) {
  try {
    const r = await api("/api/progress/record", { situation, loss });
    fb.record = r;
    S.session.total++;
    S.session.right += r.is_correct ? 1 : 0;
    await loadProgress();
  } catch (e) {
    toast("Couldn't save progress: " + e.message);
  }
}

async function loadProgress() {
  try {
    S.progress = await api("/api/progress");
  } catch (_) { /* ignore */ }
  if (S.resp) render();
}

function rewind(i) {
  S.actions = S.actions.slice(0, i);
  for (const k of Object.keys(S.autoLog)) if (+k >= i) delete S.autoLog[k];
  S.paused = true;
  S.feedback = null;
  S.analysis = null;
  refresh();
}

function retry(i) {
  rewind(i);
  S.retryAt = i;
}

async function deepAnalysis() {
  const samples = +$("#samples").value;
  S.analyzing = true;
  render();
  const at = S.actions.length;
  try {
    const data = await api("/api/analyze", { deal: S.deal, actions: S.actions, samples });
    if (S.actions.length === at) S.analysis = { at, data };
  } catch (e) {
    toast("Analysis failed: " + e.message);
  }
  S.analyzing = false;
  render();
}

// ------------------------------------------------------------------ render
function render() {
  renderSeats();
  renderCenter();
  renderDecision();
  renderAnalysis();
  renderProgress();
  renderLog();
  const m = S.resp.model || {};
  $("#model-badge").textContent = m.hands ? `model trained on ${(m.hands / 1e6).toFixed(1)}M hands` : "";
}

function bidText(a, isDealer) {
  if (a.type === "pass") return "Pass";
  const verb = isDealer ? "Pick it up" : "Order up";
  if (a.type === "order") return a.alone ? `${verb} — alone!` : verb;
  return `${SUIT_SYM[a.suit]} ${a.alone ? "alone!" : "is trump"}`;
}

function renderSeats() {
  const st = S.resp.state, ev = S.resp.eval;
  const upSuit = st.upcard[1];
  const inBidding = ["bid1", "bid2", "discard"].includes(st.phase);
  for (let seat = 0; seat < 4; seat++) {
    const box = document.querySelector(`.seat-${seat}`);
    box.replaceChildren();
    box.classList.toggle("turn", st.turn === seat);
    box.classList.toggle("out", st.out === seat);

    const name = el("div", "seat-name", SEAT_LONG[seat]);
    if (st.dealer === seat) name.append(el("span", "chip dealer", "Dealer"));
    if (st.maker === seat) name.append(el("span", "chip maker", `Called ${SUIT_SYM[st.trump]}${st.alone ? " · alone" : ""}`));
    if (st.out === seat) name.append(el("span", "chip", "Sitting out"));
    if (st.phase === "play" || st.done) {
      const t = st.tricks.filter((x) => x.winner === seat).length;
      if (t) name.append(el("span", "chip", `${t} trick${t > 1 ? "s" : ""}`));
    }
    const bids = st.bids.filter((b) => b.seat === seat);
    const bubble = el("div", "bubble", inBidding && bids.length ? bidText(bids[bids.length - 1].action, seat === st.dealer) : "");

    const hand = el("div", "hand");
    const faceUp = seat === 0 || opts.reveal || st.done || (opts.all && st.turn === seat);
    const myTurn = st.turn === seat && human(seat) && ev && ["play", "discard"].includes(st.phase);
    let qmap = {};
    if (myTurn) {
      ev.actions.forEach((x, i) => { qmap[x.action.card] = { q: x.q, best: i === ev.best }; });
    }
    const showQ = myTurn && !opts.train;
    const cards = sortHand(st.hands[seat], st.trump || (st.phase === "bid1" ? upSuit : null), upSuit);
    for (const c of cards) {
      if (!faceUp) { hand.append(cardEl(c, { back: true, small: seat !== 0 })); continue; }
      const o = { small: seat !== 0 };
      if (myTurn) {
        if (qmap[c]) {
          o.playable = true;
          o.onClick = () => act({ type: "card", card: c });
          if (showQ) { o.best = qmap[c].best; o.qtag = fmt(qmap[c].q); }
        } else o.illegal = true;
      }
      hand.append(cardEl(c, o));
    }
    if (seat === 2 || seat === 1 || seat === 3) box.append(name, bubble, hand);
    else box.append(hand, bubble, name);
  }
}

function renderCenter() {
  const st = S.resp.state;
  const c = $("#center");
  c.replaceChildren();
  const info = el("div", "center-info");
  if (st.phase === "bid1" || st.phase === "bid2" || (st.done && !st.trump)) {
    const area = el("div", "upcard-area");
    const stack = el("div", "kitty-stack");
    stack.append(cardEl("", { back: true }), cardEl("", { back: true }), cardEl("", { back: true }));
    const up = cardEl(st.upcard);
    if (st.phase !== "bid1") up.style.opacity = ".45";
    area.append(stack, up);
    c.append(area);
    if (st.done) info.innerHTML = "<b>Everyone passed</b><br>The hand is thrown in.";
    else if (st.phase === "bid1") info.innerHTML = `<b>Round 1</b><br>Order up the ${rankLabel(st.upcard)}${SUIT_SYM[st.upcard[1]]}?`;
    else info.innerHTML = `<b>Round 2</b><br>${SUIT_SYM[st.upcard[1]]} is turned down — name another suit${st.stick ? " (dealer is stuck)" : ""}`;
    c.append(info);
    return;
  }
  if (st.phase === "discard") {
    c.append(cardEl(st.upcard));
    info.innerHTML = `<b>${SEATS[st.dealer]} picked it up</b><br>${st.dealer === 0 || opts.all ? "Choose a card to discard" : "Dealer is discarding…"}`;
    c.append(info);
    return;
  }
  const zone = el("div", "trickzone");
  let shown = st.trick, winner = null;
  if (!shown.length && st.tricks.length) {
    const last = st.tricks[st.tricks.length - 1];
    shown = last.cards;
    winner = last.winner;
    zone.classList.add("faded");
  }
  for (const p of shown) {
    const slot = el("div", `slot s${p.seat}`);
    slot.append(cardEl(p.card, { win: p.seat === winner }));
    zone.append(slot);
  }
  c.append(zone);
  const us = st.won[0], them = st.won[1];
  let head = `<b>Trump ${SUIT_SYM[st.trump]}</b> · called by ${SEATS[st.maker]}${st.alone ? " (alone)" : ""}`;
  let sub = `Tricks — You & Partner <b>${us}</b> · West & East <b>${them}</b>`;
  if (winner !== null && !st.done) sub += `<br>${SEATS[winner]} took the last trick`;
  if (st.done) {
    const p = st.points;
    const team = p[0] > 0 ? "You & Partner" : "West & East";
    const pts = Math.max(p[0], p[1]);
    const makerTeam = st.maker % 2;
    const euchred = (p[makerTeam] === 0);
    sub = `<b>${euchred ? "Euchre! " : ""}${team} score ${pts} point${pts > 1 ? "s" : ""}</b><br>Tricks ${us}–${them}`;
  }
  info.innerHTML = head + "<br>" + sub;
  c.append(info);
}

function valueRows(ev, hide, clickable) {
  const wrap = el("div", "actions");
  const best = ev.actions[ev.best].q;
  const scale = Math.max(1, ...ev.actions.map((a) => Math.abs(a.q)));
  ev.actions.forEach((a, i) => {
    const b = el("button", "act" + (!hide && i === ev.best ? " best" : ""));
    const lbl = el("span", "lbl", a.label);
    b.append(lbl);
    if (!hide) {
      const v = el("span", "val " + (a.q >= 0 ? "pos" : "neg"), fmt(a.q));
      if (i !== ev.best) v.append(el("span", "sub", fmt(a.q - best) + " vs best"));
      else v.append(el("span", "sub", "AI's pick"));
      const bar = el("div", "bar");
      const fill = el("i");
      const w = (Math.abs(a.q) / scale) * 50;
      fill.style.left = a.q >= 0 ? "50%" : 50 - w + "%";
      fill.style.width = w + "%";
      fill.style.background = a.q >= 0 ? "var(--good)" : "var(--bad)";
      bar.append(fill, el("span", "zero"));
      b.append(v, bar);
    } else {
      b.style.gridTemplateColumns = "1fr";
    }
    if (clickable) b.addEventListener("click", () => act(a.action));
    else b.classList.add("static");
    wrap.append(b);
  });
  return wrap;
}

const masterAfter = () => (S.progress && S.progress.settings.streak_to_master) || 3;

function feedbackEl(f) {
  const cls = f.loss < 0.03 ? "good" : f.loss < 0.15 ? "ok" : "bad";
  const d = el("div", "feedback " + cls);
  const main = el("div");
  if (f.loss < 0.005) main.innerHTML = `✓ <b>${f.label}</b> — same as the AI (${fmt(f.q)} expected points).`;
  else main.innerHTML = `You chose <b>${f.label}</b> (${fmt(f.q)}). The AI prefers <b>${f.bestLabel}</b> (${fmt(f.bestQ)}) — ` +
    `a difference of <b>${f.loss.toFixed(2)}</b> points.`;
  d.append(main);
  if (f.situation) {
    const lesson = el("div", "lesson");
    lesson.innerHTML = `<span class="hint">Situation:</span> ${f.situation.context} → <b>${f.situation.answer}</b>`;
    d.append(lesson);
    const m = el("div", "mastery");
    const r = f.record;
    if (f.retry) m.textContent = "Retry — not scored.";
    else if (!r) m.textContent = opts.train ? "Saving…" : "";
    else if (r.is_correct && r.streak === masterAfter()) m.textContent = "★ Mastered — this spot will now play itself (with occasional reviews).";
    else if (r.is_correct) m.textContent = `Streak ${r.streak}/${masterAfter()} toward mastery.`;
    else m.textContent = "Streak reset — this spot will keep coming up until you get it right.";
    if (m.textContent) d.append(m);
  }
  if (opts.train && S.paused && f.index === S.actions.length - 1 && f.situation && f.loss > 0.03) {
    const row = el("div", "row");
    const again = el("button", "", "↺ Retry this decision");
    again.onclick = () => retry(f.index);
    const go = el("button", "primary", "▶ Continue");
    go.onclick = () => { S.paused = false; render(); scheduleNext(); };
    row.append(again, go);
    d.append(row);
  }
  return d;
}

function renderDecision() {
  const box = $("#decision");
  box.replaceChildren();
  const st = S.resp.state, ev = S.resp.eval;
  const head = el("div", "box-head");
  if (st.done) {
    head.append(el("h3", "", "Hand over"));
    box.append(head);
    const mine = S.resp.history.filter((h) => !h.forced && human(h.seat));
    const lost = mine.reduce((s, h) => s + h.loss, 0);
    const errs = mine.filter((h) => h.loss >= 0.03).sort((a, b) => b.loss - a.loss);
    const p = el("div", "feedback " + (lost < 0.05 ? "good" : lost < 0.3 ? "ok" : "bad"));
    p.innerHTML = mine.length
      ? `Your ${mine.length} decisions cost <b>${lost.toFixed(2)}</b> expected points versus the AI.` +
        (errs.length ? "<br>Biggest: " + errs.slice(0, 3).map((h) => `${h.label} → AI: ${h.best_label} (−${h.loss.toFixed(2)})`).join("; ") : " Flawless!")
      : "The AI played every seat.";
    if (S.feedback && S.feedback.index === S.actions.length - 1) box.append(feedbackEl(S.feedback));
    box.append(p);
    const row = el("div", "row");
    const again = el("button", "", "Replay this deal");
    again.onclick = () => startDeal(S.deal);
    const nd = el("button", "primary", "Next deal");
    nd.onclick = () => newDeal((S.deal.dealer + 1) % 4);
    row.append(again, nd);
    box.append(row);
    return;
  }
  const seat = ev.seat;
  const verb = { bid1: "bid", bid2: "bid", discard: "discard", play: "play" }[st.phase];
  const isHuman = human(seat);
  head.append(el("h3", "", seat === 0 ? `Your turn to ${verb}` : `${SEAT_LONG[seat]} to ${verb}`));
  head.append(el("span", "hint", "values = expected points for that seat's team"));
  box.append(head);

  const auto = autoReason(ev);
  if (!canSee(seat) || auto) {
    box.append(el("div", "hint", auto
      ? `Plays automatically (${AUTO_LABEL[auto]}).`
      : `${SEATS[seat]} is thinking… Turn on “Show all hands” to see the AI's evaluation for other seats.`));
    if (S.feedback) box.append(feedbackEl(S.feedback));
    if (!(S.feedback && S.paused && S.feedback.index === S.actions.length - 1)) box.append(pauseRow(ev));
    return;
  }
  const hide = isHuman && opts.train;
  if (isHuman && opts.train && ev.situation) {
    const sit = el("div", "situation");
    const streak = ev.situation.seen ? ` · streak ${ev.situation.streak}/${masterAfter()}` : " · new situation";
    sit.innerHTML = `${ev.situation.context}<span class="hint">${ev.situation.mastered ? " · review" : streak}</span>`;
    box.append(sit);
  }
  const needsButtons = st.phase === "bid1" || st.phase === "bid2";
  if (needsButtons || !hide) box.append(valueRows(ev, hide, isHuman));
  if (isHuman && !needsButtons) box.append(el("div", "hint", `Click a card in your hand${hide ? "" : " (or a row above)"} to ${verb} it.`));

  if (S.feedback) box.append(feedbackEl(S.feedback));

  const row = el("div", "row");
  const sel = el("select");
  sel.id = "samples";
  for (const n of [128, 256, 512]) {
    const o = el("option", "", `${n} worlds`);
    o.value = n;
    if (n === (S.samples || 256)) o.selected = true;
    sel.append(o);
  }
  sel.onchange = () => { S.samples = +sel.value; };
  const deep = el("button", "", S.analyzing ? "Analyzing…" : "Deep analysis");
  deep.disabled = S.analyzing;
  deep.onclick = deepAnalysis;
  row.append(deep, sel);
  if (isHuman) {
    const aiMove = el("button", "", "Let AI decide");
    aiMove.onclick = () => act(ev.ai_action, true);
    row.append(aiMove);
  } else row.append(...pauseRow(ev).children);
  box.append(row);
}

function pauseRow(ev) {
  const row = el("div", "row");
  if (S.paused) {
    const go = el("button", "primary", "▶ Continue");
    go.onclick = () => { S.paused = false; render(); scheduleNext(); };
    const one = el("button", "", "Step");
    one.onclick = () => act(ev.ai_action, true);
    row.append(go, one);
  } else {
    const pause = el("button", "", "Pause");
    pause.onclick = () => { S.paused = true; clearTimeout(S.timer); render(); };
    row.append(pause);
  }
  return row;
}

function renderAnalysis() {
  const box = $("#analysis");
  const a = S.analysis;
  if (!a || a.at !== S.actions.length) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.replaceChildren();
  const d = a.data;
  const head = el("div", "box-head");
  head.append(el("h3", "", "Deep analysis"));
  head.append(el("span", "hint", `${d.samples} sampled deals · ${d.ess.toFixed(0)} effective · ${d.secs}s`));
  box.append(head);
  box.append(el("div", "hint",
    `Hidden cards are re-dealt ${d.samples} ways consistent with what ${d.seat === 0 ? "you" : SEATS[d.seat]} can see, ` +
    "weighted by how well each deal explains the bidding and play, then every option is played out."));
  const ev = { actions: d.actions.map((x) => ({ ...x, q: x.value })), best: d.best };
  const rows = valueRows(ev, false, false);
  [...rows.children].forEach((r, i) => {
    const x = d.actions[i];
    const sub = r.querySelector(".sub");
    if (sub) sub.textContent = i === d.best ? `±${x.stderr.toFixed(2)} · best` : `${fmt(x.vs_best)} ±${x.vs_best_stderr.toFixed(2)}`;
  });
  box.append(rows);

  const bel = el("div", "beliefs");
  bel.append(el("div", "hint", "Where the unseen cards probably are:"));
  for (const seat of Object.keys(d.beliefs)) {
    const all = Object.entries(d.beliefs[seat]).sort((x, y) => y[1] - x[1]);
    const probs = all.filter(([, p]) => p >= 0.05).slice(0, 8);
    const row = el("div");
    row.append(el("div", "who", SEAT_LONG[+seat]));
    const cards = el("div", "cards");
    for (const [c, p] of probs) {
      const chip = el("span", "pchip" + (isRed(c) ? " red" : ""), rankLabel(c) + SUIT_SYM[c[1]]);
      chip.append(el("span", "p", Math.round(p * 100) + "%"));
      chip.style.background = `rgba(200,145,45,${(p * 0.45).toFixed(2)})`;
      cards.append(chip);
    }
    if (!probs.length) cards.append(el("span", "hint", "—"));
    const rest = all.length - probs.length;
    if (rest > 0) cards.append(el("span", "hint", `+${rest} more ≤ ${Math.round(all[probs.length][1] * 100)}%`));
    row.append(cards);
    bel.append(row);
  }
  box.append(bel);
}

function renderProgress() {
  const box = $("#progress");
  box.replaceChildren();
  const p = S.progress;
  const head = el("div", "box-head");
  head.append(el("h3", "", "Training progress"));
  box.append(head);
  if (!opts.train) {
    box.append(el("div", "hint", "Turn on Training mode to track situations you've mastered and auto-play them."));
    return;
  }
  if (!p) return;
  const sits = p.situations;
  const mastered = sits.filter((x) => x.mastered);
  head.append(el("span", "hint", `${mastered.length} mastered · ${sits.length} seen`));
  if (S.session.total) {
    box.append(el("div", "hint", `This session: ${S.session.right}/${S.session.total} decisions matched the AI.`));
  }
  if (!sits.length) {
    box.append(el("div", "hint", "Every real decision you make is filed under a situation. Get one right " +
      `${p.settings.streak_to_master} times in a row and it plays itself from then on.`));
  }
  const weak = sits.filter((x) => !x.mastered && x.seen > x.correct)
    .sort((a, b) => b.lost - a.lost || a.correct / a.seen - b.correct / b.seen).slice(0, 6);
  if (weak.length) {
    box.append(el("div", "sub-head", "Needs work"));
    const ul = el("ul", "sits");
    for (const x of weak) {
      const li = el("li");
      li.innerHTML = `${x.context} → <b>${x.answer}</b>`;
      li.append(el("span", "hint", ` ${x.correct}/${x.seen} right · −${x.lost.toFixed(2)} pts`));
      ul.append(li);
    }
    box.append(ul);
  }
  if (mastered.length) {
    const det = el("details");
    det.append(el("summary", "", `Mastered (${mastered.length})`));
    const ul = el("ul", "sits");
    for (const x of mastered.sort((a, b) => b.seen - a.seen)) {
      const li = el("li");
      li.innerHTML = `${x.context} → <b>${x.answer}</b>`;
      ul.append(li);
    }
    det.append(ul);
    box.append(det);
  }
  const row = el("div", "row");
  const lab = el("label", "hint", "Master after ");
  const sel = el("select");
  for (const n of [2, 3, 5, 8]) {
    const o = el("option", "", `${n} in a row`);
    o.value = n;
    if (n === p.settings.streak_to_master) o.selected = true;
    sel.append(o);
  }
  sel.onchange = async () => { S.progress = await api("/api/progress/settings", { streak_to_master: +sel.value }); render(); };
  lab.append(sel);
  const reset = el("button", "", "Reset progress");
  reset.onclick = async () => {
    if (!confirm("Forget all training progress?")) return;
    S.progress = await api("/api/progress/reset", {});
    S.session = { right: 0, total: 0 };
    render();
  };
  row.append(lab, reset);
  box.append(row);
}

function renderLog() {
  const ol = $("#log");
  ol.replaceChildren();
  const st = S.resp.state;
  const perTrick = st.out === null ? 4 : 3;
  let plays = 0, lastPhase = null;
  for (const h of S.resp.history) {
    if (h.phase !== lastPhase && (h.phase === "bid1" || h.phase === "bid2")) {
      ol.append(el("li", "sep", h.phase === "bid1" ? "Bidding — round 1" : "Bidding — round 2"));
    }
    if (h.phase === "play" && plays % perTrick === 0) ol.append(el("li", "sep", `Trick ${plays / perTrick + 1}`));
    if (h.phase === "play") plays++;
    lastPhase = h.phase;
    const li = el("li", h.forced ? "forced" : "");
    const visible = canSee(h.seat);
    li.append(el("span", "who", SEATS[h.seat]));
    li.append(el("span", "", h.phase === "discard" && !visible ? "Discards a card" : h.label));
    const d = el("span", "d");
    if (S.autoLog[h.index] && S.autoLog[h.index] !== "forced") d.append(el("span", "tag", "auto · " + AUTO_LABEL[S.autoLog[h.index]]));
    if (h.forced) d.append("forced");
    else if (!visible) d.textContent = "";
    else {
      const cls = h.loss < 0.03 ? "good" : h.loss < 0.15 ? "ok" : "bad";
      d.insertAdjacentHTML("beforeend", `<span class="dot ${cls}"></span>${fmt(h.q)}`);
      li.title = h.loss >= 0.005 ? `AI preferred ${h.best_label} (${fmt(h.best_q)})` : "Matches the AI";
    }
    li.append(d);
    li.addEventListener("click", () => rewind(h.index));
    ol.append(li);
  }
  ol.scrollTop = ol.scrollHeight;
}

// ------------------------------------------------------------------ editor
const ED = { target: 0, assign: {} };
const TARGETS = [["S0", "You", 5], ["S1", "West", 5], ["S2", "Partner", 5], ["S3", "East", 5], ["UP", "Upcard", 1], ["K", "Kitty", 3]];

function openEditor(fromCurrent) {
  ED.assign = {};
  if (fromCurrent && S.deal) {
    S.deal.hands.forEach((h, i) => h.forEach((c) => (ED.assign[c] = "S" + i)));
    ED.assign[S.deal.kitty[0]] = "UP";
    S.deal.kitty.slice(1).forEach((c) => (ED.assign[c] = "K"));
    $("#ed-dealer").value = S.deal.dealer;
  }
  renderEditor();
  $("#editor").showModal();
}

function renderEditor() {
  const t = $("#ed-targets");
  t.replaceChildren();
  TARGETS.forEach(([key, name, n], i) => {
    const count = Object.values(ED.assign).filter((v) => v === key).length;
    const b = el("button", i === ED.target ? "sel" : "", `${name} ${count}/${n}`);
    b.type = "button";
    b.onclick = () => { ED.target = i; renderEditor(); };
    t.append(b);
  });
  const g = $("#ed-grid");
  g.replaceChildren();
  for (const s of ["S", "H", "C", "D"]) {
    for (const r of "AKQJT9") {
      const code = r + s;
      const c = cardEl(code);
      const who = ED.assign[code];
      if (who) {
        c.classList.add("assigned");
        c.append(el("span", "tag", TARGETS.find((x) => x[0] === who)[1]));
      }
      c.onclick = () => {
        const [key, , n] = TARGETS[ED.target];
        if (ED.assign[code] === key) delete ED.assign[code];
        else {
          const count = Object.values(ED.assign).filter((v) => v === key).length;
          if (count >= n) { toast(`${TARGETS[ED.target][1]} already has ${n} card${n > 1 ? "s" : ""}`); return; }
          ED.assign[code] = key;
        }
        renderEditor();
      };
      g.append(c);
    }
  }
}

function dealFromEditor() {
  const all = [];
  for (const s of "SHCD") for (const r of RANKS) all.push(r + s);
  const free = all.filter((c) => !ED.assign[c]);
  for (let i = free.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [free[i], free[j]] = [free[j], free[i]];
  }
  const take = (key, n) => {
    const got = Object.keys(ED.assign).filter((c) => ED.assign[c] === key);
    while (got.length < n) got.push(free.pop());
    return got;
  };
  const hands = [0, 1, 2, 3].map((i) => take("S" + i, 5));
  const kitty = [...take("UP", 1), ...take("K", 3)];
  return { hands, kitty, dealer: +$("#ed-dealer").value, stick: $("#opt-stick").checked };
}

// ------------------------------------------------------------------ wire up
function bind() {
  $("#btn-new").onclick = () => newDeal(S.deal ? (S.deal.dealer + 1) % 4 : undefined);
  $("#btn-edit").onclick = () => openEditor(false);
  $("#btn-restart").onclick = () => startDeal(S.deal);
  $("#btn-share").onclick = async () => {
    const url = location.origin + location.pathname + encodeHash();
    try { await navigator.clipboard.writeText(url); toast("Link copied"); } catch (_) { toast(url); }
  };
  $("#ed-clear").onclick = () => { ED.assign = {}; renderEditor(); };
  $("#ed-current").onclick = () => openEditor(true);
  $("#ed-go").onclick = () => { $("#editor").close(); startDeal(dealFromEditor()); };
  $("#opt-stick").onchange = (e) => { if (S.deal) { S.deal.stick = e.target.checked; S.actions = []; refresh(); } };
  const query = new URLSearchParams(location.search); // e.g. ?train=0&reveal=1 overrides saved settings
  for (const k of ["reveal", "train", "all"]) {
    const box = $("#opt-" + k);
    box.checked = OPT_DEFAULTS[k];
    try {
      const v = localStorage.getItem("euchre-" + k);
      if (v !== null) box.checked = v === "1";
    } catch (_) { /* ignore */ }
    if (query.has(k)) box.checked = query.get(k) === "1";
    opts[k] = box.checked;
    box.onchange = () => {
      opts[k] = box.checked;
      try { localStorage.setItem("euchre-" + k, box.checked ? "1" : "0"); } catch (_) { /* ignore */ }
      if (S.resp) { render(); scheduleNext(); }
    };
  }
}

async function init() {
  bind();
  loadProgress();
  const fromHash = location.hash ? decodeHash(location.hash) : null;
  try {
    const info = await api("/api/info");
    if (info.model && info.model.hands) $("#model-badge").textContent = `model trained on ${(info.model.hands / 1e6).toFixed(1)}M hands`;
  } catch (e) {
    $("#status").textContent = "No trained model found yet — start training first (see README).";
    return;
  }
  if (fromHash) {
    startDeal(fromHash.deal, fromHash.actions, fromHash.actions.length > 0);
  } else newDeal();
}

init();
