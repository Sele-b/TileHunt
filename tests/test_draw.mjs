// Runs the generated page's own drawing code in Node against a recording
// canvas, so a marker is proved to be DRAWN rather than merely mentioned in
// the source. The pinned-points bug passed every string check and still drew
// nothing, because ROUTE dropped the field on its way out of the data.
import fs from "node:fs";

const file = process.argv[2];
const html = fs.readFileSync(file, "utf8");
const script = html.slice(html.indexOf("<script>") + 8,
                          html.lastIndexOf("</script>"));

const ops = [];
function rec(name) { return (...a) => ops.push([name, ...a]); }
const ctx = new Proxy({}, {
  get(t, k) {
    if (k === "measureText") return () => ({ width: 10 });
    if (k === "canvas") return { width: 900, height: 700 };
    if (k === "createLinearGradient")
      return () => ({ addColorStop() {} });
    // the grid is painted into an ImageData, so this one must be real
    if (k === "createImageData" || k === "getImageData")
      return (w, h) => ({ width: w | 0 || 1, height: h | 0 || 1,
                          data: new Uint8ClampedArray(
                            Math.max(1, (w | 0) * (h | 0)) * 4) });
    if (typeof k === "string" && /^(fillStyle|strokeStyle|lineWidth|font|textAlign|textBaseline|globalAlpha|lineJoin|lineCap|globalCompositeOperation|imageSmoothingEnabled|filter)$/.test(k))
      return t[k];
    return rec(k);
  },
  set(t, k, v) { t[k] = v; ops.push(["set:" + k, v]); return true; },
});

function el(id) {
  const node = {
    id, style: {}, dataset: {}, value: "", textContent: "", innerHTML: "",
    width: 900, height: 700, clientWidth: 900, clientHeight: 700,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    listeners: {},
    addEventListener(type, fn) {
      (this.listeners[type] = this.listeners[type] || []).push(fn);
    },
    removeEventListener() {},
    fire(type, ev) {
      for (const fn of this.listeners[type] || []) fn(ev);
    },
    getContext: () => ctx,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 700 }),
    querySelectorAll: () => [], appendChild() {}, focus() {}, blur() {},
    setPointerCapture() {}, releasePointerCapture() {},
  };
  return node;
}
const nodes = new Map();
globalThis.document = {
  getElementById: id => { if (!nodes.has(id)) nodes.set(id, el(id));
                          return nodes.get(id); },
  createElement: () => el("tmp"),
  addEventListener() {}, body: el("body"), documentElement: el("html"),
};
globalThis.window = { addEventListener() {}, innerWidth: 900, innerHeight: 700,
                      devicePixelRatio: 1, location: { href: "" } };
globalThis.devicePixelRatio = 1;
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
globalThis.setTimeout = () => 0;
globalThis.performance = { now: () => 0 };
Object.defineProperty(globalThis, "navigator", {
  value: { clipboard: { writeText: () => Promise.resolve() } },
  configurable: true, writable: true });
globalThis.atob = b => Buffer.from(b, "base64").toString("binary");
globalThis.Image = class { set src(_) {} addEventListener() {} };
globalThis.ResizeObserver = class { observe() {} };

const run = new Function(script
  + "\nresize();"            // the page sizes itself on load; do it here too
  + "\nreturn {ROUTE, draw, view, D, sx, sy, ll2t, t2ll,"
  + "        size: () => [VW, VH],"
  + "        unpackVisited: () => VISITED, unpackRoad: () => ROAD,"
  + "        unpackLand: () => LAND,"
  + "        paintGrid,"
  + "        plan: () => ({kind: planKind, mode: planMode,"
  + "                      want: [...planWant.values()], via: planVia,"
  + "                      cmd: document.getElementById(\'plan-text\').textContent})};");
let api;
try { api = run(); }
catch (err) { console.log("SCRIPT FAILED: " + err.message); process.exit(2); }

const { ROUTE, draw, view, sx, sy, ll2t } = api;
console.log("ROUTE present      :", !!ROUTE);
const routeless = !ROUTE;
// Only fields the route actually has are checked: a ride with no waypoints
// and no skipped tiles has nothing extra to draw, and that is not a failure.
const wantPinned = !routeless && !!(api.D.route && api.D.route.pinned);
const wantSkipped = !routeless && !!(api.D.route && api.D.route.skipped);
console.log("ROUTE.pinned       :",
            wantPinned ? (ROUTE.pinned ? "carried" : "LOST") : "none in data");
console.log("ROUTE.skipped      :",
            wantSkipped ? (ROUTE.skipped ? "carried" : "LOST") : "none in data");
if ((wantPinned && !ROUTE.pinned) || (wantSkipped && !ROUTE.skipped))
  process.exit(1);

const via = (ROUTE && ROUTE.pinned && ROUTE.pinned.via) || [];
if (wantPinned) console.log("waypoints in ROUTE :", via.length);

// frame the whole route so every marker lands on screen, then draw for real
let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
if (routeless) {
  x0 = api.D.x0; y0 = api.D.y0;
  x1 = api.D.x0 + api.D.w; y1 = api.D.y0 + api.D.h;
} else for (const p of ROUTE.pts) {
  x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]);
  y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
}
view.cx = (x0 + x1) / 2; view.cy = (y0 + y1) / 2;
const [VW, VH] = api.size();
console.log("viewport           :", VW + "x" + VH);
view.scale = Math.min(0.8 * VW / (x1 - x0), 0.8 * VH / (y1 - y0));
ops.length = 0;
// paintGrid ran once at load, before the recorder was cleared; run it again
try { api.paintGrid(); } catch (e) { /* no grid in this page */ }
try { draw(); }
catch (err) { console.log("DRAW FAILED: " + err.message); process.exit(2); }
console.log("canvas operations  :", ops.length);

// a diamond is four lineTo from a moveTo, closed, with no arc between
const diamonds = [];
for (let i = 0; i < ops.length; i++) {
  if (ops[i][0] !== "moveTo") continue;
  const seq = ops.slice(i, i + 6).map(o => o[0]).join(",");
  if (seq.startsWith("moveTo,lineTo,lineTo,lineTo,closePath"))
    diamonds.push([ops[i][1], ops[i][2]]);
}
console.log("diamonds drawn     :", diamonds.length);

const fails = [];

// The tiles you hold are their own layer: blue, on top of everything, and
// gone the moment the layer is switched off.
if (api.D.visited) {
  const W = api.D.w, H = api.D.h;
  const VIS = api.unpackVisited(), ROAD = api.unpackRoad(), LAND = api.unpackLand();
  const HELD = [74, 163, 255];
  const paint = () => {
    ops.length = 0;
    api.paintGrid();
    const put = ops.filter(o => o[0] === "putImageData").pop();
    return put ? put[1].data : null;
  };
  const at = (px, i) => [px[i << 2], px[(i << 2) + 1], px[(i << 2) + 2]];
  const same = (a, b) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
  const find = f => { for (let i = 0; i < W * H; i++) if (f(i)) return i; return -1; };

  const held = find(i => VIS[i] && ROAD[i]);
  const heldRed = find(i => VIS[i] && !ROAD[i] && LAND && LAND[i]);
  const plainRed = find(i => !VIS[i] && !ROAD[i] && LAND && LAND[i]);
  const plainGreen = find(i => !VIS[i] && ROAD[i]);

  let px = paint();
  console.log(`held tile drawn    : rgb(${at(px, held)})`);
  if (!same(at(px, held), HELD)) {
    console.log("FAIL a held tile is not painted in the held colour");
    fails.push("held-colour");
  }
  if (heldRed >= 0) {
    console.log(`held gap tile      : rgb(${at(px, heldRed)}) `
                + `(a tile with no road that you have ridden)`);
    if (!same(at(px, heldRed), HELD)) {
      console.log("FAIL a held tile with no road is not blue");
      fails.push("held-over-red");
    }
  } else {
    console.log("held gap tile      : none in this page");
  }

  // switch the layer off: the same tiles must go back to green and red
  const btn = nodes.get("b-visited");
  btn.onclick.call(btn);
  px = paint();
  if (!same(at(px, held), at(px, plainGreen))) {
    console.log(`FAIL with the layer off a held road tile is rgb(${at(px, held)}), `
                + `not the green rgb(${at(px, plainGreen)})`);
    fails.push("held-off-green");
  }
  if (heldRed >= 0 && plainRed >= 0
      && !same(at(px, heldRed), at(px, plainRed))) {
    console.log(`FAIL with the layer off a held gap tile is rgb(${at(px, heldRed)}), `
                + `not the red rgb(${at(px, plainRed)})`);
    fails.push("held-off-red");
  }
  console.log(`layer off          : those tiles revert to `
              + `rgb(${at(px, held)}) and rgb(${at(px, heldRed >= 0 ? heldRed : held)})`);
  btn.onclick.call(btn);

  // green and red off, visited on: only the tiles you hold remain
  for (const id of ["b-green", "b-red"]) {
    const b = nodes.get(id);
    if (b && b.onclick) b.onclick.call(b);
  }
  px = paint();
  let lit = 0, wrong = 0;
  for (let i = 0; i < W * H; i++) {
    if (!px[(i << 2) + 3]) continue;
    lit++;
    if (!same(at(px, i), HELD)) wrong++;
  }
  console.log(`only-my-tiles view : ${lit} painted, ${wrong} not in the held colour`);
  if (!lit || wrong) {
    console.log("FAIL turning green and red off does not leave just your tiles");
    fails.push("only-held");
  }
  for (const id of ["b-green", "b-red"]) {
    const b = nodes.get(id);
    if (b && b.onclick) b.onclick.call(b);
  }
}

if (wantSkipped) {
  const want = ROUTE.skipped.map(t => [sx(t[0]), sy(t[1])]);
  const rects = ops.filter(o => o[0] === "rect").map(o => [o[1], o[2]]);
  let hit = 0;
  for (const w of want)
    if (rects.some(r => Math.hypot(r[0] - w[0], r[1] - w[1]) < 1.5)) hit++;
  console.log(`skipped tiles drawn: ${hit} of ${want.length}`);
  if (hit !== want.length) fails.push("skipped");
}
if (wantPinned && diamonds.length !== via.length) {
  console.log(`FAIL expected ${via.length} diamond(s)`);
  fails.push("count");
}
// each diamond must sit on its waypoint's snapped junction
for (const [k, v] of via.entries()) {
  // use the page's own projection, not a copy of it: a copy can agree with a
  // formula while disagreeing with what the page actually put on screen
  const b = ll2t(v.node[0], v.node[1]);
  const want = [sx(b[0]), sy(b[1]) - 6];      // moveTo starts at the top point
  const hit = diamonds.some(d => Math.hypot(d[0] - want[0], d[1] - want[1]) < 1.5);
  console.log(`  waypoint ${k + 1} diamond on its junction: ${hit}`
              + (hit ? "" : `  (wanted ${want.map(n => n.toFixed(1))}, got `
                 + diamonds.map(d => d.map(n => n.toFixed(1)).join(",")).join(" / ") + ")"));
  if (!hit) fails.push("place" + k);
}
// and the hollow ring on the point actually asked for
if (wantPinned) {
  const dashed = ops.filter(o => o[0] === "setLineDash" && o[1] && o[1].length);
  console.log("dashed links drawn :", dashed.length,
              "(expect one per pinned point)");
  if (dashed.length < via.length) fails.push("links");
}

// A layer button must actually change what is painted, not just light up.
const layerTests = [
  ["b-green", "road tiles", true],
  // red tiles are land without a road, which only exists with a mask
  ["b-red", "red tiles", !!api.D.land],
];
for (const [id, label, present] of layerTests) {
  const btn = nodes.get(id);
  if (!btn || !btn.onclick) continue;
  if (!present) {
    console.log(`${id.padEnd(19)}: nothing of that layer in this page`);
    continue;
  }
  const paint = () => {
    ops.length = 0;
    try { api.paintGrid(); } catch (e) { return null; }
    const put = ops.filter(o => o[0] === "putImageData").pop();
    if (!put) return null;
    let lit = 0;
    const px = put[1].data;
    for (let o = 3; o < px.length; o += 4) if (px[o]) lit++;
    return lit;
  };
  const before = paint();
  btn.onclick.call(btn);
  const after = paint();
  if (before === null || after === null) continue;
  console.log(`${id.padEnd(19)}: ${before} painted -> ${after} after a click`);
  if (after >= before) {
    console.log(`FAIL clicking ${id} did not remove the ${label}`);
    fails.push(id);
  }
  btn.onclick.call(btn);           // put it back for the checks that follow
}


console.log(fails.length ? "\nFAILURES: " + fails.join(", ")
                         : "\ndrawing OK");
process.exit(fails.length ? 1 : 0);
