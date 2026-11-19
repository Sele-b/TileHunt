
"use strict";
const D = __DATA__;

/* ---------- unpack bitsets ---------------------------------------------- */
function unpack(b64, n){
  if(!b64) return null;
  const raw = atob(b64), out = new Uint8Array(n);
  for(let i=0;i<n;i++){
    out[i] = (raw.charCodeAt(i>>3) >> (7-(i&7))) & 1;
  }
  return out;
}
const W = D.w, H = D.h, NT = W*H;
const ROAD = unpack(D.road, NT);
const LAND = unpack(D.land, NT);
const VISITED = unpack(D.visited, NT);
let showVisited = true;
const Z = D.z, NZ = Math.pow(2, Z);

/* road length per tile, in hundredths of a km, little-endian uint16 */
function unpack16(b64, n){
  if(!b64) return null;
  const raw = atob(b64), out = new Uint16Array(n);
  for(let i=0;i<n;i++) out[i] = raw.charCodeAt(2*i) | (raw.charCodeAt(2*i+1) << 8);
  return out;
}
const KMQ = unpack16(D.km, NT);
const KM_SCALE = D.kmScale || 0.01;
const kmOf = i => KMQ ? KMQ[i] * KM_SCALE : 0;

/* Sequential encoding: one hue, light to dark. The lightest step means
   "near zero", so the ramp is reversed on a dark surface, where it is the
   darkest step that recedes into the background. */
const BLUE = ["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"];
const ZERO_COLOUR = "#d03b3b";          /* a state, not a magnitude */
/* the tiles you hold: a hue of its own, away from the green of the roads, the
   red of the gaps and the muted blues of the density ramp */
const HELD = [74,163,255];              /* #4aa3ff */

const inWin = (x,y)=> x>=D.x0 && y>=D.y0 && x<D.x0+W && y<D.y0+H;
const idx   = (x,y)=> (y-D.y0)*W + (x-D.x0);
function status(x,y){                       // 2 green, 1 red, 0 out of scope
  if(!inWin(x,y)) return 0;
  const i = idx(x,y);
  if(ROAD[i]) return 2;
  if(LAND && LAND[i]) return 1;
  return 0;
}

/* ---------- projection -------------------------------------------------- */
const MAXLAT = 85.0511287798066;
function ll2t(lon,lat){
  const la = Math.max(-MAXLAT, Math.min(MAXLAT, lat));
  const s = Math.sin(la*Math.PI/180);
  return [ (lon+180)/360*NZ,
           (0.5 - Math.log((1+s)/(1-s))/(4*Math.PI))*NZ ];
}
function t2ll(x,y){
  return [ x/NZ*360-180,
           Math.atan(Math.sinh(Math.PI*(1-2*y/NZ)))*180/Math.PI ];
}

/* ---------- offscreen tile raster (1 px = 1 explorer tile) -------------- */
const grid = document.createElement("canvas");
grid.width = W; grid.height = H;
const gctx = grid.getContext("2d");
let showRed = true;
let showGreen = true;
let mode = "coverage";
let BINS = null;            /* six thresholds, in km */
function computeBins(){
  if(!KMQ) return;
  const v = [];
  for(let i=0;i<NT;i++) if(ROAD[i] && KMQ[i] > 0) v.push(KMQ[i]);
  if(!v.length){ BINS = null; return; }
  v.sort((a,b)=>a-b);
  BINS = [];
  for(let k=1;k<=6;k++) BINS.push(v[Math.floor(v.length * k / 7)] * KM_SCALE);
}
function binOf(km){
  if(!BINS) return 0;
  let b = 0;
  while(b < BINS.length && km > BINS[b]) b++;
  return b;                 /* 0..6 */
}
function hex2rgb(h){
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}
const TA = Math.round(255 * (D.tileAlpha === undefined ? 0.80 : D.tileAlpha));
const TA_RED = Math.round(TA * 0.83);
const BM = D.basemap || null;
// With a raster basemap underneath, the vector country shapes must not be
// filled or they would hide it; and on a light basemap the labels have to
// flip to dark ink on a light halo.
const LIGHTBG = !!(BM && BM.light && !BM.grey);
const INK  = LIGHTBG ? "#16202e" : "#cfd8e3";
const INK2 = LIGHTBG ? "#0b1018" : "#e6edf3";
const HALO = LIGHTBG ? "rgba(255,255,255,0.9)" : "rgba(10,14,20,0.85)";
function paintGrid(){
  const img = gctx.createImageData(W,H), p = img.data;
  // A tile you already hold wins over everything else, in both views: that is
  // what makes "show me only my tiles" a matter of switching the other two
  // layers off, and what makes a red tile you have ridden read as held.
  const vis = showVisited && VISITED;
  if(mode === "density" && KMQ){
    if(!BINS) computeBins();
    const ramp = LIGHTBG ? BLUE : BLUE.slice().reverse();
    const rgb = ramp.map(hex2rgb);
    const zero = hex2rgb(ZERO_COLOUR);
    for(let i=0;i<NT;i++){
      const o = i<<2;
      if(vis && VISITED[i]){
        p[o]=HELD[0]; p[o+1]=HELD[1]; p[o+2]=HELD[2]; p[o+3]=TA;
      } else if(ROAD[i] && KMQ[i] > 0){
        if(!showGreen) continue;
        const c = rgb[binOf(KMQ[i] * KM_SCALE)];
        p[o]=c[0]; p[o+1]=c[1]; p[o+2]=c[2]; p[o+3]=TA;
      } else if(LAND && LAND[i]){
        if(!showRed) continue;
        p[o]=zero[0]; p[o+1]=zero[1]; p[o+2]=zero[2]; p[o+3]=TA_RED;
      }
    }
  } else {
    for(let i=0;i<NT;i++){
      const o = i<<2;
      if(vis && VISITED[i]){
        p[o]=HELD[0]; p[o+1]=HELD[1]; p[o+2]=HELD[2]; p[o+3]=TA;
      }
      else if(ROAD[i]){
        if(!showGreen) continue;
        p[o]=46; p[o+1]=160; p[o+2]=67; p[o+3]=TA;
      }
      else if(showRed && LAND && LAND[i]){ p[o]=218; p[o+1]=54; p[o+2]=51; p[o+3]=TA_RED; }
    }
  }
  gctx.putImageData(img,0,0);
}
paintGrid();

/* ---------- view -------------------------------------------------------- */
const cv = document.getElementById("map"), ctx = cv.getContext("2d");
let DPR = 1, VW = 0, VH = 0;
const view = { cx:0, cy:0, scale:1 };   // scale = css px per tile unit

function resize(){
  if(document.getElementById("legend").style.display === "block") placeLegend();
  DPR = Math.min(window.devicePixelRatio||1, 2);
  VW = cv.clientWidth; VH = cv.clientHeight;
  cv.width = Math.round(VW*DPR); cv.height = Math.round(VH*DPR);
  draw();
}
function fitAll(){
  const m = 0.06;
  view.scale = Math.min(VW*(1-m)/W, VH*(1-m)/H);
  view.cx = D.x0 + W/2; view.cy = D.y0 + H/2;
}
const sx = tx => (tx - view.cx)*view.scale + VW/2;
const sy = ty => (ty - view.cy)*view.scale + VH/2;
const ix = px => (px - VW/2)/view.scale + view.cx;
const iy = py => (py - VH/2)/view.scale + view.cy;

/* ---------- raster basemap (local files only; blocked in sandboxes) ----- */
const timg = new Map();
let rasterSkipped = false;
function basemapTile(z,x,y){
  const k = z+"/"+x+"/"+y;
  let e = timg.get(k);
  if(e === undefined){
    const im = new Image();
    im.onload = ()=>{ e.ok = true; requestDraw(); };
    im.onerror = ()=>{ e.ok = false; };
    e = {im:im, ok:false};
    timg.set(k,e);
    const sub = BM.sub ? BM.sub[Math.abs(x + y) % BM.sub.length] : "";
    const r2x = (window.devicePixelRatio || 1) > 1.3 ? "@2x" : "";
    im.src = BM.url.replace("{s}", sub).replace("{z}", z)
                   .replace("{x}", x).replace("{y}", y).replace("{r}", r2x);
    if(timg.size > 600){ for(const kk of timg.keys()){ timg.delete(kk); if(timg.size<400) break; } }
  }
  return e;
}
// The raster basemap is confined to the mapped country: everything outside
// keeps the plain dark background. The regions outline is used when present
// because it is far sharper than the 1:10m country shape.
let _clip = null;
function countryClip(){
  if(_clip) return _clip;
  _clip = (REGIONS && REGIONS.rings.length)
            ? REGIONS.rings
            : RINGS.filter(r => r.h).map(r => r.pts);
  return _clip;
}

function drawRaster(alpha, blend){
  // pick the slippy zoom whose 256-px tiles best match the current scale:
  // one z14 tile spans view.scale css px, so a z-tile spans scale*2^(Z-z) px,
  // and we want that close to 256.
  let z = Math.round(Math.log2(view.scale) + Z - 8);
  z = Math.max(2, Math.min(16, z));
  const f = Math.pow(2, z - Z);                 // basemap tiles per z14 tile
  const bx0 = Math.floor(ix(0)*f), bx1 = Math.ceil(ix(VW)*f);
  const by0 = Math.floor(iy(0)*f), by1 = Math.ceil(iy(VH)*f);
  const n = 1<<z;
  // OSM's tile policy forbids bulk downloading. If the current view would need
  // more than a modest number of tiles, draw none: the vector outlines carry
  // the map until the user zooms into something reasonable.
  const need = (bx1-bx0+1) * (by1-by0+1);
  if(need > (BM.budget || 120)){ rasterSkipped = true; return; }
  rasterSkipped = false;
  ctx.save();
  if(BM.clip){
    const rings = countryClip();
    if(rings.length){
      ctx.beginPath();
      for(const pts of rings){
        ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
        for(let i=1;i<pts.length;i++) ctx.lineTo(sx(pts[i][0]), sy(pts[i][1]));
        ctx.closePath();
      }
      ctx.clip();
    }
  }
  ctx.globalAlpha = alpha;
  if(blend) ctx.globalCompositeOperation = blend;
  if(BM.grey && "filter" in ctx) ctx.filter = "grayscale(1)";
  for(let bx=bx0; bx<=bx1; bx++){
    for(let by=by0; by<=by1; by++){
      if(by<0||by>=n) continue;
      const wx = ((bx % n)+n)%n;
      const e = basemapTile(z,wx,by);
      if(!e.ok) continue;
      // Snap each tile to whole pixels and derive its size from the next
      // boundary, so neighbours abut exactly. Drawing them 1px oversized
      // instead would composite the shared strip twice and leave a bright
      // seam -- visible as a spurious half-tile grid at some zooms.
      const X0 = Math.round(sx(bx/f)),     Y0 = Math.round(sy(by/f));
      const X1 = Math.round(sx((bx+1)/f)), Y1 = Math.round(sy((by+1)/f));
      ctx.drawImage(e.im, X0, Y0, X1 - X0, Y1 - Y0);
    }
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  if("filter" in ctx) ctx.filter = "none";
  ctx.restore();
}

/* ---------- vector basemap --------------------------------------------- */
const RINGS = (D.base && D.base.rings) ? D.base.rings.map(r=>{
  const pts = []; let lx=0, ly=0;
  for(let i=0;i<r.d.length;i+=2){
    lx += r.d[i]; ly += r.d[i+1];
    pts.push(ll2t(lx/1e4, ly/1e4));
  }
  return {h:r.h, pts:pts};
}) : [];

/* ---------- administrative outlines ------------------------------------ */
function decodeAdmin(layer){
  if(!layer) return null;
  const rings = layer.rings.map(d=>{
    const pts = []; let lx=0, ly=0;
    for(let i=0;i<d.length;i+=2){
      lx += d[i]; ly += d[i+1];
      pts.push(ll2t(lx/1e4, ly/1e4));
    }
    return pts;
  });
  const labels = (layer.labels||[]).map(l=>({name:l[0], code:l[1], t:ll2t(l[2], l[3])}));
  return {rings:rings, labels:labels};
}
const REGIONS = decodeAdmin(D.admin && D.admin.regions);
const DEPTS   = decodeAdmin(D.admin && D.admin.departments);

function drawAdmin(layer, col, lw){
  if(!layer) return;
  ctx.strokeStyle = col;
  ctx.lineWidth = lw;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for(const p of layer.rings){
    ctx.moveTo(sx(p[0][0]), sy(p[0][1]));
    for(let i=1;i<p.length;i++) ctx.lineTo(sx(p[i][0]), sy(p[i][1]));
    ctx.closePath();
  }
  ctx.stroke();
}

/* Admin labels are laid out before the cities are drawn, so a city can drop
   its own text where an admin name already occupies the spot (Paris the city
   vs Paris the departement). */
let activeLabels = [];
function layoutAdminLabels(){
  activeLabels = [];
  const useDep = showDep && DEPTS && view.scale >= 2.2;
  const useReg = showReg && REGIONS && view.scale < 2.2;
  const layer = useDep ? DEPTS : (useReg ? REGIONS : null);
  if(!layer) return;
  const size = useDep ? 12 : 13;
  const col  = LIGHTBG ? (useDep ? "#223042" : "#0b1018")
                       : (useDep ? "#dbe7f5" : "#ffffff");
  for(const l of layer.labels){
    const X = sx(l.t[0]), Y = sy(l.t[1]);
    if(X < -60 || X > VW+60 || Y < -20 || Y > VH+20) continue;
    activeLabels.push({name:l.name, X:X, Y:Y, size:size, col:col});
  }
}
function paintAdminLabels(){
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for(const l of activeLabels){
    ctx.font = "600 " + l.size + "px ui-sans-serif,system-ui,sans-serif";
    ctx.strokeStyle = HALO;
    ctx.lineWidth = 3.5;
    ctx.strokeText(l.name, l.X, l.Y);
    ctx.fillStyle = l.col;
    ctx.fillText(l.name, l.X, l.Y);
  }
  ctx.textAlign = "left";
}
function labelOccupied(X, Y){
  for(const l of activeLabels){
    if(Math.abs(l.X - X) < 90 && Math.abs(l.Y - Y) < 16) return true;
  }
  return false;
}

function drawVector(){
  for(const r of RINGS){
    ctx.beginPath();
    const p = r.pts;
    ctx.moveTo(sx(p[0][0]), sy(p[0][1]));
    for(let i=1;i<p.length;i++) ctx.lineTo(sx(p[i][0]), sy(p[i][1]));
    ctx.closePath();
    if(!BM || BM.clip || !r.h){    // keep the dark land everywhere the raster
      ctx.fillStyle = r.h ? "#1c2531" : "#141a22";   // will not cover
      ctx.fill();
    }
    ctx.strokeStyle = (LIGHTBG && !BM.clip) ? (r.h ? "#6b7a8d" : "#aab4c0")
                                            : (r.h ? "#4a5a70" : "#262f3c");
    ctx.lineWidth = r.h ? 1.2 : 0.8;
    ctx.stroke();
  }
}

/* ---------- the route --------------------------------------------------- */
const ROUTE = (function(){
  const r = D.route;
  if(!r || !r.d || !r.d.length) return null;
  const pts = []; let lx = 0, ly = 0;
  for(let i=0;i<r.d.length;i+=2){
    lx += r.d[i]; ly += r.d[i+1];
    pts.push(ll2t(lx/1e5, ly/1e5));
  }
  // cumulative length along the track, so the profile can be pointed at a
  // place on the map
  const cum = [0];
  for(let i=1;i<pts.length;i++)
    cum.push(cum[i-1] + Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]));
  return {pts:pts, cum:cum, total:cum[cum.length-1],
          stats:r.stats||{}, start:r.start, end:r.end, rect:r.rect,
          profile:r.profile||null, pinned:r.pinned||null,
          skipped:r.skipped||null, wanted:r.wanted||null};
})();

/* ---------- elevation profile ------------------------------------------ */
let profMark = null;          /* 0..1 along the ride, or null */

function routePointAt(frac){
  if(!ROUTE || !ROUTE.total) return null;
  const want = frac * ROUTE.total, c = ROUTE.cum;
  let lo = 0, hi = c.length - 1;
  while(lo < hi - 1){ const mid = (lo + hi) >> 1; (c[mid] <= want ? lo = mid : hi = mid); }
  const span = c[hi] - c[lo] || 1, t = (want - c[lo]) / span;
  return [ROUTE.pts[lo][0] + (ROUTE.pts[hi][0]-ROUTE.pts[lo][0]) * t,
          ROUTE.pts[lo][1] + (ROUTE.pts[hi][1]-ROUTE.pts[lo][1]) * t];
}

function drawProfile(){
  const pr = ROUTE && ROUTE.profile;
  const box = document.getElementById("profile");
  if(!pr || !pr.elev || pr.elev.length < 2){ box.style.display = "none"; return; }
  box.style.display = "block";
  const cv2 = document.getElementById("prof");
  const w = cv2.clientWidth, h = cv2.clientHeight;
  cv2.width = Math.round(w * DPR); cv2.height = Math.round(h * DPR);
  const c = cv2.getContext("2d");
  c.setTransform(DPR,0,0,DPR,0,0);
  c.clearRect(0,0,w,h);

  const e = pr.elev, n = e.length;
  const lo = pr.min, hi = pr.max, span = Math.max(1, hi - lo);
  const pad = 6;
  const X = i => i / (n - 1) * w;
  const Y = v => h - pad - (v - lo) / span * (h - 2*pad);

  // recessive gridlines at round altitudes
  const stepM = span > 800 ? 200 : span > 300 ? 100 : span > 120 ? 50 : 20;
  c.strokeStyle = "rgba(255,255,255,0.10)"; c.lineWidth = 1;
  c.fillStyle = "rgba(180,190,205,0.75)";
  c.font = "10px ui-sans-serif,system-ui,sans-serif";
  c.textBaseline = "middle";
  for(let v = Math.ceil(lo/stepM)*stepM; v <= hi; v += stepM){
    const y = Math.round(Y(v)) + 0.5;
    c.beginPath(); c.moveTo(0,y); c.lineTo(w,y); c.stroke();
    c.fillText(v + " m", 2, y - 6);
  }

  c.beginPath();
  c.moveTo(X(0), h);
  for(let i=0;i<n;i++) c.lineTo(X(i), Y(e[i]));
  c.lineTo(X(n-1), h); c.closePath();
  const grad = c.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(57,135,229,0.55)");
  grad.addColorStop(1, "rgba(57,135,229,0.06)");
  c.fillStyle = grad; c.fill();

  c.beginPath();
  for(let i=0;i<n;i++) (i ? c.lineTo(X(i), Y(e[i])) : c.moveTo(X(i), Y(e[i])));
  c.strokeStyle = "#86b6ef"; c.lineWidth = 2; c.stroke();

  if(profMark !== null){
    const i = Math.max(0, Math.min(n-1, Math.round(profMark * (n-1))));
    const x = X(i), y = Y(e[i]);
    c.strokeStyle = "rgba(255,210,74,0.9)"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke();
    c.fillStyle = "#ffd24a";
    c.beginPath(); c.arc(x, y, 3.5, 0, 6.2832); c.fill();
    document.getElementById("pf-at").textContent =
      (profMark * pr.km).toFixed(1) + " km  \u00b7  " + e[i] + " m";
  } else {
    document.getElementById("pf-at").textContent = "";
  }
}

function drawRouteBrief(){
  // the rectangle the ride was asked to cover, plus the pinned start and
  // finish tiles -- dashed, so it reads as the brief rather than as data
  if(!ROUTE) return;
  ctx.save();
  ctx.setLineDash([7, 5]);
  if(ROUTE.rect && ROUTE.rect.length === 4){
    const [x0,y0,x1,y1] = ROUTE.rect;
    outline(x0, y0, x1-x0+1, y1-y0+1, "rgba(255,255,255,0.85)", 2);
  }
  ctx.setLineDash([]);
  if(ROUTE.start) outline(ROUTE.start[0], ROUTE.start[1], 1, 1, "#2ea043", 2.5);
  if(ROUTE.end)   outline(ROUTE.end[0], ROUTE.end[1], 1, 1, "#da3633", 2.5);
  ctx.restore();
}

function drawRoute(){
  if(!ROUTE) return;
  const p = ROUTE.pts;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for(const pass of [{c:"rgba(8,12,18,0.85)", w:6}, {c:"#ffd24a", w:2.6}]){
    ctx.strokeStyle = pass.c;
    ctx.lineWidth = pass.w;
    ctx.beginPath();
    ctx.moveTo(sx(p[0][0]), sy(p[0][1]));
    for(let i=1;i<p.length;i++) ctx.lineTo(sx(p[i][0]), sy(p[i][1]));
    ctx.stroke();
  }
  // the tiles this ride set out to collect
  for(const t of (ROUTE.wanted || []))
    outline(t[0], t[1], 1, 1, "#ffd24a", 2);

  // the tiles that had to be given up: shown where they are, so the gap in
  // the square is something you can see rather than a number in a warning
  for(const t of (ROUTE.skipped || [])){
    const X = sx(t[0]), Y = sy(t[1]), W = view.scale;
    ctx.save();
    ctx.beginPath(); ctx.rect(X, Y, W, W); ctx.clip();
    ctx.strokeStyle = "rgba(218,54,51,0.85)"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for(let d = -W; d < W * 2; d += 7){
      ctx.moveTo(X + d, Y); ctx.lineTo(X + d + W, Y + W);
    }
    ctx.stroke();
    ctx.restore();
    outline(t[0], t[1], 1, 1, "#da3633", 1.5);
  }

  // where you asked to start, against where the network actually let you:
  // a hollow ring on the point you clicked, a hair line to the junction
  const pin = ROUTE.pinned;
  if(pin){
    const marks = [[pin.start, "#2ea043"], [pin.end, "#da3633"]];
    for(const v of (pin.via || [])) marks.push([v, "#ffd24a"]);
    for(const [q, col] of marks){
      if(!q) continue;
      const a = ll2t(q.asked[0], q.asked[1]);
      const b = ll2t(q.node[0], q.node[1]);
      ctx.beginPath();
      ctx.moveTo(sx(a[0]), sy(a[1])); ctx.lineTo(sx(b[0]), sy(b[1]));
      ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.setLineDash([3,3]);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(sx(a[0]), sy(a[1]), 4.5, 0, 6.2832);
      ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.stroke();
    }
    // the waypoints keep the planner's symbol, and stay unnumbered: the ride
    // order is the solver's, not the order you clicked them in
    for(const v of (pin.via || [])){
      const b = ll2t(v.node[0], v.node[1]);
      diamond(b[0], b[1], "#ffd24a");
    }
  }
  const ends = [[p[0], "#2ea043", "S"], [p[p.length-1], "#da3633", "F"]];
  ctx.font = "700 10px ui-sans-serif,system-ui,sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for(const [q, col, lab] of ends){
    const X = sx(q[0]), Y = sy(q[1]);
    ctx.beginPath(); ctx.arc(X, Y, 7, 0, 6.2832);
    ctx.fillStyle = col; ctx.fill();
    ctx.strokeStyle = "rgba(8,12,18,0.9)"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.fillText(lab, X, Y);
  }
  ctx.textAlign = "left";
}

/* ---------- state ------------------------------------------------------- */
let showTiles=true, showGrid=true, showBase=true, showCities=true;
let showRoute=true;
/* planning: 0 off, 1 awaiting the rectangle, 2 the start tile,
   3 the end tile, 4 done */
let planMode = 0, planRect = null, planStart = null, planEnd = null,
    planVia = [], band = null;
let planKind = "square";          // or "seeker"
let planWant = new Map();         // "x,y" -> [x,y], the tiles to collect
const lastStep = () => 5;      // both planners end on the name
let showReg=true, showDep=true;
let hover=null, pick=[], best=null;
let raf = 0;
// The first resize() happens before fitAll(), when the view still sits at
// scale 1 over tile 0,0. Drawing then would fire a burst of basemap requests
// for a meaningless part of the world, so nothing paints until we are framed.
let ready = false;
function requestDraw(){ if(!raf) raf = requestAnimationFrame(()=>{ raf=0; draw(); }); }

function draw(){
  if(!ready) return;
  ctx.setTransform(DPR,0,0,DPR,0,0);
  ctx.clearRect(0,0,VW,VH);
  ctx.fillStyle = (BM && BM.light && !BM.grey && !BM.clip) ? "#eceff3" : "#0e1116";
  ctx.fillRect(0,0,VW,VH);

  if(showBase){
    if(RINGS.length) drawVector();   // dark land shapes, as without a basemap
    if(BM) drawRaster(BM.opacity === undefined ? 1 : BM.opacity, null);
  }

  if(showTiles){
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(grid, sx(D.x0), sy(D.y0), W*view.scale, H*view.scale);
    ctx.imageSmoothingEnabled = true;
  }

  // tile grid lines, once tiles are big enough to be legible
  if(showGrid && view.scale > 7){
    const a = Math.min(0.5, (view.scale-7)/60);
    ctx.strokeStyle = (LIGHTBG ? "rgba(40,50,65," : "rgba(255,255,255,")
                      + a.toFixed(3) + ")";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const X0 = Math.floor(ix(0)), X1 = Math.ceil(ix(VW));
    const Y0 = Math.floor(iy(0)), Y1 = Math.ceil(iy(VH));
    for(let x=X0;x<=X1;x++){ const s=Math.round(sx(x))+0.5; ctx.moveTo(s,0); ctx.lineTo(s,VH); }
    for(let y=Y0;y<=Y1;y++){ const s=Math.round(sy(y))+0.5; ctx.moveTo(0,s); ctx.lineTo(VW,s); }
    ctx.stroke();
  }

  // Re-assert the basemap over the tint. Multiply leaves pale ground almost
  // untouched while the dark road lines punch back through, so the streets
  // stay readable instead of drowning under a flat wash of colour.
  if(showBase && showTiles && BM && BM.detail > 0 && !rasterSkipped){
    drawRaster(BM.detail, "multiply");
  }

  if(showDep && DEPTS) drawAdmin(DEPTS,
      LIGHTBG ? "rgba(20,30,45,0.30)" : "rgba(226,238,255,0.30)", 1);
  if(showReg && REGIONS) drawAdmin(REGIONS,
      LIGHTBG ? "rgba(10,16,26,0.70)" : "rgba(255,255,255,0.72)", 2);
  layoutAdminLabels();

  if(showCities && D.cities){
    ctx.font = "600 11px ui-sans-serif,system-ui,sans-serif";
    ctx.textBaseline = "middle";
    // tier 2 (the smaller prefectures) only appears once there is room
    const minTier = view.scale < 1.8 ? 1 : 2;
    const placed = [];
    for(const c of D.cities){
      const name = c[0], tier = c[3] || 1;
      if(tier > minTier) continue;
      const t = ll2t(c[1], c[2]), X = sx(t[0]), Y = sy(t[1]);
      if(X<-80||X>VW+80||Y<-20||Y>VH+20) continue;
      ctx.fillStyle = INK2;
      ctx.beginPath(); ctx.arc(X,Y,2.6,0,6.2832); ctx.fill();
      if(labelOccupied(X+6, Y)) continue;
      // and do not stack two city names on top of each other
      const wdt = ctx.measureText(name).width;
      let clash = false;
      for(const q of placed){
        if(X+6 < q.x1 + 6 && q.x0 - 6 < X + 6 + wdt
           && Math.abs(q.y - Y) < 13){ clash = true; break; }
      }
      if(clash) continue;
      placed.push({x0:X+6, x1:X+6+wdt, y:Y});
      ctx.strokeStyle = HALO; ctx.lineWidth = 3;
      ctx.strokeText(name, X+6, Y);
      ctx.fillStyle = INK;
      ctx.fillText(name, X+6, Y);
    }
  }

  paintAdminLabels();

  if(band){
    const r = bandRect();
    outline(r.x0, r.y0, r.x1-r.x0+1, r.y1-r.y0+1, "#ffd24a", 2);
    const d = rectDims(r);
    const txt = d.w + " \u00d7 " + d.h + "  \u00b7  " + d.n + " tiles  \u00b7  "
              + d.kx.toFixed(1) + " \u00d7 " + d.ky.toFixed(1) + " km";
    ctx.font = "600 12px ui-sans-serif,system-ui,sans-serif";
    ctx.textBaseline = "bottom";
    const X = sx(r.x0), Y = sy(r.y0) - 7;
    const wdt = ctx.measureText(txt).width;
    ctx.fillStyle = "rgba(10,14,20,0.82)";
    ctx.fillRect(X - 5, Y - 16, wdt + 10, 20);
    ctx.fillStyle = "#ffd24a";
    ctx.fillText(txt, X, Y);
    ctx.textBaseline = "middle";
  } else if(planRect){
    outline(planRect.x0, planRect.y0, planRect.x1-planRect.x0+1,
            planRect.y1-planRect.y0+1, "#ffd24a", 2);
  }
  if(flashTile){
    const t = (flashUntil - performance.now()) / 4000;
    const pulse = 0.45 + 0.55 * Math.abs(Math.sin(performance.now() / 180));
    outline(flashTile[0], flashTile[1], 1, 1,
            "rgba(255,210,74," + pulse.toFixed(2) + ")", 3);
    const r = (1 + 14 * (1 - (t * 4 % 1))) * view.scale;
    ctx.beginPath();
    ctx.arc(sx(flashTile[0] + 0.5), sy(flashTile[1] + 0.5), Math.max(r, 2),
            0, 6.2832);
    ctx.strokeStyle = "rgba(255,210,74," + (0.5 * pulse).toFixed(2) + ")";
    ctx.lineWidth = 2; ctx.stroke();
  }
  // the tiles picked for a tile hunt
  if(planKind === "seeker"){
    for(const [x, y] of planWant.values()){
      ctx.fillStyle = "rgba(255,210,74,0.30)";
      ctx.fillRect(sx(x), sy(y), view.scale, view.scale);
      outline(x, y, 1, 1, "#ffd24a", 2);
    }
  }

  // no numbers on the waypoints: the order you clicked them in is not the
  // order they will be ridden in -- the solver decides that
  for(const v of planVia) diamond(v.x, v.y, "#ffd24a");
  if(planStart) marker(planStart, "#2ea043", "S");
  if(planEnd)   marker(planEnd, "#da3633", "F");

  if(showRoute && ROUTE){
    drawRouteBrief(); drawRoute();
    if(profMark !== null){
      const q = routePointAt(profMark);
      if(q){
        const X = sx(q[0]), Y = sy(q[1]);
        ctx.beginPath(); ctx.arc(X, Y, 8, 0, 6.2832);
        ctx.fillStyle = "rgba(255,210,74,0.25)"; ctx.fill();
        ctx.beginPath(); ctx.arc(X, Y, 4.5, 0, 6.2832);
        ctx.fillStyle = "#ffd24a"; ctx.fill();
        ctx.strokeStyle = "rgba(8,12,18,0.9)"; ctx.lineWidth = 2; ctx.stroke();
      }
    }
  }

  if(best) outline(best.x, best.y, best.n, best.n, "#9085e9", 2.5);
  if(pick.length === 2){
    const x0=Math.min(pick[0][0],pick[1][0]), x1=Math.max(pick[0][0],pick[1][0]);
    const y0=Math.min(pick[0][1],pick[1][1]), y1=Math.max(pick[0][1],pick[1][1]);
    outline(x0,y0,x1-x0+1,y1-y0+1,"#539bf5",2);
  } else if(pick.length === 1){
    outline(pick[0][0],pick[0][1],1,1,"#539bf5",2);
  }
  if(hover) outline(hover[0],hover[1],1,1,"#ffffff",1.5);

  drawScale();
}
function diamond(tx, ty, col){
  const X = sx(tx), Y = sy(ty), r = 6;
  ctx.beginPath();
  ctx.moveTo(X, Y-r); ctx.lineTo(X+r, Y); ctx.lineTo(X, Y+r);
  ctx.lineTo(X-r, Y); ctx.closePath();
  ctx.fillStyle = col; ctx.fill();
  ctx.strokeStyle = "rgba(8,12,18,0.9)"; ctx.lineWidth = 1.5; ctx.stroke();
}

function marker(pt, col, letter){
  // the point you clicked, with its tile outlined behind it: the ride will
  // begin at the nearest junction, which the solver reports once it has the
  // road database in hand
  outline(pt.tx, pt.ty, 1, 1, col, 1.5);
  const X = sx(pt.x), Y = sy(pt.y);
  ctx.beginPath(); ctx.moveTo(X-9, Y); ctx.lineTo(X+9, Y);
  ctx.moveTo(X, Y-9); ctx.lineTo(X, Y+9);
  ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(X, Y, 5, 0, 6.2832);
  ctx.fillStyle = col; ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = 1; ctx.stroke();
  ctx.font = "600 10px ui-sans-serif,system-ui,sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = "#fff"; ctx.fillText(letter, X, Y + 0.5);
  ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
}

function outline(x,y,w,h,col,lw){
  ctx.strokeStyle = col; ctx.lineWidth = lw;
  ctx.strokeRect(sx(x), sy(y), w*view.scale, h*view.scale);
}

function drawScale(){
  const mid = t2ll(view.cx, view.cy)[1];
  const mPerTile = 40075016.686 * Math.cos(mid*Math.PI/180) / NZ;
  const mPerPx = mPerTile / view.scale;
  let target = 90*mPerPx, p = Math.pow(10, Math.floor(Math.log10(target)));
  let n = p; for(const m of [1,2,5,10]){ if(p*m <= target) n = p*m; }
  document.getElementById("sc-ln").style.width = (n/mPerPx).toFixed(0)+"px";
  document.getElementById("sc-lab").textContent =
    n>=1000 ? (n/1000)+" km" : n+" m";
}

/* ---------- interaction ------------------------------------------------- */
let drag=null;
cv.addEventListener("pointerdown", e=>{
  cv.setPointerCapture(e.pointerId);
  if(((planKind === "square" && planMode === 1)
      || (planKind === "seeker" && planMode === 3))
     && (e.ctrlKey || e.metaKey)){
    const r = cv.getBoundingClientRect();
    const t = [Math.floor(ix(e.clientX-r.left)), Math.floor(iy(e.clientY-r.top))];
    band = {a:t, b:t};
    requestDraw();
    return;                       // a planning drag must not pan the map
  }
  drag = {px:e.clientX, py:e.clientY, cx:view.cx, cy:view.cy, moved:0};
  cv.classList.add("drag");
});
cv.addEventListener("pointermove", e=>{
  const r = cv.getBoundingClientRect();
  if(band){
    band.b = [Math.floor(ix(e.clientX-r.left)), Math.floor(iy(e.clientY-r.top))];
    const tx = band.b[0], ty = band.b[1];
    hover = [tx,ty]; readout(tx,ty); renderPlan(); requestDraw();
    return;
  }
  if(drag){
    const dx = e.clientX-drag.px, dy = e.clientY-drag.py;
    drag.moved = Math.max(drag.moved, Math.abs(dx)+Math.abs(dy));
    view.cx = drag.cx - dx/view.scale;
    view.cy = drag.cy - dy/view.scale;
  }
  const tx = Math.floor(ix(e.clientX-r.left)), ty = Math.floor(iy(e.clientY-r.top));
  hover = [tx,ty];
  readout(tx,ty);
  requestDraw();
});
cv.addEventListener("pointerup", e=>{
  cv.classList.remove("drag");
  if(band){
    const b = bandRect();
    band = null;
    if(planKind === "seeker"){
      for(let x = b.x0; x <= b.x1; x++)
        for(let y = b.y0; y <= b.y1; y++)
          planWant.set(x + "," + y, [x, y]);
    } else {
      planRect = b;
      planMode = 2; planStart = planEnd = null; planVia = [];
    }
    renderPlan(); requestDraw();
    return;
  }
  const wasClick = drag && drag.moved < 4;
  drag = null;
  if(!wasClick) return;
  const r = cv.getBoundingClientRect();
  const fx = ix(e.clientX-r.left), fy = iy(e.clientY-r.top);
  const tx = Math.floor(fx), ty = Math.floor(fy);
  const seeking = planKind === "seeker";
  if(planMode >= (seeking ? 1 : 2) && planMode <= 4){
    // an exact point, not a tile: it may sit outside the area, which is what
    // you want when you park or take a train to the edge of it
    const ll = t2ll(fx, fy);
    const pt = {x:fx, y:fy, lon:ll[0], lat:ll[1], tx:tx, ty:ty};
    const firstPoint = seeking ? 1 : 2;
    if(planMode === firstPoint){ planStart = pt; planMode = firstPoint + 1; }
    else if(planMode === firstPoint + 1){ planEnd = pt; planMode = firstPoint + 2; }
    else if(seeking && planMode === 3){
      // a click toggles: picking a tile twice unpicks it
      const key = tx + "," + ty;
      if(planWant.has(key)) planWant.delete(key);
      else planWant.set(key, [tx, ty]);
    }
    else { planVia.push(pt); }      // stay on 4: add as many as you like
    renderPlan(); requestDraw();
    return;
  }
  if(e.shiftKey){
    if(pick.length >= 2) pick = [];
    pick.push([tx,ty]);
    if(pick.length === 2) showZone();
  } else {
    pick = []; document.getElementById("zone").style.display="none";
  }
  requestDraw();
});
cv.addEventListener("contextmenu", e=>{
  if(planMode < 4 || !planVia.length) return;
  const r = cv.getBoundingClientRect();
  const fx = ix(e.clientX-r.left), fy = iy(e.clientY-r.top);
  let best = -1, bd = 14 / view.scale;      // within 14 screen pixels
  planVia.forEach((v, i) => {
    const d = Math.hypot(v.x - fx, v.y - fy);
    if(d < bd){ bd = d; best = i; }
  });
  if(best >= 0){
    e.preventDefault();
    planVia.splice(best, 1);
    renderPlan(); requestDraw();
  }
});
cv.addEventListener("pointerleave", ()=>{ hover=null; requestDraw(); });
cv.addEventListener("wheel", e=>{
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  const tx = ix(mx), ty = iy(my);
  const k = Math.exp(-e.deltaY * (e.deltaMode===1 ? 0.05 : 0.0016));
  view.scale = Math.max(0.05, Math.min(400, view.scale*k));
  view.cx = tx - (mx - VW/2)/view.scale;
  view.cy = ty - (my - VH/2)/view.scale;
  requestDraw();
}, {passive:false});

function readout(x,y){
  const st = status(x,y);
  document.getElementById("r-tile").textContent = x+", "+y;
  const el = document.getElementById("r-stat");
  el.innerHTML = st===2 ? '<span class="sw" style="background:var(--green)"></span>paved road'
               : st===1 ? '<span class="sw" style="background:var(--red)"></span>no paved road'
               : '<span class="sw" style="background:#555"></span>out of scope';
  const a = t2ll(x, y+1), b = t2ll(x+1, y);
  document.getElementById("r-ll").textContent =
    ((a[1]+b[1])/2).toFixed(5)+", "+((a[0]+b[0])/2).toFixed(5);
  const latm = (b[1]-a[1])*110574, lonm = (b[0]-a[0])*111320*Math.cos((a[1]+b[1])/2*Math.PI/180);
  document.getElementById("r-sz").textContent =
    (lonm/1000).toFixed(2)+" x "+(latm/1000).toFixed(2)+" km";
  const kmEl = document.getElementById("r-km");
  if(!KMQ || !inWin(x,y)) kmEl.textContent = "\u2014";
  else {
    const km = kmOf(idx(x,y));
    kmEl.textContent = km > 0 ? km.toFixed(2)+" km" : (st===1 ? "none" : "\u2014");
  }
}

function zoneStats(x0,y0,x1,y1){
  let g=0,r=0,o=0;
  for(let y=y0;y<=y1;y++) for(let x=x0;x<=x1;x++){
    const s=status(x,y); if(s===2) g++; else if(s===1) r++; else o++;
  }
  return {g:g,r:r,o:o};
}
function maxSquareIn(x0,y0,x1,y1){
  const w=x1-x0+1, h=y1-y0+1;
  const dp = new Int32Array(w*h);
  let bn=0,bx=0,by=0;
  for(let j=0;j<h;j++) for(let i=0;i<w;i++){
    const ok = status(x0+i, y0+j)===2;
    let v=0;
    if(ok){
      v = 1 + Math.min(i>0?dp[j*w+i-1]:0,
                       j>0?dp[(j-1)*w+i]:0,
                       (i>0&&j>0)?dp[(j-1)*w+i-1]:0);
    }
    dp[j*w+i]=v;
    if(v>bn){ bn=v; bx=x0+i-v+1; by=y0+j-v+1; }
  }
  return bn? {n:bn,x:bx,y:by} : null;
}
function showZone(){
  const x0=Math.min(pick[0][0],pick[1][0]), x1=Math.max(pick[0][0],pick[1][0]);
  const y0=Math.min(pick[0][1],pick[1][1]), y1=Math.max(pick[0][1],pick[1][1]);
  const w=x1-x0+1, h=y1-y0+1, s=zoneStats(x0,y0,x1,y1);
  const sq = maxSquareIn(x0,y0,x1,y1);
  document.getElementById("zone").style.display="block";
  document.getElementById("z-cor").textContent = x0+","+y0+"  →  "+x1+","+y1;
  document.getElementById("z-grid").textContent = w+" × "+h+" = "+(w*h);
  document.getElementById("z-green").textContent = s.g+"  ("+(100*s.g/(w*h)).toFixed(1)+"%)";
  document.getElementById("z-red").textContent = s.r + (s.o? "  (+"+s.o+" out)":"");
  document.getElementById("z-sq").textContent = sq? (sq.n+" × "+sq.n+" @ "+sq.x+","+sq.y) : "none";
  document.getElementById("z-cmd").textContent =
    "tilehunt.py zone --corners "+x0+","+y0+" "+x1+","+y1;
  best = sq;
}

function fmtKm(v){
  return v >= 10 ? v.toFixed(0) : (v >= 1 ? v.toFixed(1) : v.toFixed(2));
}
function placeLegend(){
  // the left column stacks the panels itself now; nothing to position by hand
}
function renderLegend(){
  const box = document.getElementById("legend");
  if(mode !== "density" || !KMQ || !BINS){ box.style.display = "none"; return; }
  box.style.display = "block";
  const ramp = LIGHTBG ? BLUE : BLUE.slice().reverse();
  const rows = [];
  rows.push('<b><span class="sw" style="background:'+ZERO_COLOUR+'"></span></b>'
            + '<span>none</span>');
  for(let b=0;b<ramp.length;b++){
    const lo = b === 0 ? 0 : BINS[b-1];
    const hi = b < BINS.length ? BINS[b] : null;
    const label = hi === null ? "\u2265 " + fmtKm(lo) + " km"
                              : fmtKm(lo) + " \u2013 " + fmtKm(hi);
    rows.push('<b><span class="sw" style="background:'+ramp[b]+'"></span></b>'
              + '<span>' + label + '</span>');
  }
  document.getElementById("leg-rows").innerHTML = rows.join("");
  placeLegend();
}

function rectDims(r){
  const w = r.x1 - r.x0 + 1, h = r.y1 - r.y0 + 1;
  const midLat = t2ll(r.x0, (r.y0 + r.y1) / 2 + 0.5)[1];
  const kmPerTile = 40075.016686 * Math.cos(midLat * Math.PI / 180) / NZ;
  return {w:w, h:h, n:w*h, kx:w*kmPerTile, ky:h*kmPerTile};
}

function bandRect(){
  if(!band) return null;
  return {x0:Math.min(band.a[0],band.b[0]), x1:Math.max(band.a[0],band.b[0]),
          y0:Math.min(band.a[1],band.b[1]), y1:Math.max(band.a[1],band.b[1])};
}

function safeRunName(v){
  // the name becomes a folder, so keep it to what every filesystem accepts.
  // The same cleaning runs in the tool, so what you see here is what you get.
  return String(v || "").trim()
         .normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
         .replace(/[^A-Za-z0-9._-]+/g, "-")
         .replace(/^[-._]+|[-._]+$/g, "").slice(0, 64).replace(/[-._]+$/, "");
}

function renderPlan(){
  const box = document.getElementById("plan");
  if(!planMode){ box.style.display = "none"; cv.classList.remove("plan"); return; }
  box.style.display = "block";
  const seek = planKind === "seeker";
  cv.classList.toggle("plan", seek ? planMode === 3 : planMode === 1);
  document.getElementById("plan-steps-square").style.display =
    seek ? "none" : "block";
  document.getElementById("plan-steps-seek").style.display =
    seek ? "block" : "none";
  document.getElementById("plan-title").textContent =
    seek ? "TileSeeker" : "SquarePlanner";

  const steps = seek
    ? [[1,"qs1"],[2,"qs2"],[3,"qs3"],[4,"qs4"],[5,"qs5"]]
    : [[1,"ps1"],[2,"ps2"],[3,"ps3"],[4,"ps4"],[5,"ps5"]];
  for(const [i, id] of steps){
    const li = document.getElementById(id);
    li.className = planMode > i ? "done" : (planMode === i ? "now" : "");
  }
  document.getElementById("b-seek-done").style.display =
    seek && planMode === 3 ? "inline-block" : "none";
  document.getElementById("b-seek-via-done").style.display =
    seek && planMode === 4 ? "inline-block" : "none";

  const rows = [];
  const add = (k,v)=>rows.push("<b>"+k+"</b><span>"+v+"</span>");
  if(seek){
    const picked = bandRect();
    let n = planWant.size;
    if(picked)
      n = new Set([...planWant.keys(),
        ...(function*(){ for(let x=picked.x0;x<=picked.x1;x++)
                           for(let y=picked.y0;y<=picked.y1;y++)
                             yield x+","+y; })()]).size;
    add("wanted", n + " tile" + (n === 1 ? "" : "s"));
    if(picked)
      add("adding", rectDims(picked).w + " \u00d7 " + rectDims(picked).h);
  }
  const rect = seek ? null : (bandRect() || planRect);
  if(rect){
    const d = rectDims(rect);
    add("grid", d.w+" \u00d7 "+d.h+" = "+d.n+" tiles");
    add("size", d.kx.toFixed(1)+" \u00d7 "+d.ky.toFixed(1)+" km");
    const kmTile = d.kx / d.w;
    add("ideal floor", (d.n * kmTile * Math.SQRT2 / 3).toFixed(1) + " km");
    // counting every tile on each mouse move would stutter on a huge drag,
    // so the census waits for the release once the rectangle gets big
    if(!band || d.n <= 20000){
      const st = zoneStats(rect.x0, rect.y0, rect.x1, rect.y1);
      add("green", st.g);
      add("red", st.r + (st.o ? " (+"+st.o+" out)" : ""));
      if(st.r || st.o)
        add("note", "<span style='color:#f0b429'>red tiles cannot be ridden</span>");
    } else {
      add("green", "\u2026");
      add("red", "\u2026 on release");
    }
  }
  const ptRow = (lab, pt) => {
    if(!pt) return;
    add(lab, pt.lat.toFixed(5) + "," + pt.lon.toFixed(5));
    const out = rect && (pt.tx < rect.x0 || pt.tx > rect.x1
                      || pt.ty < rect.y0 || pt.ty > rect.y1);
    add("", "tile " + pt.tx + "," + pt.ty
         + (out ? " <span style='color:#f0b429'>outside the square</span>" : ""));
  };
  ptRow("start", planStart);
  ptRow("finish", planEnd);
  if(planStart || planEnd)
    add("note", "pinned to the nearest junction; the command reports how far");
  document.getElementById("plan-rows").innerHTML = rows.join("");

  const nameBox = document.getElementById("plan-name");
  const cmdBox = document.getElementById("plan-cmd");
  document.getElementById("b-via-done").style.display =
    planMode === 4 ? "inline-block" : "none";
  const viaBox = document.getElementById("via-list");
  if(planVia.length && planMode >= 4){
    viaBox.style.display = "grid";
    viaBox.innerHTML = planVia.map((v, i) =>
      "<b>via " + (i+1) + "</b><span>" + v.lat.toFixed(5) + ","
      + v.lon.toFixed(5)
      + " <a href='#' data-via='" + i + "' style='color:var(--red);"
      + "text-decoration:none'>&times;</a></span>").join("");
    for(const a of viaBox.querySelectorAll("a[data-via]"))
      a.onclick = ev => { ev.preventDefault();
        planVia.splice(+a.dataset.via, 1); renderPlan(); requestDraw(); };
  } else {
    viaBox.style.display = "none"; viaBox.innerHTML = "";
  }
  nameBox.style.display = planMode === lastStep() ? "block" : "none";
  if(planMode === lastStep()){
    const c = D.cmd || {};
    const field = document.getElementById("run-name");
    if(!field.value && !field.dataset.touched)
      field.value = seek
        ? "seek_" + (planWant.values().next().value || [0,0]).join("_")
        : "z" + planRect.x0 + "_" + planRect.y0;
    const name = safeRunName(field.value);
    field.classList.toggle("bad", !name);
    document.getElementById("run-path").textContent = name
      ? "Runs/" + (seek ? "TileSeeker" : "Square") + "/" + name + "/" + name
        + ".{route.json, gpx, html}"
      : "letters, digits, dash or underscore";
    if(!name){ cmdBox.style.display = "none"; return; }
    const py = c.py || "python";
    const dem = c.dem || "DEM_DB";
    const corners = planRect ? planRect.x0 + "," + planRect.y0
                  + " " + planRect.x1 + "," + planRect.y1 : "";
    const ends =
      " --start-at " + planStart.lat.toFixed(6) + "," + planStart.lon.toFixed(6)
      + " --end-at " + planEnd.lat.toFixed(6) + "," + planEnd.lon.toFixed(6);
    const cmd =
      py + " tilehunt.py "
      + (seek ? "compTileSeekerRoute" : "compSquareRoute")
      + " --db " + (c.db || "Database_OSM")
      + (seek ? "" : " --corners " + corners)
      + ends
      + (seek
         ? [...planWant.values()].map(t => " --want " + t[0] + "," + t[1]).join("")
         : "")
      + planVia.map(v => " --via " + v.lat.toFixed(6) + ","
                       + v.lon.toFixed(6)).join("")
      + (c.args ? " " + c.args : "")
      + " --runname " + name
      + "\n\n"
      + py + " tilehunt.py buildmap --tiles " + (c.tiles || "france.paved.npz")
      + " --clip-to-mask --basemap esri-gray --runname " + name;
    document.getElementById("plan-text").textContent = cmd;
    cmdBox.style.display = "block";
  } else {
    cmdBox.style.display = "none";
  }
}

(function(){
  const cv2 = document.getElementById("prof");
  if(!cv2) return;
  cv2.addEventListener("pointermove", e=>{
    const r = cv2.getBoundingClientRect();
    profMark = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    drawProfile(); requestDraw();
  });
  cv2.addEventListener("pointerleave", ()=>{
    profMark = null; drawProfile(); requestDraw();
  });
})();

/* ---------- buttons ----------------------------------------------------- */
function tog(id, get, set){
  const b = document.getElementById(id);
  b.onclick = ()=>{ set(!get()); b.classList.toggle("on", get()); requestDraw(); };
}
tog("b-tiles", ()=>showTiles, v=>showTiles=v);
tog("b-grid",  ()=>showGrid,  v=>showGrid=v);
tog("b-base",  ()=>showBase,  v=>showBase=v);
(function(){
  const f = document.getElementById("run-name");
  f.addEventListener("input", ()=>{ f.dataset.touched = "1"; renderPlan(); });
  // typing a name must not pan the map or toggle a layer
  for(const ev of ["keydown","keyup","keypress"])
    f.addEventListener(ev, e=>e.stopPropagation());
})();
tog("b-visited",()=>showVisited,v=>{showVisited=v; paintGrid();});
tog("b-cities",()=>showCities,v=>showCities=v);
tog("b-route", ()=>showRoute, v=>showRoute=v);
tog("b-reg",  ()=>showReg, v=>showReg=v);
tog("b-dep",  ()=>showDep, v=>showDep=v);
document.getElementById("b-mode").onclick = function(){
  mode = (mode === "density") ? "coverage" : "density";
  this.classList.toggle("on", mode === "density");
  this.textContent = mode === "density" ? "coverage" : "density";
  paintGrid(); renderLegend(); updateStats(); requestDraw();
};
document.getElementById("b-red").onclick = function(){
  showRed = !showRed; this.classList.toggle("on", showRed); paintGrid(); requestDraw();
};
document.getElementById("b-green").onclick = function(){
  showGreen = !showGreen; this.classList.toggle("on", showGreen);
  paintGrid(); requestDraw();
};
function clearPlan(){
  planRect = planStart = planEnd = null;
  planVia = []; planWant = new Map(); band = null;
}
function setPlanKind(kind){
  // the two planners are exclusive: switching starts the other one cleanly
  const same = planMode && planKind === kind;
  planKind = kind;
  planMode = same ? 0 : 1;
  clearPlan();
  document.getElementById("b-plan").classList.toggle(
    "on", planMode !== 0 && kind === "square");
  document.getElementById("b-seek").classList.toggle(
    "on", planMode !== 0 && kind === "seeker");
  renderPlan(); requestDraw();
}
document.getElementById("b-plan").onclick = () => setPlanKind("square");
document.getElementById("b-seek").onclick = () => setPlanKind("seeker");
document.getElementById("b-seek-done").onclick = e => {
  e.stopPropagation();
  if(!planWant.size) return;          // nothing to look for yet
  planMode = 4; renderPlan(); requestDraw();
};
document.getElementById("b-seek-via-done").onclick = e => {
  e.stopPropagation();
  planMode = 5; renderPlan(); requestDraw();
};
document.getElementById("qs4").onclick = e => {
  if(e.target.id === "b-seek-via-done") return;
  if(planKind === "seeker" && planMode >= 4){
    planMode = 4; renderPlan(); requestDraw();
  }
};
// re-enterable, like the waypoint step: you will want one more tile
document.getElementById("qs3").onclick = e => {
  if(e.target.id === "b-seek-done") return;
  if(planKind === "seeker" && planMode >= 3){
    planMode = 3; renderPlan(); requestDraw();
  }
};
document.getElementById("b-via-done").onclick = e => {
  e.stopPropagation();
  planMode = 5; renderPlan(); requestDraw();
};
// the step is re-enterable: you will want one more after looking at the map
document.getElementById("ps4").onclick = e => {
  if(e.target.id === "b-via-done") return;
  if(planMode >= 4){ planMode = 4; renderPlan(); requestDraw(); }
};
document.getElementById("b-plan-reset").onclick = ()=>{
  planMode = 1; clearPlan();
  renderPlan(); requestDraw();
};
document.getElementById("b-copy").onclick = function(){
  const txt = document.getElementById("plan-text").textContent;
  try {
    navigator.clipboard.writeText(txt).then(
      ()=>{ this.textContent = "copied"; setTimeout(()=>this.textContent="copy", 1200); },
      ()=>{ this.textContent = "select it by hand"; });
  } catch(err) { this.textContent = "select it by hand"; }
};
window.addEventListener("keydown", e=>{
  if(e.key === "Escape" && planMode){
    planMode = 0; clearPlan();
    document.getElementById("b-plan").classList.remove("on");
    document.getElementById("b-seek").classList.remove("on");
    renderPlan(); requestDraw();
  }
});
document.getElementById("b-fit-route").onclick = ()=>{
  if(!ROUTE) return;
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  for(const q of ROUTE.pts){
    if(q[0]<x0)x0=q[0]; if(q[0]>x1)x1=q[0];
    if(q[1]<y0)y0=q[1]; if(q[1]>y1)y1=q[1];
  }
  if(ROUTE.rect && ROUTE.rect.length === 4){
    x0=Math.min(x0,ROUTE.rect[0]); y0=Math.min(y0,ROUTE.rect[1]);
    x1=Math.max(x1,ROUTE.rect[2]+1); y1=Math.max(y1,ROUTE.rect[3]+1);
  }
  const m = 1.25;
  view.scale = Math.min(VW/((x1-x0)*m||1), VH/((y1-y0)*m||1));
  view.cx = (x0+x1)/2; view.cy = (y0+y1)/2;
  requestDraw();
};
document.getElementById("b-reset").onclick = ()=>{
  fitAll(); pick=[]; best=null;
  document.getElementById("zone").style.display="none"; requestDraw();
};
document.getElementById("b-sq").onclick = ()=>{
  const sq = maxSquareIn(D.x0, D.y0, D.x0+W-1, D.y0+H-1);
  best = sq;
  if(sq){
    view.cx = sq.x + sq.n/2; view.cy = sq.y + sq.n/2;
    view.scale = Math.max(2, Math.min(VW,VH)*0.7/sq.n);
    pick = [[sq.x,sq.y],[sq.x+sq.n-1, sq.y+sq.n-1]];
    showZone();
  }
  requestDraw();
};

/* ---------- boot -------------------------------------------------------- */
if(!ROUTE) document.getElementById("b-route").style.display="none";
if(!REGIONS) document.getElementById("b-reg").style.display="none";
if(!DEPTS) document.getElementById("b-dep").style.display="none";
document.getElementById("ttl").textContent = D.title;
const mt = D.meta || {};
document.getElementById("sub").textContent =
  "zoom "+Z+" tiles · "+W+"×"+H+" window · "
  + (D.stats.inscope||0).toLocaleString() + " in scope"
  + (mt.built ? " · " + String(mt.built).slice(0,10) : "");
function setStat(id, label, value){
  document.getElementById("lab-" + id).textContent = label;
  document.getElementById("s-" + id).textContent = value;
}
let flashTile = null, flashUntil = 0, flashTimer = null;
function gotoTile(tx, ty, z){
  // centre on a tile and pulse it, so "the busiest tile" is somewhere you can
  // actually look at rather than a number in a panel
  view.scale = z || Math.max(view.scale, 60);
  view.cx = tx + 0.5; view.cy = ty + 0.5;
  flashTile = [tx, ty]; flashUntil = performance.now() + 4000;
  if(flashTimer) clearInterval(flashTimer);
  flashTimer = setInterval(() => {
    if(performance.now() > flashUntil){
      clearInterval(flashTimer); flashTimer = null; flashTile = null;
    }
    requestDraw();
  }, 60);
  requestDraw();
}

function updateStats(){
  const k = D.kmStats;
  if(mode === "density" && k){
    setStat("green", "total",   fmtKm(k.total) + " km");
    setStat("red",   "0 km",    D.stats.red.toLocaleString());
    setStat("pct",   "median",  fmtKm(k.p50) + " km");
    setStat("out",   "busiest", fmtKm(k.max) + " km");
    const cell = document.getElementById("s-out");
    if(k.top && k.top.length){
      cell.style.cursor = "pointer";
      cell.style.textDecoration = "underline dotted";
      cell.title = "go to " + k.top[0][0] + "," + k.top[0][1];
      cell.onclick = () => gotoTile(k.top[0][0], k.top[0][1]);
    }
  } else {
    const cell = document.getElementById("s-out");
    cell.style.cursor = ""; cell.style.textDecoration = ""; cell.onclick = null;
    cell.title = "";
    if(D.visitedStats){
      const v = D.visitedStats;
      document.getElementById("vis-kv").style.display = "grid";
      document.getElementById("s-have").textContent = v.have.toLocaleString();
      document.getElementById("s-need").textContent = v.need.toLocaleString();
      const sq = v.square;
      const cell = document.getElementById("s-sq");
      cell.textContent = sq[0] + " \u00d7 " + sq[0];
      if(sq[0]){
        cell.style.cursor = "pointer";
        cell.style.textDecoration = "underline dotted";
        cell.title = "go to " + sq[1] + "," + sq[2];
        cell.onclick = () => gotoTile(sq[1] + (sq[0]>>1), sq[2] + (sq[0]>>1),
                                      Math.max(6, 420 / sq[0]));
      }
    } else {
      document.getElementById("vis-kv").style.display = "none";
    }
    setStat("green", "green",   D.stats.green.toLocaleString());
    setStat("red",   "red",     D.stats.red.toLocaleString());
    setStat("pct",   "covered", D.stats.pct + "%");
    setStat("out",   "out",     (W*H - D.stats.inscope).toLocaleString());
  }
}
if(!KMQ) document.getElementById("b-mode").style.display = "none";
if(!ROUTE) document.getElementById("b-fit-route").style.display = "none";
else {
  const st = ROUTE.stats, rows = [];
  const add = (k,v)=>rows.push("<b>"+k+"</b><span>"+v+"</span>");
  if(ROUTE.wanted && ROUTE.wanted.length)
    add("wanted", ROUTE.wanted.length + " tile(s) sought");
  if(ROUTE.skipped && ROUTE.skipped.length)
    add("skipped", ROUTE.skipped.length + " tile(s) with no usable road");
  if(ROUTE.pinned){
    for(const [k, lab] of [["start","start pinned"],["end","finish pinned"]]){
      const q = ROUTE.pinned[k];
      if(q) add(lab, q.m + " m from your point");
    }
    const vs = ROUTE.pinned.via || [];
    if(vs.length)
      add("waypoints", vs.length + " forced, up to "
          + Math.max.apply(null, vs.map(v => v.m)) + " m off");
  }
  if(ROUTE.rect && ROUTE.rect.length === 4){
    const [x0,y0,x1,y1] = ROUTE.rect;
    add("square", (x1-x0+1) + " \u00d7 " + (y1-y0+1));
  }
  if(st.km !== undefined) add("distance", st.km + " km");
  if(st.theory !== undefined)
    add("ideal floor", st.theory + " km"
        + (st.pct ? "  \u00b7  " + st.pct.toFixed(0) + "%" : ""));
  if(st.kmPerTile !== undefined) add("per tile", st.kmPerTile.toFixed(2) + " km");
  if(st.up !== undefined && st.up !== null)
    add("climb", "+" + st.up.toLocaleString() + " / \u2212"
        + (st.down||0).toLocaleString() + " m");
  if(st.tiles !== undefined) add("tiles", (st.covered||0) + " / " + st.tiles);
  if(st.extra) add("extra swept", st.extra);
  if(st.visits !== undefined) add("visits", st.visits);
  document.getElementById("route-rows").innerHTML = rows.join("");
  document.getElementById("routebox").style.display = "block";
}
updateStats();
if(D.note) document.getElementById("note").textContent = D.note;
else document.getElementById("note").style.display = "none";
let attr = BM ? BM.attr
  : 'tile data derived from <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';
if(ROUTE && ROUTE.profile && ROUTE.profile.src){
  const src = ROUTE.profile.src;
  if(src.indexOf("GeoTIFF") >= 0)
    attr += ' &middot; heights from the <a href="https://spacedata.copernicus.eu/'
         +  'collections/copernicus-digital-elevation-model">Copernicus DEM</a> '
         +  '(WorldDEM-30 &copy; DLR e.V. and &copy; Airbus Defence and Space '
         +  'GmbH, provided under COPERNICUS by the European Union and ESA)';
  else if(src.length)
    attr += ' &middot; heights from NASA/USGS SRTM';
}
document.getElementById("attr").innerHTML = attr;

if(ROUTE && ROUTE.profile){
  const pr = ROUTE.profile;
  document.getElementById("pf-up").textContent = "+" + pr.up.toLocaleString() + " m";
  document.getElementById("pf-down").textContent = "\u2212" + pr.down.toLocaleString() + " m";
  document.getElementById("scale").style.bottom = "136px";
  // with a profile along the bottom, the right column stops above it
  document.getElementById("rightstack").style.maxHeight =
    "calc(100% - 160px)";
}

window.addEventListener("resize", ()=>{ resize(); drawProfile(); });
resize(); fitAll(); ready = true; draw(); drawProfile();

/* a small console API, handy for poking at the dataset by hand */
window.tilehunt = {
  data: D,
  get view(){ return {cx:view.cx, cy:view.cy, scale:view.scale, VW:VW, VH:VH}; },
  get selection(){ return pick.slice(); },
  status: status,
  tileAt: (lon,lat)=>{ const t = ll2t(lon,lat); return [Math.floor(t[0]), Math.floor(t[1])]; },
  centreOf: (x,y)=> t2ll(x+0.5, y+0.5),
  goto: (x,y,s)=>{ view.cx=x+0.5; view.cy=y+0.5; if(s) view.scale=s; requestDraw(); },
  maxSquare: ()=> maxSquareIn(D.x0, D.y0, D.x0+W-1, D.y0+H-1),
  zoneStats: zoneStats
};
