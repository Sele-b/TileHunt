// Drives the planner's own state machine: clicks go through the page's real
// pointer handlers, so what is checked is the panel a person would use.
import fs from "node:fs";
import { createRequire } from "node:module";

const harness = fs.readFileSync(
  new URL("./test_draw.mjs", import.meta.url), "utf8");
// reuse the DOM stub by running the harness up to the point it returns the api
const prelude = harness.slice(0, harness.indexOf("const { ROUTE, draw"))
  .replace(/^import fs from "node:fs";$/m, "")
  .replace('const file = process.argv[2];', 'const file = process.argv[2];')
  .replace(/console\.log\("ROUTE present[\s\S]*$/, "");
const run = new Function("fs", "process", prelude + "\nreturn {api, document, nodes, ops};");
const { api, document: doc, nodes } = run(fs, process);

const cv = nodes.get("map");
const fails = [];
const plan = () => api.plan();

function click(tileX, tileY) {
  // screen position of a tile centre, through the page's own projection
  const x = api.sx(tileX + 0.5), y = api.sy(tileY + 0.5);
  const ev = { clientX: x, clientY: y, pointerId: 1, button: 0,
               ctrlKey: false, metaKey: false, preventDefault() {},
               stopPropagation() {} };
  cv.fire("pointerdown", ev);
  cv.fire("pointerup", ev);
}
function drag(x0, y0, x1, y1) {
  const a = { clientX: api.sx(x0 + 0.5), clientY: api.sy(y0 + 0.5),
              pointerId: 1, ctrlKey: true, metaKey: false,
              preventDefault() {}, stopPropagation() {} };
  const b = { ...a, clientX: api.sx(x1 + 0.5), clientY: api.sy(y1 + 0.5) };
  cv.fire("pointerdown", a);
  cv.fire("pointermove", b);
  cv.fire("pointerup", b);
}

// park the view somewhere with room, at a readable scale
api.view.cx = 8128; api.view.cy = 5658; api.view.scale = 30;

doc.getElementById("b-seek").onclick();
console.log("after TileSeeker :", plan().kind, "step", plan().mode);
if (plan().kind !== "seeker" || plan().mode !== 1) fails.push("enter");

click(8122, 5652);                       // start
click(8135, 5665);                       // end
console.log("after two points :", "step", plan().mode);
if (plan().mode !== 3) fails.push("points");

click(8124, 5654);
click(8131, 5657);
click(8127, 5663);
console.log("after three picks:", plan().want.length, "tiles");
click(8131, 5657);                       // toggles that one back off
console.log("after re-click   :", plan().want.length, "tiles (toggled off)");
if (plan().want.length !== 2) fails.push("toggle");

drag(8140, 5660, 8142, 5661);            // ctrl-drag adds a block of six
console.log("after ctrl-drag  :", plan().want.length, "tiles");
if (plan().want.length !== 8) fails.push("drag");

doc.getElementById("b-seek-done").onclick({ stopPropagation() {},
                                            target: { id: "b-seek-done" } });
console.log("after done       :", "step", plan().mode, "(waypoints)");
if (plan().mode !== 4) fails.push("done");

click(8133, 5652);                       // a waypoint, on the seeker too
click(8129, 5659);
doc.getElementById("b-seek-via-done").onclick(
  { stopPropagation() {}, target: { id: "b-seek-via-done" } });
console.log("after via done   :", "step", plan().mode,
            "with", plan().via.length, "waypoint(s)");
if (plan().mode !== 5 || plan().via.length !== 2) fails.push("seek-via");

const cmd = plan().cmd;
console.log("\ncommand emitted:\n" + cmd.split("\n").map(l => "  " + l).join("\n"));
const want = plan().want.map(t => `--want ${t[0]},${t[1]}`);
const checks = {
  "uses compTileSeekerRoute": cmd.includes("compTileSeekerRoute"),
  "no --corners": !cmd.includes("--corners"),
  "has --start-at and --end-at":
    cmd.includes("--start-at") && cmd.includes("--end-at"),
  "one --want per picked tile":
    want.every(w => cmd.includes(w))
    && (cmd.match(/--want /g) || []).length === want.length,
  "names the run": /--runname \S+/.test(cmd),
  "buildmap follows": cmd.includes("buildmap"),
  "waypoints too": (cmd.match(/--via /g) || []).length === 2,
  "no comment lines": !cmd.split("\n").some(l => l.trim().startsWith("#")),
  "exactly two commands": cmd.split("\n").filter(l => l.trim()).length === 2,
};
for (const [k, v] of Object.entries(checks)) {
  console.log(`  ${k}: ${v ? "yes" : "NO"}`);
  if (!v) fails.push(k);
}

// and the square planner still works, from the same page
doc.getElementById("b-plan").onclick();
console.log("\nafter SquarePlanner:", plan().kind, "step", plan().mode);
drag(8124, 5654, 8127, 5657);
click(8124, 5654);                       // start point
click(8127, 5657);                       // end point
console.log("square reaches the waypoint step:", plan().mode === 4);
if (plan().mode !== 4) fails.push("square-steps");
doc.getElementById("b-via-done").onclick({ stopPropagation() {},
                                           target: { id: "b-via-done" } });
console.log("square reaches the name step    :", plan().mode === 5);
const sqcmd = plan().cmd;
console.log("square command uses compSquareRoute:",
            sqcmd.includes("compSquareRoute"), "| has --corners:",
            sqcmd.includes("--corners 8124,5654 8127,5657"));
if (!sqcmd.includes("compSquareRoute")
    || !sqcmd.includes("--corners 8124,5654 8127,5657")) fails.push("square");

console.log(fails.length ? "\nFAILURES: " + fails.join(", ") : "\nplanner OK");
process.exit(fails.length ? 1 : 0);
