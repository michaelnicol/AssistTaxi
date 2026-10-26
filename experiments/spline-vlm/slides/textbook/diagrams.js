'use strict';
(() => {
const NS = 'http://www.w3.org/2000/svg';
const C = {
  ink: '#e8e6e3', dim: '#8d8d97', faint: '#2c2c36', line: '#44444f',
  blue: '#58c4dd', yellow: '#ffd75e', green: '#83c167', red: '#fc6255',
  purple: '#c39ae6', orange: '#f0a35e', teal: '#5cd0b3',
};

const S = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
};
const H = (tag, attrs = {}, parent, text) => {
  const e = document.createElement(tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const sig = z => 1 / (1 + Math.exp(-z));
const softplus = z => (z > 30 ? z : Math.log1p(Math.exp(z)));
const relu = z => Math.max(0, z);
const ACT = { softplus, relu, sigmoid: sig };
const DACT = { softplus: sig, relu: z => (z > 0 ? 1 : 0), sigmoid: z => sig(z) * (1 - sig(z)) };
const fmt = (v, d = 2) => {
  const s = v.toFixed(d);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
};
const comma = n => Math.round(n).toLocaleString('en-US');
const softmax = zs => {
  const m = Math.max(...zs);
  const e = zs.map(z => Math.exp(z - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
};
const rng = seed => {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
};
const gauss = r => Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r());
const lerp = (a, b, t) => a + (b - a) * t;
const mix = (c1, c2, t) => {
  const p = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
  const a = p(c1), b = p(c2);
  return `rgb(${a.map((v, i) => Math.round(lerp(v, b[i], clamp(t, 0, 1)))).join(',')})`;
};
const heat = (v, scale = 1) => {
  const t = clamp(Math.abs(v) / scale, 0, 1);
  return v >= 0 ? mix('#15151b', C.blue, t) : mix('#15151b', C.red, t);
};

const svgBox = (el, w, h) => S('svg', { viewBox: `0 0 ${w} ${h}`, class: 'fig-svg', role: 'img' }, el);
const controls = el => H('div', { class: 'controls' }, el);
const readout = el => H('div', { class: 'readout' }, el);
function slider(box, label, min, max, step, value, onInput, digits = 2) {
  const wrap = H('label', { class: 'slider' }, box);
  H('span', { class: 'sl-name' }, wrap, label);
  const input = H('input', { type: 'range', min, max, step, value }, wrap);
  const out = H('output', {}, wrap, fmt(value, digits));
  input.addEventListener('input', () => { out.textContent = fmt(+input.value, digits); onInput(+input.value); });
  return {
    get value() { return +input.value; },
    set(v) { input.value = v; out.textContent = fmt(+input.value, digits); },
  };
}
function button(box, label, onClick, cls = '') {
  const b = H('button', { type: 'button', class: cls }, box, label);
  b.addEventListener('click', onClick);
  return b;
}
function select(box, label, options, value, onChange) {
  const wrap = H('label', { class: 'slider' }, box);
  H('span', { class: 'sl-name' }, wrap, label);
  const s = H('select', {}, wrap);
  for (const [v, t] of options) {
    const o = H('option', { value: v }, s, t);
    if (v === value) o.selected = true;
  }
  s.addEventListener('change', () => onChange(s.value));
  return s;
}
function segmented(box, options, value, onChange) {
  const wrap = H('div', { class: 'seg' }, box);
  const btns = options.map(([v, t]) => {
    const b = button(wrap, t, () => { pick(v); onChange(v); });
    b.dataset.v = v;
    return b;
  });
  const pick = v => btns.forEach(b => b.classList.toggle('on', b.dataset.v === String(v)));
  pick(value);
  return { pick };
}
function frame(svg, x, y, w, h, xr, yr) {
  const X = v => x + ((v - xr[0]) / (xr[1] - xr[0])) * w;
  const Y = v => y + h - ((v - yr[0]) / (yr[1] - yr[0])) * h;
  const iX = px => xr[0] + ((px - x) / w) * (xr[1] - xr[0]);
  const iY = py => yr[0] + ((y + h - py) / h) * (yr[1] - yr[0]);
  return { X, Y, iX, iY, g: S('g', {}, svg), x, y, w, h, xr, yr };
}
function text(parent, x, y, str, attrs = {}) {
  const t = S('text', { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}
function axes(F, { xlabel = '', ylabel = '', xticks = [], yticks = [], zero = true } = {}) {
  const g = S('g', { class: 'axes' }, F.g);
  const y0 = zero && F.yr[0] < 0 && F.yr[1] > 0 ? F.Y(0) : F.y + F.h;
  S('line', { x1: F.x, y1: y0, x2: F.x + F.w, y2: y0, class: 'ax' }, g);
  S('line', { x1: F.x, y1: F.y, x2: F.x, y2: F.y + F.h, class: 'ax' }, g);
  for (const t of xticks) {
    S('line', { x1: F.X(t), y1: y0, x2: F.X(t), y2: y0 + 4, class: 'ax' }, g);
    text(g, F.X(t), y0 + 16, String(t), { class: 'tk', 'text-anchor': 'middle' });
  }
  for (const t of yticks) {
    S('line', { x1: F.x - 4, y1: F.Y(t), x2: F.x, y2: F.Y(t), class: 'ax' }, g);
    text(g, F.x - 7, F.Y(t) + 4, String(t), { class: 'tk', 'text-anchor': 'end' });
  }
  if (xlabel) text(g, F.x + F.w, y0 - 6, xlabel, { class: 'axl', 'text-anchor': 'end' });
  if (ylabel) text(g, F.x + 6, F.y + 4, ylabel, { class: 'axl' });
  return g;
}
const pathD = pts => pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('');
function fnPath(F, fn, n = 160) {
  const pts = [];
  const pad = (F.yr[1] - F.yr[0]) * 0.03;
  for (let i = 0; i <= n; i++) {
    const x = lerp(F.xr[0], F.xr[1], i / n);
    pts.push([F.X(x), F.Y(clamp(fn(x), F.yr[0] - pad, F.yr[1] + pad))]);
  }
  return pathD(pts);
}
function curve(F, fn, attrs = {}) {
  const p = S('path', { fill: 'none', 'stroke-width': 2.5, 'stroke-linejoin': 'round', ...attrs }, F.g);
  const set = f => p.setAttribute('d', fnPath(F, f));
  if (fn) set(fn);
  return { el: p, set };
}
function makeArrow(g, color, width = 2.5) {
  const grp = S('g', {}, g);
  const ln = S('line', { stroke: color, 'stroke-width': width, 'stroke-linecap': 'round' }, grp);
  const hd = S('polygon', { fill: color }, grp);
  return {
    el: grp,
    set(x1, y1, x2, y2) {
      const a = Math.atan2(y2 - y1, x2 - x1), L = Math.hypot(x2 - x1, y2 - y1), hl = Math.min(11, L * 0.5);
      const bx = x2 - hl * Math.cos(a), by = y2 - hl * Math.sin(a);
      ln.setAttribute('x1', x1); ln.setAttribute('y1', y1); ln.setAttribute('x2', bx); ln.setAttribute('y2', by);
      const w = hl * 0.5;
      hd.setAttribute('points', `${x2},${y2} ${bx + w * Math.sin(a)},${by - w * Math.cos(a)} ${bx - w * Math.sin(a)},${by + w * Math.cos(a)}`);
    },
  };
}
const svgPoint = (svg, e) => {
  const p = svg.createSVGPoint();
  p.x = e.clientX; p.y = e.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
};
function dragArea(svg, onMove) {
  svg.style.touchAction = 'none';
  svg.addEventListener('pointerdown', e => {
    svg.setPointerCapture(e.pointerId);
    onMove(svgPoint(svg, e));
    const move = ev => onMove(svgPoint(svg, ev));
    const up = () => { svg.removeEventListener('pointermove', move); svg.removeEventListener('pointerup', up); };
    svg.addEventListener('pointermove', move);
    svg.addEventListener('pointerup', up);
  });
}
function loop(step) {
  let id = 0;
  const tick = () => { if (step() === false) { id = 0; return; } id = requestAnimationFrame(tick); };
  return {
    start() { if (!id) id = requestAnimationFrame(tick); },
    stop() { cancelAnimationFrame(id); id = 0; },
    get running() { return id !== 0; },
  };
}
function stepper(box, steps, onStep) {
  const row = H('div', { class: 'controls' }, box);
  let i = 0;
  const prev = button(row, '← back', () => go(i - 1));
  const next = button(row, 'next →', () => go(i + 1), 'primary');
  const count = H('span', { class: 'count' }, row);
  const go = k => {
    i = clamp(k, 0, steps - 1);
    prev.disabled = i === 0; next.disabled = i === steps - 1;
    count.textContent = `step ${i + 1} of ${steps}`;
    onStep(i);
  };
  go(0);
  return { go };
}

const SQ = { w1: 3.34, b1: -1.43, w2: -3.53, b2: 0.57, w3: -1.22, w4: -2.30, b3: 2.61 };
const HAND_RELU = { w1: 2, b1: 0, w2: 4, b2: -2, w3: 1, w4: -1, b3: 0 };
const DOSES = [[0, 0], [0.5, 1], [1, 0]];
const net = (p, act = 'softplus') => x => p.w3 * ACT[act](p.w1 * x + p.b1) + p.w4 * ACT[act](p.w2 * x + p.b2) + p.b3;
const PKEYS = ['w1', 'b1', 'w2', 'b2', 'w3', 'w4', 'b3'];

const FIGS = {};

FIGS.bend = body => {
  const svg = svgBox(body, 660, 290);
  const F = frame(svg, 50, 20, 570, 230, [0, 1], [-0.3, 1.3]);
  axes(F, { xlabel: 'dose', ylabel: 'effect', xticks: [0, 0.5, 1], yticks: [0, 1] });
  const line = curve(F, () => 1 / 3, { stroke: C.red });
  const squig = curve(F, net(SQ), { stroke: C.yellow, 'stroke-width': 3 });
  DOSES.forEach(([x, y]) => S('circle', { cx: F.X(x), cy: F.Y(y), r: 7, fill: C.green }, F.g));
  const ro = readout(body);
  const ctl = controls(body);
  const show = v => {
    line.el.style.display = v === 'line' ? '' : 'none';
    squig.el.style.display = v === 'net' ? '' : 'none';
    ro.innerHTML = v === 'line'
      ? 'The best <span class="c-red">straight line</span> sits at one third everywhere. It is wrong at every dose.'
      : 'A small network bends its output into a <span class="c-yellow">squiggle</span> that passes through all three points.';
  };
  segmented(ctl, [['line', 'best straight line'], ['net', 'small neural network']], 'line', show);
  show('line');
};

FIGS.neuron = body => {
  const svg = svgBox(body, 680, 290);
  const st = { w: 3.34, b: -1.43, x: 0.5, act: 'softplus' };
  const g = S('g', {}, svg);
  const nodes = { x: [40, 140], z: [150, 140], y: [270, 140] };
  S('line', { x1: 52, y1: 140, x2: 110, y2: 140, stroke: C.blue, 'stroke-width': 3 }, g);
  S('line', { x1: 190, y1: 140, x2: 232, y2: 140, stroke: C.line, 'stroke-width': 3 }, g);
  S('circle', { cx: nodes.x[0], cy: 140, r: 12, fill: '#1b1b22', stroke: C.ink, 'stroke-width': 2 }, g);
  S('rect', { x: 110, y: 110, width: 80, height: 60, rx: 10, fill: '#1b1b22', stroke: C.blue, 'stroke-width': 2 }, g);
  S('rect', { x: 232, y: 110, width: 76, height: 60, rx: 30, fill: '#1b1b22', stroke: C.yellow, 'stroke-width': 2 }, g);
  text(g, 150, 135, '× w', { class: 'lbl', 'text-anchor': 'middle' });
  text(g, 150, 155, '+ b', { class: 'lbl', 'text-anchor': 'middle' });
  const bendIcon = S('path', { fill: 'none', stroke: C.yellow, 'stroke-width': 2 }, g);
  const tx = text(g, 40, 100, '', { class: 'val', 'text-anchor': 'middle' });
  const tz = text(g, 150, 100, '', { class: 'val', 'text-anchor': 'middle' });
  const ty = text(g, 270, 100, '', { class: 'val', 'text-anchor': 'middle' });
  text(g, 40, 190, 'input', { class: 'tk', 'text-anchor': 'middle' });
  text(g, 150, 190, 'weigh and shift', { class: 'tk', 'text-anchor': 'middle' });
  text(g, 270, 190, 'bend', { class: 'tk', 'text-anchor': 'middle' });
  const F = frame(svg, 360, 20, 290, 230, [0, 1], [-4, 6]);
  axes(F, { xlabel: 'input', xticks: [0, 0.5, 1], yticks: [-4, 0, 3, 6] });
  const raw = curve(F, null, { stroke: C.blue, 'stroke-dasharray': '5 5', 'stroke-width': 2 });
  const out = curve(F, null, { stroke: C.yellow, 'stroke-width': 3 });
  const dz = S('circle', { r: 5, fill: C.blue }, F.g), dy = S('circle', { r: 6, fill: C.yellow }, F.g);
  const ro = readout(body);
  const ctl = controls(body);
  const draw = () => {
    const f = ACT[st.act];
    raw.set(x => st.w * x + st.b);
    out.set(x => f(st.w * x + st.b));
    const z = st.w * st.x + st.b, y = f(z);
    dz.setAttribute('cx', F.X(st.x)); dz.setAttribute('cy', F.Y(clamp(z, -4.2, 6.2)));
    dy.setAttribute('cx', F.X(st.x)); dy.setAttribute('cy', F.Y(clamp(y, -4.2, 6.2)));
    tx.textContent = fmt(st.x); tz.textContent = fmt(z); ty.textContent = fmt(y);
    const pts = [];
    for (let i = 0; i <= 20; i++) { const u = -3 + (6 * i) / 20; pts.push([270 + u * 9, 152 - clamp(f(u), -1, 3) * 7]); }
    bendIcon.setAttribute('d', pathD(pts));
    ro.innerHTML = `${fmt(st.x)} × <span class="c-blue">${fmt(st.w)}</span> + <span class="c-blue">${fmt(st.b)}</span> = ${fmt(z)}, and the bend turns ${fmt(z)} into <span class="c-yellow">${fmt(y)}</span>.`;
  };
  slider(ctl, 'weight w', -6, 6, 0.01, st.w, v => { st.w = v; draw(); });
  slider(ctl, 'bias b', -6, 6, 0.01, st.b, v => { st.b = v; draw(); });
  slider(ctl, 'input', 0, 1, 0.01, st.x, v => { st.x = v; draw(); });
  select(ctl, 'bend', [['softplus', 'soft plus'], ['relu', 'ReLU'], ['sigmoid', 'sigmoid']], st.act, v => { st.act = v; draw(); });
  draw();
};

FIGS.dosage = (body, data) => {
  const act = data.act || 'softplus';
  const start = data.preset === 'relu' ? HAND_RELU : SQ;
  const p = { ...start };
  const svg = svgBox(body, 680, 300);
  const g = S('g', {}, svg);
  const N = { x: [30, 150], h1: [140, 70], h2: [140, 230], y: [250, 150] };
  const edge = (a, b) => S('line', { x1: N[a][0], y1: N[a][1], x2: N[b][0], y2: N[b][1], 'stroke-linecap': 'round' }, g);
  const E = { w1: edge('x', 'h1'), w2: edge('x', 'h2'), w3: edge('h1', 'y'), w4: edge('h2', 'y') };
  for (const [k, c] of [['x', C.ink], ['h1', C.blue], ['h2', C.orange], ['y', C.yellow]])
    S('circle', { cx: N[k][0], cy: N[k][1], r: 16, fill: '#1b1b22', stroke: c, 'stroke-width': 2.5 }, g);
  const lab = {
    w1: text(g, 70, 98, '', { class: 'val' }), w2: text(g, 70, 212, '', { class: 'val' }),
    w3: text(g, 200, 98, '', { class: 'val' }), w4: text(g, 200, 212, '', { class: 'val' }),
    b1: text(g, 140, 40, '', { class: 'val', 'text-anchor': 'middle' }), b2: text(g, 140, 270, '', { class: 'val', 'text-anchor': 'middle' }),
    b3: text(g, 250, 120, '', { class: 'val', 'text-anchor': 'middle' }),
  };
  text(g, 30, 185, 'dose', { class: 'tk', 'text-anchor': 'middle' });
  text(g, 250, 185, 'effect', { class: 'tk', 'text-anchor': 'middle' });
  const F = frame(svg, 330, 20, 320, 240, [0, 1], [-1.5, 2.5]);
  axes(F, { xlabel: 'dose', xticks: [0, 0.5, 1], yticks: [-1, 0, 1, 2] });
  const c1 = curve(F, null, { stroke: C.blue, 'stroke-width': 2 });
  const c2 = curve(F, null, { stroke: C.orange, 'stroke-width': 2 });
  const cy = curve(F, null, { stroke: C.yellow, 'stroke-width': 3.5 });
  DOSES.forEach(([x, y]) => S('circle', { cx: F.X(x), cy: F.Y(y), r: 6, fill: C.green }, F.g));
  const ro = readout(body);
  const ctl = controls(body);
  const sl = {};
  const draw = () => {
    const f = ACT[act];
    c1.set(x => p.w3 * f(p.w1 * x + p.b1));
    c2.set(x => p.w4 * f(p.w2 * x + p.b2));
    cy.set(net(p, act));
    for (const k of ['w1', 'w2', 'w3', 'w4']) {
      E[k].setAttribute('stroke', p[k] >= 0 ? C.blue : C.red);
      E[k].setAttribute('stroke-width', 1 + Math.min(6, Math.abs(p[k]) * 1.4));
    }
    for (const k of PKEYS) lab[k].textContent = `${k} ${fmt(p[k])}`;
    const ssr = DOSES.reduce((s, [x, y]) => s + (y - net(p, act)(x)) ** 2, 0);
    ro.innerHTML = `<span class="c-blue">upper path</span> plus <span class="c-orange">lower path</span> plus b3 gives the <span class="c-yellow">output</span>. Squared misses added up: <b>${fmt(ssr, 3)}</b>`;
  };
  for (const k of PKEYS) sl[k] = slider(ctl, k, -6, 6, 0.01, p[k], v => { p[k] = v; draw(); });
  button(ctl, data.preset === 'relu' ? 'hand-built values' : 'trained values', () => {
    Object.assign(p, start); for (const k of PKEYS) sl[k].set(p[k]); draw();
  });
  draw();
};

FIGS.chain = body => {
  const h = w => 0.5 + 1.5 * w - 0.4 * w * w, dh = w => 1.5 - 0.8 * w;
  const s = v => 0.6 * v * v + 0.2 * v, ds = v => 1.2 * v + 0.2;
  const svg = svgBox(body, 680, 290);
  const F1 = frame(svg, 50, 20, 260, 220, [0, 1.6], [0, 2]);
  const F2 = frame(svg, 400, 20, 260, 220, [0, 2], [0, 3]);
  axes(F1, { xlabel: 'weight', ylabel: 'height', xticks: [0, 0.5, 1, 1.5], yticks: [0, 1, 2] });
  axes(F2, { xlabel: 'height', ylabel: 'shoe size', xticks: [0, 1, 2], yticks: [0, 1, 2, 3] });
  curve(F1, h, { stroke: C.blue });
  curve(F2, s, { stroke: C.purple });
  const t1 = S('line', { stroke: C.yellow, 'stroke-width': 2 }, F1.g), t2 = S('line', { stroke: C.yellow, 'stroke-width': 2 }, F2.g);
  const g1 = S('line', { stroke: C.dim, 'stroke-dasharray': '4 4' }, F1.g), g2 = S('line', { stroke: C.dim, 'stroke-dasharray': '4 4' }, F2.g);
  const d1 = S('circle', { r: 6, fill: C.yellow }, F1.g), d2 = S('circle', { r: 6, fill: C.yellow }, F2.g);
  const hand = S('path', { fill: 'none', stroke: C.dim, 'stroke-dasharray': '3 5' }, svg);
  const ro = readout(body);
  const ctl = controls(body);
  const tangent = (F, ln, x, y, m, span) => {
    ln.setAttribute('x1', F.X(x - span)); ln.setAttribute('y1', F.Y(y - m * span));
    ln.setAttribute('x2', F.X(x + span)); ln.setAttribute('y2', F.Y(y + m * span));
  };
  const draw = w => {
    const hv = h(w), sv = s(hv), a = dh(w), b = ds(hv);
    tangent(F1, t1, w, hv, a, 0.3); tangent(F2, t2, hv, sv, b, 0.3);
    d1.setAttribute('cx', F1.X(w)); d1.setAttribute('cy', F1.Y(hv));
    d2.setAttribute('cx', F2.X(hv)); d2.setAttribute('cy', F2.Y(sv));
    g1.setAttribute('x1', F1.X(w)); g1.setAttribute('y1', F1.Y(hv)); g1.setAttribute('x2', F1.x + F1.w); g1.setAttribute('y2', F1.Y(hv));
    g2.setAttribute('x1', F2.X(hv)); g2.setAttribute('y1', F2.Y(sv)); g2.setAttribute('x2', F2.X(hv)); g2.setAttribute('y2', F2.y + F2.h);
    hand.setAttribute('d', `M${F1.x + F1.w} ${F1.Y(hv)} C ${F1.x + F1.w + 50} ${F1.Y(hv)}, ${F2.X(hv)} ${F2.y + F2.h + 40}, ${F2.X(hv)} ${F2.y + F2.h}`);
    const nudge = s(h(w + 0.01)) - sv;
    ro.innerHTML = `Height rises <span class="c-yellow">${fmt(a)}</span> per unit of weight here. Shoe size rises <span class="c-yellow">${fmt(b)}</span> per unit of height here. So shoe size rises ${fmt(a)} × ${fmt(b)} = <b>${fmt(a * b)}</b> per unit of weight. Check: nudge weight by 0.01 and shoe size moves ${fmt(nudge, 4)}, about ${fmt(a * b * 0.01, 4)}.`;
  };
  slider(ctl, 'weight', 0, 1.5, 0.01, 0.8, draw);
  draw(0.8);
};

const GD_PTS = [[0.5, 1.4], [2.3, 1.9], [2.9, 3.2]];
const SLOPE = 0.64;
FIGS.gd = body => {
  const ssr = b => GD_PTS.reduce((s, [x, y]) => s + (y - (b + SLOPE * x)) ** 2, 0);
  const dssr = b => GD_PTS.reduce((s, [x, y]) => s - 2 * (y - (b + SLOPE * x)), 0);
  const svg = svgBox(body, 680, 290);
  const F1 = frame(svg, 40, 20, 270, 230, [0, 3.5], [0, 4]);
  const F2 = frame(svg, 390, 20, 260, 230, [-1, 3], [0, 20]);
  axes(F1, { xlabel: 'weight', ylabel: 'height', xticks: [0, 1, 2, 3], yticks: [0, 1, 2, 3, 4] });
  axes(F2, { xlabel: 'intercept', ylabel: 'sum of squared misses', xticks: [-1, 0, 1, 2, 3], yticks: [0, 10, 20] });
  const res = GD_PTS.map(() => S('line', { stroke: C.red, 'stroke-width': 2 }, F1.g));
  const ln = S('line', { stroke: C.blue, 'stroke-width': 3 }, F1.g);
  GD_PTS.forEach(([x, y]) => S('circle', { cx: F1.X(x), cy: F1.Y(y), r: 6, fill: C.green }, F1.g));
  curve(F2, ssr, { stroke: C.purple });
  const tan = S('line', { stroke: C.yellow, 'stroke-width': 2 }, F2.g);
  const trail = S('path', { fill: 'none', stroke: C.yellow, 'stroke-width': 1, opacity: 0.5 }, F2.g);
  const dot = S('circle', { r: 7, fill: C.yellow }, F2.g);
  const ro = readout(body);
  const ctl = controls(body);
  let b = 0, lr = 0.1, hist = [];
  const draw = () => {
    ln.setAttribute('x1', F1.X(0)); ln.setAttribute('y1', F1.Y(b));
    ln.setAttribute('x2', F1.X(3.5)); ln.setAttribute('y2', F1.Y(b + SLOPE * 3.5));
    GD_PTS.forEach(([x, y], i) => {
      res[i].setAttribute('x1', F1.X(x)); res[i].setAttribute('x2', F1.X(x));
      res[i].setAttribute('y1', F1.Y(y)); res[i].setAttribute('y2', F1.Y(b + SLOPE * x));
    });
    const v = ssr(b), m = dssr(b);
    dot.setAttribute('cx', F2.X(b)); dot.setAttribute('cy', F2.Y(clamp(v, 0, 20)));
    tan.setAttribute('x1', F2.X(b - 0.6)); tan.setAttribute('y1', F2.Y(v - 0.6 * m));
    tan.setAttribute('x2', F2.X(b + 0.6)); tan.setAttribute('y2', F2.Y(v + 0.6 * m));
    trail.setAttribute('d', pathD(hist.map(q => [F2.X(q), F2.Y(clamp(ssr(q), 0, 20))])));
    ro.innerHTML = `intercept <b>${fmt(b, 3)}</b> · total miss <b>${fmt(v, 3)}</b> · slope of the miss curve <span class="c-yellow">${fmt(m, 3)}</span> · next step = slope × learning rate = <b>${fmt(-m * lr, 4)}</b>`;
  };
  const step = () => { hist.push(b); b -= lr * dssr(b); bSl.set(b); draw(); return Math.abs(lr * dssr(b)) > 0.0005; };
  const run = loop(() => { const go = step(); if (!go) runBtn.textContent = 'run'; return go; });
  const bSl = slider(ctl, 'intercept', -1, 3, 0.001, b, v => { b = v; hist = []; draw(); }, 3);
  slider(ctl, 'learning rate', 0.01, 0.6, 0.01, lr, v => { lr = v; draw(); });
  button(ctl, 'one step', step, 'primary');
  const runBtn = button(ctl, 'run', () => {
    if (run.running) { run.stop(); runBtn.textContent = 'run'; } else { run.start(); runBtn.textContent = 'pause'; }
  });
  button(ctl, 'reset', () => { run.stop(); runBtn.textContent = 'run'; b = 0; hist = []; bSl.set(0); draw(); });
  draw();
};

FIGS.gd2 = body => {
  const ssr = (b, m) => GD_PTS.reduce((s, [x, y]) => s + (y - (b + m * x)) ** 2, 0);
  const grad = (b, m) => GD_PTS.reduce((g, [x, y]) => { const r = y - (b + m * x); return [g[0] - 2 * r, g[1] - 2 * r * x]; }, [0, 0]);
  const svg = svgBox(body, 680, 330);
  const F = frame(svg, 60, 15, 420, 280, [-1.5, 3], [-0.5, 1.6]);
  const n = 56, cw = F.w / n, ch = F.h / n;
  const lo = Math.log(0.6), hi = Math.log(80);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const b = F.iX(F.x + (i + 0.5) * cw), m = F.iY(F.y + (j + 0.5) * ch);
    const t = (Math.log(ssr(b, m)) - lo) / (hi - lo);
    S('rect', { x: F.x + i * cw, y: F.y + j * ch, width: cw + 0.5, height: ch + 0.5, fill: mix('#3d8fa6', '#121217', Math.sqrt(clamp(t, 0, 1))) }, F.g);
  }
  axes(F, { xlabel: 'intercept', ylabel: 'slope', xticks: [-1, 0, 1, 2, 3], yticks: [0, 0.5, 1, 1.5], zero: false });
  const path = S('path', { fill: 'none', stroke: C.yellow, 'stroke-width': 2 }, F.g);
  const dot = S('circle', { r: 7, fill: C.yellow }, F.g);
  const side = S('g', {}, svg);
  text(side, 510, 40, 'click anywhere in the', { class: 'tk' });
  text(side, 510, 56, 'valley to choose a start', { class: 'tk' });
  const ro = readout(body);
  const ctl = controls(body);
  let p = [-1, 1.4], trail = [p.slice()], k = 0;
  const lr = 0.04;
  const draw = () => {
    dot.setAttribute('cx', F.X(p[0])); dot.setAttribute('cy', F.Y(p[1]));
    path.setAttribute('d', pathD(trail.map(q => [F.X(q[0]), F.Y(q[1])])));
    const g = grad(...p);
    ro.innerHTML = `step ${k} · intercept <b>${fmt(p[0], 3)}</b> · slope <b>${fmt(p[1], 3)}</b> · total miss <b>${fmt(ssr(...p), 3)}</b> · downhill push (${fmt(-g[0], 2)}, ${fmt(-g[1], 2)})`;
  };
  const run = loop(() => {
    for (let s = 0; s < 3; s++) {
      const g = grad(...p);
      p = [p[0] - lr * g[0], p[1] - lr * g[1]]; k++;
      trail.push(p.slice());
      if (Math.hypot(...g) < 0.002) { draw(); return false; }
    }
    draw();
    return k < 2000;
  });
  dragArea(svg, pt => {
    if (pt.x < F.x || pt.x > F.x + F.w || pt.y < F.y || pt.y > F.y + F.h) return;
    p = [F.iX(pt.x), F.iY(pt.y)]; trail = [p.slice()]; k = 0; draw();
  });
  button(ctl, 'roll downhill', () => run.start(), 'primary');
  button(ctl, 'reset', () => { run.stop(); p = [-1, 1.4]; trail = [p.slice()]; k = 0; draw(); });
  draw();
};

FIGS.backprop = body => {
  const START_ALL = { w1: 2.74, b1: 0, w2: -1.13, b2: 0, w3: 0.36, w4: 0.63, b3: 0 };
  let mode = 'b3', p, k, losses;
  const grads = q => {
    const g = Object.fromEntries(PKEYS.map(key => [key, 0]));
    for (const [x, obs] of DOSES) {
      const z1 = q.w1 * x + q.b1, z2 = q.w2 * x + q.b2;
      const h1 = softplus(z1), h2 = softplus(z2);
      const dy = -2 * (obs - (q.w3 * h1 + q.w4 * h2 + q.b3));
      g.b3 += dy; g.w3 += dy * h1; g.w4 += dy * h2;
      g.w1 += dy * q.w3 * sig(z1) * x; g.b1 += dy * q.w3 * sig(z1);
      g.w2 += dy * q.w4 * sig(z2) * x; g.b2 += dy * q.w4 * sig(z2);
    }
    return g;
  };
  const loss = q => DOSES.reduce((s, [x, y]) => s + (y - net(q)(x)) ** 2, 0);
  const svg = svgBox(body, 680, 400);
  const F = frame(svg, 40, 20, 280, 210, [0, 1], [-3, 1.6]);
  axes(F, { xlabel: 'dose', xticks: [0, 0.5, 1], yticks: [-3, -2, -1, 0, 1] });
  const cy = curve(F, null, { stroke: C.yellow, 'stroke-width': 3 });
  const res = DOSES.map(() => S('line', { stroke: C.red, 'stroke-width': 2 }, F.g));
  DOSES.forEach(([x, y]) => S('circle', { cx: F.X(x), cy: F.Y(y), r: 6, fill: C.green }, F.g));
  const L = frame(svg, 390, 20, 260, 210, [0, 1], [0, 1]);
  axes(L, { xlabel: 'steps', ylabel: 'total miss', yticks: [] });
  const lp = S('path', { fill: 'none', stroke: C.purple, 'stroke-width': 2 }, L.g);
  const bars = S('g', {}, svg);
  text(bars, 40, 270, 'blame: how much each number is pushed this step (downhill direction)', { class: 'tk' });
  const B = PKEYS.map((key, i) => {
    const x = 60 + i * 86;
    S('line', { x1: x - 30, y1: 330, x2: x + 30, y2: 330, stroke: C.line }, bars);
    const r = S('rect', { x: x - 14, width: 28, fill: C.blue }, bars);
    text(bars, x, 390, key, { class: 'lbl', 'text-anchor': 'middle' });
    const v = text(bars, x, 374, '', { class: 'tk', 'text-anchor': 'middle' });
    return { r, v };
  });
  const ro = readout(body);
  const ctl = controls(body);
  const reset = () => {
    p = mode === 'b3' ? { ...SQ, b3: 0 } : { ...START_ALL };
    k = 0; losses = [loss(p)];
  };
  const draw = () => {
    const f = net(p);
    cy.set(f);
    DOSES.forEach(([x, y], i) => {
      res[i].setAttribute('x1', F.X(x)); res[i].setAttribute('x2', F.X(x));
      res[i].setAttribute('y1', F.Y(y)); res[i].setAttribute('y2', F.Y(clamp(f(x), -3, 1.6)));
    });
    const top = losses[0] || 1;
    lp.setAttribute('d', pathD(losses.map((v, i) => [L.X(losses.length > 1 ? i / (losses.length - 1) : 0), L.Y(clamp(v / top, 0, 1))])));
    const g = grads(p);
    const scale = Math.max(1e-6, ...PKEYS.map(key => Math.abs(g[key])));
    PKEYS.forEach((key, i) => {
      const live = mode === 'all' || key === 'b3';
      const v = live ? -g[key] : 0, hgt = (Math.abs(v) / scale) * 50;
      B[i].r.setAttribute('y', v >= 0 ? 330 - hgt : 330); B[i].r.setAttribute('height', hgt);
      B[i].r.setAttribute('fill', v >= 0 ? C.blue : C.red);
      B[i].v.textContent = live ? `${fmt(p[key])}` : 'fixed';
    });
    ro.innerHTML = `step <b>${k}</b> · total miss <b>${fmt(losses[losses.length - 1], 4)}</b> · b3 = <b>${fmt(p.b3, 3)}</b> · slope of the miss with respect to b3 = sum of (−2 × miss) = <span class="c-yellow">${fmt(g.b3, 4)}</span>`;
  };
  const step = () => {
    const g = grads(p);
    for (const key of PKEYS) if (mode === 'all' || key === 'b3') p[key] -= 0.1 * g[key];
    k++; losses.push(loss(p));
    const size = Math.hypot(...PKEYS.filter(key => mode === 'all' || key === 'b3').map(key => g[key]));
    return size > 1e-4;
  };
  const run = loop(() => {
    let go = true;
    for (let s = 0; s < (mode === 'all' ? 25 : 1) && go; s++) go = step();
    draw();
    if (!go || k > 20000) { runBtn.textContent = 'train'; return false; }
    return true;
  });
  select(ctl, 'what can change', [['b3', 'only the last bias b3'], ['all', 'all seven numbers']], mode, v => {
    mode = v; run.stop(); runBtn.textContent = 'train'; reset(); draw();
  });
  const runBtn = button(ctl, 'train', () => {
    if (run.running) { run.stop(); runBtn.textContent = 'train'; } else { run.start(); runBtn.textContent = 'pause'; }
  }, 'primary');
  button(ctl, 'one step', () => { step(); draw(); });
  button(ctl, 'reset', () => { run.stop(); runBtn.textContent = 'train'; reset(); draw(); });
  reset(); draw();
};

const IRIS = ['Setosa', 'Versicolor', 'Virginica'];
const IRIS_C = [C.blue, C.yellow, C.purple];
const iris = (pw, sw) => {
  const h1 = relu(-2.4 * pw + 0.6 * sw + 1.0), h2 = relu(2.6 * pw - 0.4 * sw - 0.9);
  return { h1, h2, out: [h1 - 0.2 * h2, -0.6 * h1 - 0.6 * h2 + 0.55, -0.2 * h1 + h2] };
};
FIGS.iris = body => {
  const svg = svgBox(body, 680, 340);
  const F = frame(svg, 50, 15, 300, 300, [0, 1], [0, 1]);
  const n = 50, cs = 300 / n;
  const cells = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
    cells.push({ r: S('rect', { x: F.x + i * cs, y: F.y + j * cs, width: cs + 0.4, height: cs + 0.4 }, F.g), pw: (i + 0.5) / n, sw: 1 - (j + 0.5) / n });
  axes(F, { xlabel: 'petal width', ylabel: 'sepal width', xticks: [0, 0.5, 1], yticks: [0.5, 1], zero: false });
  const cross = S('circle', { r: 7, fill: 'none', stroke: C.ink, 'stroke-width': 2.5 }, F.g);
  const side = S('g', {}, svg);
  const rows = IRIS.map((name, i) => {
    const y = 40 + i * 80;
    text(side, 390, y, name, { class: 'lbl', fill: IRIS_C[i] });
    S('line', { x1: 470, y1: y + 4, x2: 470, y2: y + 26, stroke: C.line }, side);
    const r = S('rect', { y: y + 7, height: 16, fill: IRIS_C[i] }, side);
    const v = text(side, 390, y + 42, '', { class: 'tk' });
    return { r, v };
  });
  const hid = text(side, 390, 290, '', { class: 'tk' });
  const ro = readout(body);
  const ctl = controls(body);
  let view = 'winner', at = [0.5, 0.5];
  const paint = () => {
    for (const c of cells) {
      const o = iris(c.pw, c.sw);
      let fill;
      if (view === 'winner') {
        const s = [...o.out].sort((a, b) => b - a), w = o.out.indexOf(s[0]);
        fill = mix('#15151b', IRIS_C[w], 0.25 + clamp((s[0] - s[1]) * 1.5, 0, 0.6));
      } else {
        fill = mix('#15151b', view === 'h1' ? C.blue : C.orange, clamp((view === 'h1' ? o.h1 : o.h2) / 1.6, 0, 1));
      }
      c.r.setAttribute('fill', fill);
    }
  };
  const draw = () => {
    const o = iris(...at), p = softmax(o.out), w = o.out.indexOf(Math.max(...o.out));
    cross.setAttribute('cx', F.X(at[0])); cross.setAttribute('cy', F.Y(at[1]));
    rows.forEach((r, i) => {
      const v = o.out[i], wpx = Math.abs(v) * 90;
      r.r.setAttribute('x', v >= 0 ? 470 : 470 - wpx); r.r.setAttribute('width', wpx);
      r.r.setAttribute('opacity', i === w ? 1 : 0.45);
      r.v.textContent = `raw score ${fmt(v)}   ·   SoftMax ${fmt(p[i] * 100, 0)}%`;
    });
    hid.textContent = `hidden bends: upper ${fmt(o.h1)}, lower ${fmt(o.h2)}`;
    ro.innerHTML = `petal ${fmt(at[0])}, sepal ${fmt(at[1])} → biggest raw score is <b style="color:${IRIS_C[w]}">${IRIS[w]}</b>. Drag on the map.`;
  };
  dragArea(svg, pt => {
    if (pt.x < F.x || pt.x > F.x + F.w || pt.y < F.y || pt.y > F.y + F.h) return;
    at = [F.iX(pt.x), F.iY(pt.y)]; draw();
  });
  segmented(ctl, [['winner', 'which output wins'], ['h1', 'hidden bend 1'], ['h2', 'hidden bend 2']], view, v => { view = v; paint(); });
  paint(); draw();
};

FIGS.softmax = body => {
  const z = [1.43, -0.4, 0.23];
  const svg = svgBox(body, 680, 230);
  const g = S('g', {}, svg);
  const cols = [['raw score', 60], ['ArgMax', 280], ['SoftMax', 480]];
  cols.forEach(([t, x]) => text(g, x, 20, t, { class: 'lbl' }));
  const R = IRIS.map((name, i) => {
    const y = 50 + i * 60;
    text(g, 60, y + 40, name, { class: 'tk', fill: IRIS_C[i] });
    S('line', { x1: 140, y1: y, x2: 140, y2: y + 26, stroke: C.line }, g);
    return {
      raw: S('rect', { y, height: 26, fill: IRIS_C[i] }, g),
      arg: S('rect', { x: 280, y, height: 26, fill: IRIS_C[i] }, g),
      sm: S('rect', { x: 480, y, height: 26, fill: IRIS_C[i] }, g),
      av: text(g, 280, y + 44, '', { class: 'tk' }), sv: text(g, 480, y + 44, '', { class: 'tk' }),
    };
  });
  H('p', { class: 'tk', style: 'text-align:center;margin:10px 0 0' }, body, 'Each row: nudge that raw score up by a small amount. Each column: how much that SoftMax output moves, per unit of nudge.');
  const tbl = H('table', { class: 'mtx sens' }, body);
  const ro = readout(body);
  const ctl = controls(body);
  const draw = () => {
    const p = softmax(z), w = z.indexOf(Math.max(...z));
    R.forEach((r, i) => {
      const px = Math.abs(z[i]) * 25;
      r.raw.setAttribute('x', z[i] >= 0 ? 140 : 140 - px); r.raw.setAttribute('width', px);
      r.arg.setAttribute('width', i === w ? 160 : 1); r.av.textContent = i === w ? '1' : '0';
      r.sm.setAttribute('width', 1 + p[i] * 160); r.sm.setAttribute('opacity', 0.35 + p[i] * 0.65);
      r.sv.textContent = fmt(p[i], 3);
    });
    tbl.innerHTML = '<tr><th></th>' + IRIS.map(n => `<th>${n}</th>`).join('') + '</tr>'
      + IRIS.map((n, i) => `<tr><th>${n} ↑</th>` + IRIS.map((_, j) => {
        const v = p[i] * ((i === j ? 1 : 0) - p[j]);
        return `<td style="background:${heat(v, 0.25)}">${fmt(v, 3)}</td>`;
      }).join('') + '</tr>').join('');
    ro.innerHTML = `SoftMax outputs add to <b>${fmt(p.reduce((a, b) => a + b), 3)}</b>. ArgMax only says who won, so its slope is zero almost everywhere and it cannot teach (§9 needs a slope).`;
  };
  IRIS.forEach((n, i) => slider(ctl, n, -3, 5, 0.01, z[i], v => { z[i] = v; draw(); }));
  draw();
};

FIGS.xent = body => {
  const z = [1.43, -0.4, 0.23];
  let target = 0;
  const svg = svgBox(body, 680, 260);
  const F = frame(svg, 50, 20, 260, 200, [0, 1], [0, 5]);
  axes(F, { xlabel: 'probability given to the right answer', ylabel: 'loss', xticks: [0, 0.5, 1], yticks: [0, 1, 2, 3, 4, 5] });
  curve(F, p => -Math.log(Math.max(p, 1e-4)), { stroke: C.red });
  const dot = S('circle', { r: 7, fill: C.yellow }, F.g);
  const g = S('g', {}, svg);
  text(g, 380, 30, 'push on each raw score = probability − target', { class: 'tk' });
  const R = IRIS.map((name, i) => {
    const y = 60 + i * 60;
    text(g, 380, y + 18, name, { class: 'lbl', fill: IRIS_C[i] });
    S('line', { x1: 540, y1: y, x2: 540, y2: y + 26, stroke: C.line }, g);
    return { r: S('rect', { y, height: 26 }, g), v: text(g, 380, y + 38, '', { class: 'tk' }) };
  });
  const ro = readout(body);
  const ctl = controls(body);
  const sl = [];
  const draw = () => {
    const p = softmax(z), loss = -Math.log(p[target]);
    dot.setAttribute('cx', F.X(p[target])); dot.setAttribute('cy', F.Y(Math.min(loss, 5)));
    R.forEach((r, i) => {
      const push = p[i] - (i === target ? 1 : 0), w = Math.abs(push) * 120;
      r.r.setAttribute('x', push >= 0 ? 540 : 540 - w); r.r.setAttribute('width', w);
      r.r.setAttribute('fill', push >= 0 ? C.red : C.green);
      r.v.textContent = `p ${fmt(p[i], 3)} − target ${i === target ? 1 : 0} = ${fmt(push, 3)}`;
    });
    ro.innerHTML = `The right answer is <b style="color:${IRIS_C[target]}">${IRIS[target]}</b>. It gets probability ${fmt(p[target], 3)}, so the loss is −log(${fmt(p[target], 3)}) = <b>${fmt(loss, 3)}</b>. <span class="c-green">Green</span> means "raise this score"; <span class="c-red">red</span> means "lower it".`;
  };
  IRIS.forEach((n, i) => sl.push(slider(ctl, n, -3, 5, 0.01, z[i], v => { z[i] = v; draw(); })));
  select(ctl, 'right answer', IRIS.map((n, i) => [String(i), n]), '0', v => { target = +v; draw(); });
  button(ctl, 'take one learning step', () => {
    const p = softmax(z);
    z.forEach((v, i) => { z[i] = clamp(v - 1.0 * (p[i] - (i === target ? 1 : 0)), -3, 5); sl[i].set(z[i]); });
    draw();
  }, 'primary');
  draw();
};

FIGS.cnn = body => {
  const X = ['100001', '010010', '001100', '001100', '010010', '100001'];
  const O = ['011110', '100001', '100001', '100001', '100001', '011110'];
  const XS = ['000000', '100010', '010100', '001000', '010100', '100010'];
  const grid = s => s.map(r => [...r].map(Number));
  let img = grid(X), filt = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], bias = -2, hover = null;
  const wrap = H('div', { class: 'cnn' }, body);
  const mk = (title, n, cls) => {
    const col = H('div', { class: 'cnn-col' }, wrap);
    H('div', { class: 'cnn-t' }, col, title);
    const gEl = H('div', { class: `cnn-grid ${cls}`, style: `grid-template-columns:repeat(${n},1fr)` }, col);
    return Array.from({ length: n * n }, () => H('div', { class: 'cell' }, gEl));
  };
  const inC = mk('picture (click cells)', 6, 'g6');
  const fC = mk('filter (click)', 3, 'g3');
  const mC = mk('filter score + bias', 4, 'g4');
  const rC = mk('after ReLU', 4, 'g4');
  const pC = mk('max of each 2×2', 2, 'g2');
  const ro = readout(body);
  const ctl = controls(body);
  const conv = () => {
    const m = [];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = bias;
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += img[i + a][j + b] * filt[a][b];
      m.push(s);
    }
    return m;
  };
  const draw = () => {
    const m = conv(), r = m.map(relu);
    const pool = [0, 1, 2, 3].map(q => {
      const bi = Math.floor(q / 2) * 2, bj = (q % 2) * 2;
      return Math.max(r[bi * 4 + bj], r[bi * 4 + bj + 1], r[(bi + 1) * 4 + bj], r[(bi + 1) * 4 + bj + 1]);
    });
    inC.forEach((c, k) => {
      const i = Math.floor(k / 6), j = k % 6;
      c.classList.toggle('on', img[i][j] === 1);
      c.classList.toggle('hl', hover != null && i >= hover[0] && i < hover[0] + 3 && j >= hover[1] && j < hover[1] + 3);
    });
    fC.forEach((c, k) => c.classList.toggle('on', filt[Math.floor(k / 3)][k % 3] === 1));
    const paintNum = (cells, vals, scale) => cells.forEach((c, k) => { c.textContent = fmt(vals[k], 0); c.style.background = heat(vals[k], scale); });
    paintNum(mC, m, 3); paintNum(rC, r, 3); paintNum(pC, pool, 3);
    ro.innerHTML = `The filter is a 3×3 stencil of weights. Lay it on each 3×3 window, multiply cell by cell, add, add the bias (${bias}), then ReLU. Max pooling keeps the strongest response in each corner: [${pool.join(', ')}]. Those four numbers go into an ordinary network like §7.`;
  };
  inC.forEach((c, k) => c.addEventListener('click', () => { const i = Math.floor(k / 6), j = k % 6; img[i][j] ^= 1; draw(); }));
  fC.forEach((c, k) => c.addEventListener('click', () => { filt[Math.floor(k / 3)][k % 3] ^= 1; draw(); }));
  [mC, rC].forEach(cells => cells.forEach((c, k) => {
    c.addEventListener('mouseenter', () => { hover = [Math.floor(k / 4), k % 4]; draw(); });
    c.addEventListener('mouseleave', () => { hover = null; draw(); });
  }));
  button(ctl, 'X', () => { img = grid(X); draw(); });
  button(ctl, 'O', () => { img = grid(O); draw(); });
  button(ctl, 'X moved one cell', () => { img = grid(XS); draw(); });
  slider(ctl, 'bias', -4, 1, 1, bias, v => { bias = v; draw(); }, 0);
  draw();
};

FIGS.rnn = body => {
  let w = 0.5, n = 20;
  const svg = svgBox(body, 680, 330);
  const g = S('g', {}, svg);
  const boxes = [];
  for (let i = 0; i < 6; i++) {
    const x = 30 + i * 105;
    boxes.push(S('rect', { x, y: 30, width: 62, height: 44, rx: 8, fill: '#1b1b22', stroke: C.blue, 'stroke-width': 2 }, g));
    text(g, x + 31, 57, i < 5 ? `day ${i + 1}` : '…', { class: 'lbl', 'text-anchor': 'middle' });
    S('line', { x1: x + 31, y1: 100, x2: x + 31, y2: 74, stroke: C.dim, 'stroke-width': 2 }, g);
    if (i < 5) {
      const a = makeArrow(g, C.yellow, 2); a.set(x + 62, 52, x + 103, 52);
      text(g, x + 83, 44, '× w', { class: 'tk', 'text-anchor': 'middle' });
    }
  }
  const F = frame(svg, 60, 130, 590, 170, [0, 50], [-6, 6]);
  axes(F, { xlabel: 'steps back in time', xticks: [0, 10, 20, 30, 40, 50], yticks: [] });
  [-6, -3, 0, 3, 6].forEach(t => {
    S('line', { x1: F.x, y1: F.Y(t), x2: F.x + F.w, y2: F.Y(t), stroke: C.faint }, F.g);
    text(F.g, F.x - 6, F.Y(t) + 4, t === 0 ? '1' : `1e${t}`, { class: 'tk', 'text-anchor': 'end' });
  });
  const bars = Array.from({ length: 51 }, () => S('rect', { width: 8 }, F.g));
  const ro = readout(body);
  const ctl = controls(body);
  const draw = () => {
    bars.forEach((r, k) => {
      const v = k <= n ? clamp(k * Math.log10(Math.max(w, 1e-9)), -6, 6) : 0;
      r.setAttribute('x', F.X(k) - 4);
      r.setAttribute('y', v >= 0 ? F.Y(v) : F.Y(0)); r.setAttribute('height', Math.abs(F.Y(v) - F.Y(0)));
      r.setAttribute('fill', k > n ? 'none' : w > 1 ? C.red : w < 1 ? C.blue : C.green);
    });
    const tot = Math.pow(w, n);
    const word = w > 1.0001 ? '<span class="c-red">explodes</span>' : w < 0.9999 ? '<span class="c-blue">vanishes</span>' : 'stays put';
    ro.innerHTML = `The first day's input reaches the end multiplied by w ${n} times: ${fmt(w)}<sup>${n}</sup> = <b>${tot > 1e5 || tot < 1e-4 ? tot.toExponential(2) : fmt(tot, 4)}</b>. Its effect, and its slope during training, ${word}.`;
  };
  slider(ctl, 'shared weight w', 0.1, 2, 0.01, w, v => { w = v; draw(); });
  slider(ctl, 'number of steps', 1, 50, 1, n, v => { n = v; draw(); }, 0);
  draw();
};

FIGS.lstm = body => {
  const W = { f: [2.7, 1.63, 1.62], i: [2.0, 1.65, 0.62], g: [1.41, 0.94, -0.32], o: [4.38, -0.19, 0.59] };
  const xs = [1, 0.5, 0.25, 1];
  let day = 3;
  const run = () => {
    let c = 0, h = 0;
    return xs.map(x => {
      const f = sig(W.f[0] * h + W.f[1] * x + W.f[2]);
      const i = sig(W.i[0] * h + W.i[1] * x + W.i[2]);
      const gg = Math.tanh(W.g[0] * h + W.g[1] * x + W.g[2]);
      const o = sig(W.o[0] * h + W.o[1] * x + W.o[2]);
      c = f * c + i * gg; h = o * Math.tanh(c);
      return { x, f, i, g: gg, o, c, h };
    });
  };
  const svg = svgBox(body, 680, 330);
  const g = S('g', {}, svg);
  S('line', { x1: 20, y1: 50, x2: 560, y2: 50, stroke: C.green, 'stroke-width': 5 }, g);
  text(g, 24, 38, 'long-term memory (a straight belt: no weight multiplies it)', { class: 'tk', fill: C.green });
  S('line', { x1: 20, y1: 220, x2: 560, y2: 220, stroke: C.purple, 'stroke-width': 3 }, g);
  text(g, 24, 244, 'short-term memory (the output)', { class: 'tk', fill: C.purple });
  const gate = (x, label, color, top) => {
    S('rect', { x: x - 34, y: 90, width: 68, height: 90, rx: 8, fill: '#1b1b22', stroke: color, 'stroke-width': 2 }, g);
    const fill = S('rect', { x: x - 30, width: 60, fill: color, opacity: 0.55 }, g);
    text(g, x, 200, label, { class: 'tk', 'text-anchor': 'middle' });
    const v = text(g, x, 140, '', { class: 'val', 'text-anchor': 'middle' });
    S('line', { x1: x, y1: top ? 90 : 180, x2: x, y2: top ? 52 : 218, stroke: color, 'stroke-width': 2 }, g);
    return (pct, str) => { fill.setAttribute('y', 176 - pct * 82); fill.setAttribute('height', pct * 82); v.textContent = str; };
  };
  const gf = gate(120, 'how much to keep', C.red, true);
  const gi = gate(250, 'how much new to add', C.yellow, true);
  const gc = gate(360, 'new candidate', C.orange, true);
  const go = gate(490, 'how much to show', C.blue, false);
  const P = frame(svg, 600, 40, 70, 200, [0, 3], [-1, 1]);
  axes(P, { xticks: [], yticks: [] });
  const pc = S('path', { fill: 'none', stroke: C.green, 'stroke-width': 2.5 }, P.g);
  const ph = S('path', { fill: 'none', stroke: C.purple, 'stroke-width': 2.5 }, P.g);
  const mark = S('line', { stroke: C.dim, 'stroke-dasharray': '3 3' }, P.g);
  text(g, 600, 290, 'days 1–4', { class: 'tk' });
  const ro = readout(body);
  const ctl = controls(body);
  const draw = () => {
    const r = run(), s = r[day];
    gf(s.f, `${fmt(s.f * 100, 0)}%`); gi(s.i, `${fmt(s.i * 100, 0)}%`);
    gc((s.g + 1) / 2, fmt(s.g)); go(s.o, `${fmt(s.o * 100, 0)}%`);
    pc.setAttribute('d', pathD(r.map((q, k) => [P.X(k), P.Y(clamp(q.c, -1, 1))])));
    ph.setAttribute('d', pathD(r.map((q, k) => [P.X(k), P.Y(clamp(q.h, -1, 1))])));
    mark.setAttribute('x1', P.X(day)); mark.setAttribute('x2', P.X(day)); mark.setAttribute('y1', P.y); mark.setAttribute('y2', P.y + P.h);
    ro.innerHTML = `Day ${day + 1}, input ${fmt(s.x)}: keep ${fmt(s.f * 100, 0)}% of the old long-term memory, add ${fmt(s.i * 100, 0)}% of the candidate ${fmt(s.g)} → long-term memory <b class="c-green">${fmt(s.c, 3)}</b>; show ${fmt(s.o * 100, 0)}% of it, bent → output <b class="c-purple">${fmt(s.h, 3)}</b>.`;
  };
  xs.forEach((v, k) => slider(ctl, `day ${k + 1} input`, 0, 1, 0.01, v, nv => { xs[k] = nv; draw(); }));
  segmented(ctl, [0, 1, 2, 3].map(k => [String(k), `look at day ${k + 1}`]), String(day), v => { day = +v; draw(); });
  draw();
};

FIGS.w2v = body => {
  const V = ['Troll 2', 'is', 'great', 'Gymkata'];
  const VC = [C.blue, C.dim, C.green, C.yellow];
  const pairs = [[0, 1], [1, 2], [3, 1], [1, 2]];
  let E, U, epoch, trails;
  const init = () => {
    const r = rng(7);
    E = V.map(() => [gauss(r) * 0.9, gauss(r) * 0.9]);
    U = [V.map(() => gauss(r) * 0.5), V.map(() => gauss(r) * 0.5)];
    epoch = 0; trails = V.map((_, i) => [E[i].slice()]);
  };
  const lossOf = () => pairs.reduce((s, [a, b]) => {
    const p = softmax(V.map((_, j) => E[a][0] * U[0][j] + E[a][1] * U[1][j]));
    return s - Math.log(p[b]);
  }, 0) / pairs.length;
  const train = () => {
    for (const [a, b] of pairs) {
      const e = E[a];
      const p = softmax(V.map((_, j) => e[0] * U[0][j] + e[1] * U[1][j]));
      const dz = p.map((v, j) => v - (j === b ? 1 : 0));
      const de = [0, 1].map(d => dz.reduce((s, v, j) => s + v * U[d][j], 0));
      for (let d = 0; d < 2; d++) for (let j = 0; j < V.length; j++) U[d][j] -= 0.1 * e[d] * dz[j];
      e[0] -= 0.1 * de[0]; e[1] -= 0.1 * de[1];
    }
    epoch++;
    if (epoch % 4 === 0) E.forEach((e, i) => trails[i].push(e.slice()));
  };
  const svg = svgBox(body, 680, 340);
  const F = frame(svg, 40, 10, 320, 320, [-3.5, 3.5], [-3.5, 3.5]);
  axes(F, { xticks: [], yticks: [] });
  const tr = V.map((_, i) => S('path', { fill: 'none', stroke: VC[i], 'stroke-width': 1, opacity: 0.45 }, F.g));
  const dots = V.map((w, i) => {
    const grp = S('g', {}, F.g);
    S('circle', { r: 7, fill: VC[i] }, grp);
    text(grp, 11, 5, w, { class: 'lbl', fill: VC[i] });
    return grp;
  });
  const side = S('g', {}, svg);
  text(side, 400, 40, 'training examples (word → next word):', { class: 'tk' });
  pairs.forEach(([a, b], i) => text(side, 400, 66 + i * 22, `${V[a]} → ${V[b]}`, { class: 'lbl' }));
  const L = frame(svg, 400, 180, 250, 130, [0, 1], [0, 1]);
  axes(L, { xlabel: 'training rounds', ylabel: 'average loss', yticks: [] });
  const lp = S('path', { fill: 'none', stroke: C.purple, 'stroke-width': 2 }, L.g);
  let losses;
  const ro = readout(body);
  const ctl = controls(body);
  const draw = () => {
    E.forEach((e, i) => {
      dots[i].setAttribute('transform', `translate(${F.X(clamp(e[0], -3.4, 3.4))},${F.Y(clamp(e[1], -3.4, 3.4))})`);
      tr[i].setAttribute('d', pathD(trails[i].map(q => [F.X(clamp(q[0], -3.4, 3.4)), F.Y(clamp(q[1], -3.4, 3.4))])));
    });
    lp.setAttribute('d', pathD(losses.map((v, i) => [L.X(i / Math.max(1, losses.length - 1)), L.Y(clamp(v / losses[0], 0, 1))])));
    const d = Math.hypot(E[0][0] - E[3][0], E[0][1] - E[3][1]);
    ro.innerHTML = `round ${epoch} · average loss <b>${fmt(losses[losses.length - 1], 3)}</b> · distance between <span class="c-blue">Troll 2</span> and <span class="c-yellow">Gymkata</span>: <b>${fmt(d)}</b>. Both are followed by "is", so training pulls them to the same place.`;
  };
  const reset = () => { init(); losses = [lossOf()]; };
  const run = loop(() => {
    for (let s = 0; s < 2; s++) { train(); losses.push(lossOf()); }
    draw();
    if (epoch >= 600) { runBtn.textContent = 'train'; return false; }
    return true;
  });
  const runBtn = button(ctl, 'train', () => {
    if (run.running) { run.stop(); runBtn.textContent = 'train'; } else { if (epoch >= 600) reset(); run.start(); runBtn.textContent = 'pause'; }
  }, 'primary');
  button(ctl, 'reset', () => { run.stop(); runBtn.textContent = 'train'; reset(); draw(); });
  reset(); draw();
};

FIGS.seq2seq = body => {
  const svg = svgBox(body, 680, 260);
  const g = S('g', {}, svg);
  const enc = ["Let's", 'go', '<EOS>'], dec = ['<EOS>', 'vamos'], outs = ['vamos', '<EOS>'];
  const box = (x, y, label, color) => {
    const r = S('rect', { x, y, width: 80, height: 48, rx: 8, fill: '#1b1b22', stroke: color, 'stroke-width': 2 }, g);
    text(g, x + 40, y + 29, label, { class: 'lbl', 'text-anchor': 'middle' });
    return r;
  };
  const E = enc.map((w, i) => { box(30 + i * 100, 190, w, C.dim); return box(30 + i * 100, 110, 'LSTM', C.blue); });
  const D = dec.map((w, i) => { box(400 + i * 130, 190, w, C.dim); return box(400 + i * 130, 110, 'LSTM', C.orange); });
  const O = outs.map((w, i) => box(400 + i * 130, 30, w, C.green));
  for (let i = 0; i < 2; i++) makeArrow(g, C.line, 2).set(110 + i * 100, 134, 128 + i * 100, 134);
  makeArrow(g, C.line, 2).set(480, 134, 528, 134);
  const ctx = S('path', { d: 'M310 134 C 350 134, 360 134, 398 134', fill: 'none', stroke: C.yellow, 'stroke-width': 4 }, g);
  text(g, 354, 124, 'context', { class: 'tk', 'text-anchor': 'middle', fill: C.yellow });
  const fb = S('path', { d: 'M480 54 C 505 54, 505 214, 528 214', fill: 'none', stroke: C.green, 'stroke-width': 2, 'stroke-dasharray': '5 4' }, g);
  const cap = readout(body);
  const steps = [
    ['enc', 0, `The encoder reads "Let's". It looks up the word's embedding (§13) and updates its two memories (§12).`],
    ['enc', 1, 'It reads "go" with the same weights, carrying the memories forward (§11).'],
    ['enc', 2, 'It reads "<EOS>", the end-of-sentence marker. Its final memories are the context: the whole sentence squeezed into a fixed set of numbers.'],
    ['dec', 0, 'The decoder starts from the context. It is fed "<EOS>" to mean "begin", and its output passes through a fully connected layer and SoftMax (§8) over Spanish words. The biggest is "vamos".'],
    ['dec', 1, 'The decoder is fed its own output "vamos" and predicts "<EOS>": stop. During training it is fed the correct word instead of its own guess ("teacher forcing").'],
  ];
  stepper(body, steps.length, i => {
    const [side, k, t] = steps[i];
    E.forEach((r, j) => r.setAttribute('stroke-width', side === 'enc' && j === k ? 5 : 2));
    D.forEach((r, j) => r.setAttribute('stroke-width', side === 'dec' && j === k ? 5 : 2));
    O.forEach((r, j) => r.style.opacity = side === 'dec' && j <= k ? 1 : 0.15);
    ctx.style.opacity = i >= 2 ? 1 : 0.15; fb.style.opacity = i === 4 ? 1 : 0.15;
    cap.innerHTML = t;
  });
};

FIGS.attention = body => {
  const words = ["Let's", 'go', '<EOS>'];
  const vecs = [[1.6, 0.5], [0.3, 1.5], [-1.2, 0.4]];
  const cols = [C.blue, C.orange, C.purple];
  let q = [1.0, 1.0];
  const svg = svgBox(body, 680, 340);
  const F = frame(svg, 20, 10, 320, 320, [-2.2, 2.2], [-2.2, 2.2]);
  for (let t = -2; t <= 2; t++) {
    S('line', { x1: F.X(t), y1: F.y, x2: F.X(t), y2: F.y + F.h, stroke: C.faint }, F.g);
    S('line', { x1: F.x, y1: F.Y(t), x2: F.x + F.w, y2: F.Y(t), stroke: C.faint }, F.g);
  }
  vecs.forEach((v, i) => {
    makeArrow(F.g, cols[i], 3).set(F.X(0), F.Y(0), F.X(v[0]), F.Y(v[1]));
    text(F.g, F.X(v[0]) + 6, F.Y(v[1]) - 6, words[i], { class: 'lbl', fill: cols[i] });
  });
  const mixA = makeArrow(F.g, C.green, 3);
  const qa = makeArrow(F.g, C.yellow, 3.5);
  const qh = S('circle', { r: 9, fill: C.yellow, opacity: 0.35 }, F.g);
  const side = S('g', {}, svg);
  text(side, 380, 30, 'similarity (dot product) → SoftMax → share', { class: 'tk' });
  const R = words.map((w, i) => {
    const y = 60 + i * 64;
    text(side, 380, y + 17, w, { class: 'lbl', fill: cols[i] });
    return { r: S('rect', { x: 470, y, height: 24, fill: cols[i] }, side), v: text(side, 470, y + 42, '', { class: 'tk' }) };
  });
  text(side, 380, 280, 'green = shares × encoder vectors, added up', { class: 'tk', fill: C.green });
  const ro = readout(body);
  const draw = () => {
    const s = vecs.map(v => v[0] * q[0] + v[1] * q[1]), p = softmax(s);
    const m = [0, 1].map(d => vecs.reduce((acc, v, i) => acc + p[i] * v[d], 0));
    qa.set(F.X(0), F.Y(0), F.X(q[0]), F.Y(q[1]));
    qh.setAttribute('cx', F.X(q[0])); qh.setAttribute('cy', F.Y(q[1]));
    mixA.set(F.X(0), F.Y(0), F.X(m[0]), F.Y(m[1]));
    R.forEach((r, i) => { r.r.setAttribute('width', 2 + p[i] * 180); r.v.textContent = `similarity ${fmt(s[i])}  ·  share ${fmt(p[i] * 100, 0)}%`; });
    ro.innerHTML = `Drag the <span class="c-yellow">yellow query</span> (the decoder's current state). Words pointing the same way get a big dot product, so they get a big share of the <span class="c-green">mix</span>.`;
  };
  dragArea(svg, pt => {
    if (pt.x > F.x + F.w) return;
    q = [clamp(F.iX(pt.x), -2.1, 2.1), clamp(F.iY(pt.y), -2.1, 2.1)]; draw();
  });
  draw();
};

FIGS.tensors = body => {
  const svg = svgBox(body, 680, 280);
  const g = S('g', {}, svg);
  const ro = readout(body);
  const ctl = controls(body);
  const sheet = (x, y, rows, cols, size, color, op = 1) => {
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++)
      S('rect', { x: x + j * size, y: y + i * size, width: size - 2, height: size - 2, fill: color, opacity: op, rx: 2 }, g);
  };
  const views = {
    0: ['one number', () => sheet(320, 120, 1, 1, 34, C.yellow), 'A scalar: one number, such as one weight or a loss. Shape: []. Count: 1.'],
    1: ['a list', () => sheet(170, 120, 1, 10, 34, C.blue), 'A list (vector): one token\'s embedding is a list. In SmolVLM2 each text token is a list of 2,048 numbers. Shape: [2048].'],
    2: ['a table', () => sheet(230, 40, 6, 7, 30, C.blue), 'A table (matrix): rows × columns. A weight table such as q_proj in SmolVLM2 is 2,048 × 2,048 = 4,194,304 numbers. Shape: [2048, 2048].'],
    3: ['a stack of tables', () => { const c = [C.red, C.green, C.blue]; [2, 1, 0].forEach(k => sheet(230 + k * 26, 40 + k * 26, 6, 6, 28, c[k], 0.85)); }, 'A stack: a colour picture is three tables (red, green, blue). One SmolVLM2 tile is 3 × 384 × 384 = 442,368 numbers.'],
    4: ['a stack of stacks', () => { for (let t = 4; t >= 0; t--) [2, 1, 0].forEach(k => sheet(90 + t * 100 + k * 10, 60 + k * 10, 5, 5, 14, [C.red, C.green, C.blue][k], 0.85)); }, 'A batch: many pictures. One photo at the default 13 tiles is 13 × 3 × 384 × 384 = 5,750,784 numbers. "Tensor" is just the general word for all of these.'],
  };
  const show = k => {
    g.innerHTML = '';
    views[k][1]();
    text(g, 340, 268, views[k][0], { class: 'lbl', 'text-anchor': 'middle' });
    ro.innerHTML = views[k][2];
  };
  segmented(ctl, Object.keys(views).map(k => [k, `${k} direction${k === '1' ? '' : 's'}`]), '2', show);
  show('2');
};

FIGS.matrix = body => {
  const m = { a: 1, b: 1, c: 0, d: 1 };
  let v = [1, 1];
  const svg = svgBox(body, 680, 340);
  const F = frame(svg, 20, 10, 320, 320, [-4, 4], [-4, 4]);
  for (let t = -4; t <= 4; t++) {
    S('line', { x1: F.X(t), y1: F.y, x2: F.X(t), y2: F.y + F.h, stroke: C.faint }, F.g);
    S('line', { x1: F.x, y1: F.Y(t), x2: F.x + F.w, y2: F.Y(t), stroke: C.faint }, F.g);
  }
  const clipId = `clip-matrix-${Math.random().toString(36).slice(2)}`;
  S('rect', { x: F.x, y: F.y, width: F.w, height: F.h }, S('clipPath', { id: clipId }, S('defs', {}, svg)));
  F.g.setAttribute('clip-path', `url(#${clipId})`);
  const grid = S('g', {}, F.g);
  const ia = makeArrow(F.g, C.green, 3), ja = makeArrow(F.g, C.red, 3), va = makeArrow(F.g, C.yellow, 3.5), v0 = makeArrow(F.g, C.dim, 2);
  const eq = H('div', { class: 'mathline' }, body);
  const ro = readout(body);
  const ctl = controls(body);
  const T = ([x, y]) => [m.a * x + m.b * y, m.c * x + m.d * y];
  const draw = () => {
    grid.innerHTML = '';
    for (let t = -6; t <= 6; t++) {
      const p1 = T([t, -6]), p2 = T([t, 6]), p3 = T([-6, t]), p4 = T([6, t]);
      S('line', { x1: F.X(p1[0]), y1: F.Y(p1[1]), x2: F.X(p2[0]), y2: F.Y(p2[1]), stroke: C.blue, opacity: t ? 0.35 : 0.8 }, grid);
      S('line', { x1: F.X(p3[0]), y1: F.Y(p3[1]), x2: F.X(p4[0]), y2: F.Y(p4[1]), stroke: C.blue, opacity: t ? 0.35 : 0.8 }, grid);
    }
    const o = [F.X(0), F.Y(0)];
    ia.set(...o, F.X(m.a), F.Y(m.c)); ja.set(...o, F.X(m.b), F.Y(m.d));
    const w = T(v);
    v0.set(...o, F.X(v[0]), F.Y(v[1])); va.set(...o, F.X(w[0]), F.Y(w[1]));
    eq.innerHTML = `<span class="mat">${fmt(m.a)}&nbsp;&nbsp;${fmt(m.b)}<br>${fmt(m.c)}&nbsp;&nbsp;${fmt(m.d)}</span> × <span class="mat">${fmt(v[0])}<br>${fmt(v[1])}</span> = <span class="mat">${fmt(m.a)}×${fmt(v[0])} + ${fmt(m.b)}×${fmt(v[1])}<br>${fmt(m.c)}×${fmt(v[0])} + ${fmt(m.d)}×${fmt(v[1])}</span> = <span class="mat c-yellow">${fmt(w[0])}<br>${fmt(w[1])}</span>`;
    ro.innerHTML = `The <span class="c-green">first column</span> is where the right-pointing arrow lands. The <span class="c-red">second column</span> is where the up-pointing arrow lands. Every other point follows along, so the whole grid moves. Drag to move the grey input arrow.`;
  };
  for (const k of ['a', 'b', 'c', 'd']) slider(ctl, k, -2, 2, 0.01, m[k], val => { m[k] = val; draw(); });
  dragArea(svg, pt => { if (pt.x > F.x + F.w) return; v = [clamp(F.iX(pt.x), -4, 4), clamp(F.iY(pt.y), -4, 4)]; draw(); });
  draw();
};

FIGS.posenc = body => {
  let pos = 3;
  const dims = [[Math.sin, 1, C.blue], [Math.cos, 1, C.orange], [Math.sin, 0.3, C.green], [Math.cos, 0.3, C.purple]];
  const svg = svgBox(body, 680, 320);
  const rows = dims.map(([f, freq, col], i) => {
    const F = frame(svg, 60, 15 + i * 72, 590, 58, [0, 12], [-1.2, 1.2]);
    S('line', { x1: F.x, y1: F.Y(0), x2: F.x + F.w, y2: F.Y(0), stroke: C.faint }, F.g);
    curve(F, x => f(freq * x), { stroke: col, 'stroke-width': 2 });
    text(F.g, 10, F.Y(0) + 4, `slot ${i + 1}`, { class: 'tk' });
    for (let k = 0; k <= 12; k++) S('circle', { cx: F.X(k), cy: F.Y(f(freq * k)), r: 2.5, fill: C.dim }, F.g);
    return { F, d: S('circle', { r: 6, fill: col }, F.g), f, freq };
  });
  const mark = S('line', { stroke: C.yellow, 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }, svg);
  const ro = readout(body);
  const ctl = controls(body);
  const draw = () => {
    const vals = rows.map(r => { const y = r.f(r.freq * pos); r.d.setAttribute('cx', r.F.X(pos)); r.d.setAttribute('cy', r.F.Y(y)); return y; });
    const x = rows[0].F.X(pos);
    mark.setAttribute('x1', x); mark.setAttribute('x2', x); mark.setAttribute('y1', 10); mark.setAttribute('y2', 305);
    ro.innerHTML = `Word number ${pos} gets the position list <b>[${vals.map(v => fmt(v)).join(', ')}]</b>, which is added to its word embedding. No other position gets exactly this list.`;
  };
  slider(ctl, 'position in the sentence', 0, 12, 1, pos, v => { pos = v; draw(); }, 0);
  draw();
};

FIGS.transformer = body => {
  const blocks = [
    ['tok', 'Tokens', 'The text is cut into pieces (tokens), and each piece is given an ID number from the vocabulary. SmolVLM2 has 49,280 IDs.', ['s13']],
    ['emb', 'Word embedding', 'Look up the row of the embedding table for each ID (§13). SmolVLM2: a 49,280 × 2,048 table, so each token becomes 2,048 numbers.', ['s13', 's16']],
    ['pos', 'Position', 'Tell each token where it sits. The original transformer adds sine waves (§18). SmolVLM2 rotates the query and key lists by an angle that grows with position ("RoPE"); the idea is the same.', ['s18']],
    ['att', 'Self-attention', 'Every token asks a question (query), offers a label (key) and carries content (value). It mixes in the values of the tokens whose keys match its query (§15, §19).', ['s15', 's19']],
    ['res1', 'Add (residual)', 'Add the attention result back onto the input instead of replacing it. Each block only has to learn a correction, and slopes flow straight back through the addition during training (§6).', ['s6']],
    ['norm', 'Normalise', 'Rescale each token\'s list to a steady size so that numbers do not drift as they pass through 24 blocks. SmolVLM2 uses RMSNorm: divide by the root-mean-square, then multiply by learned per-slot gains.', ['s16']],
    ['ffn', 'Feed-forward network', 'The same small network from §2 and §7, applied to each token on its own: widen to 8,192, bend, narrow back to 2,048. SmolVLM2 uses a gated version: one widened copy, bent by SiLU, multiplies a second widened copy.', ['s2', 's7']],
    ['res2', 'Add (residual)', 'Add again. Then repeat attention, add, normalise, feed-forward, add: 24 times in SmolVLM2.', ['s6']],
    ['out', 'Unembed + SoftMax', 'Score every vocabulary entry with one more table (2,048 → 49,280) and turn the scores into probabilities (§8). Training uses cross entropy (§9) on the next token.', ['s8', 's9']],
  ];
  const wrap = H('div', { class: 'tf' }, body);
  const col = H('div', { class: 'tf-stack' }, wrap);
  const info = H('div', { class: 'tf-info' }, wrap);
  const loopBox = H('div', { class: 'tf-loop' }, col);
  H('div', { class: 'tf-x' }, loopBox, '× 24 blocks');
  const els = blocks.map(([k, name]) => {
    const parent = ['att', 'res1', 'norm', 'ffn', 'res2'].includes(k) ? loopBox : col;
    const b = H('button', { type: 'button', class: `tf-b tf-${k}` }, parent, name);
    return b;
  });
  col.insertBefore(els[0], loopBox); col.insertBefore(els[1], loopBox); col.insertBefore(els[2], loopBox);
  col.appendChild(els[8]);
  const pick = i => {
    els.forEach((b, j) => b.classList.toggle('on', i === j));
    const [, name, desc, refs] = blocks[i];
    info.innerHTML = `<h4>${name}</h4><p>${desc}</p><p class="tk">Builds on: ${refs.map(r => `<a class="ref" href="#${r}">§${r.slice(1)}</a>`).join(' ')}</p>`;
    wireRefs(info);
  };
  els.forEach((b, i) => b.addEventListener('click', () => pick(i)));
  pick(3);
};

FIGS.selfattn = body => {
  const toks = ['What', 'is', 'StatQuest', '&lt;EOS&gt;'];
  const E = [[1.0, 0.2], [0.2, 1.0], [0.9, 0.8], [-0.5, 0.4]];
  const Wq = [[0.9, 0.3], [-0.2, 1.1]], Wk = [[1.0, -0.4], [0.5, 0.8]], Wv = [[0.7, 0.6], [-0.3, 0.9]];
  const mul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
  const Q = mul(E, Wq), K = mul(E, Wk), Vv = mul(E, Wv);
  let masked = false, scaled = true, hover = 2;
  const wrap = H('div', { class: 'sa' }, body);
  const ro = readout(body);
  const ctl = controls(body);
  const table = (title, M, opts = {}) => {
    const t = H('table', { class: 'mtx' }, wrap);
    t.innerHTML = `<caption>${title}</caption>` + (opts.head ? '<tr><th></th>' + opts.head.map(h => `<th>${h}</th>`).join('') + '</tr>' : '')
      + M.map((r, i) => `<tr data-r="${i}" class="${i === hover ? 'hl' : ''}"><th>${toks[i]}</th>` + r.map(v => v === -Infinity
        ? '<td class="masked">−∞</td>'
        : `<td style="background:${heat(v, opts.scale || 2)}">${fmt(v)}</td>`).join('') + '</tr>').join('');
    t.querySelectorAll('tr[data-r]').forEach(tr => tr.addEventListener('mouseenter', () => { hover = +tr.dataset.r; draw(); }));
  };
  const draw = () => {
    const d = Math.sqrt(2);
    const sc = Q.map((q, i) => K.map((k, j) => (masked && j > i ? -Infinity : (q[0] * k[0] + q[1] * k[1]) / (scaled ? d : 1))));
    const W = sc.map(softmax);
    const O = W.map(r => [0, 1].map(c => r.reduce((s, w, j) => s + w * Vv[j][c], 0)));
    wrap.innerHTML = '';
    table('queries Q = X·Wq', Q, { head: ['1', '2'] });
    table('keys K = X·Wk', K, { head: ['1', '2'] });
    table('values V = X·Wv', Vv, { head: ['1', '2'] });
    table(`scores Q·Kᵀ${scaled ? ' ÷ √2' : ''}${masked ? ', masked' : ''}`, sc, { head: toks, scale: 1.5 });
    table('SoftMax of each row', W, { head: toks, scale: 1 });
    table('output = shares · V', O, { head: ['1', '2'] });
    const w = W[hover];
    ro.innerHTML = `Row <b>${toks[hover]}</b>: its query is compared with every key; SoftMax turns the row into shares ${w.map((v, j) => `${toks[j]} ${fmt(v * 100, 0)}%`).join(', ')}; the output is that mix of value rows. Hover a row.`;
  };
  button(ctl, 'mask the future (decoder)', e => { masked = !masked; e.target.classList.toggle('on', masked); draw(); });
  const sb = button(ctl, 'divide by √(list length)', e => { scaled = !scaled; e.target.classList.toggle('on', scaled); draw(); });
  sb.classList.add('on');
  draw();
};

FIGS.generate = body => {
  const prompt = ['What', 'is', 'StatQuest', '?'];
  const gen = ['Awesome', '&lt;EOS&gt;'];
  const cands = [['Awesome', 'Great', 'A', 'It'], ['&lt;EOS&gt;', '!', 'and', '.']];
  const probs = [[0.71, 0.14, 0.09, 0.06], [0.83, 0.09, 0.05, 0.03]];
  const wrap = H('div', { class: 'gen' }, body);
  const strip = H('div', { class: 'tokstrip' }, wrap);
  const grid = H('div', { class: 'tri' }, wrap);
  const barsEl = H('div', { class: 'bars' }, wrap);
  const ro = readout(body);
  const steps = [
    [0, 'The prompt is read all at once. Each token attends to itself and to tokens before it, never after (the mask from §19).'],
    [1, 'The last position\'s output goes through the unembedding and SoftMax (§8). "Awesome" has the highest probability (the numbers here are illustrative), so it is appended.'],
    [2, '"Awesome" is fed back in as a new token. Only the new row of the attention grid is new; the earlier rows are unchanged, which is why the model can cache them.'],
    [3, 'The model predicts "<EOS>", the stop token. The answer is "Awesome". The same network both reads the prompt and writes the answer; there is no separate encoder.'],
  ];
  stepper(body, steps.length, i => {
    const toks = [...prompt, ...gen.slice(0, Math.min(i, 2))];
    strip.innerHTML = toks.map((t, k) => `<span class="tok ${k >= prompt.length ? 'gen' : ''}">${t}</span>`).join('');
    const n = toks.length;
    grid.style.gridTemplateColumns = `repeat(${n},1fr)`;
    grid.innerHTML = '';
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const cell = H('div', { class: `tc ${c > r ? 'off' : 'on'} ${r === n - 1 && i > 0 ? 'last' : ''}` }, grid);
      if (c > r) cell.textContent = '×';
    }
    const k = i === 0 ? -1 : Math.min(i - 1, 1);
    barsEl.innerHTML = k < 0 ? '<p class="tk">no prediction yet</p>' : cands[k].map((w, j) => `<div class="bar"><span>${w}</span><i style="width:${probs[k][j] * 100}%"></i><b>${fmt(probs[k][j] * 100, 0)}%</b></div>`).join('');
    ro.innerHTML = steps[i][1];
  });
};

FIGS.rag = body => {
  const docs = [
    ['Pizza is great', 0.25], ['Pizza is awesome', 0.4], ['I love pasta', 0.9],
    ['The taxiway is closed', 2.3], ['Runway lights are on', 2.7], ['Rain expected tonight', 4.2],
  ];
  let q = [0.9, 0.5];
  const svg = svgBox(body, 680, 340);
  const F = frame(svg, 180, 10, 320, 320, [-1.4, 1.4], [-1.4, 1.4]);
  S('circle', { cx: F.X(0), cy: F.Y(0), r: F.X(1) - F.X(0), fill: 'none', stroke: C.faint }, F.g);
  const arrows = docs.map(([t, a], i) => {
    const ar = makeArrow(F.g, C.blue, 2);
    ar.set(F.X(0), F.Y(0), F.X(Math.cos(a)), F.Y(Math.sin(a)));
    const lx = Math.cos(a) * 1.08, ly = Math.sin(a) * 1.08;
    text(F.g, F.X(lx), F.Y(ly) + 4, t, { class: 'tk', 'text-anchor': lx < -0.1 ? 'end' : 'start' });
    return ar;
  });
  const qa = makeArrow(F.g, C.yellow, 3.5);
  const side = H('div', { class: 'rag-list' }, body);
  const ro = readout(body);
  const draw = () => {
    const qa0 = Math.atan2(q[1], q[0]);
    qa.set(F.X(0), F.Y(0), F.X(q[0]), F.Y(q[1]));
    const sims = docs.map(([t, a]) => [t, Math.cos(a - qa0)]).sort((x, y) => y[1] - x[1]);
    docs.forEach(([t], i) => arrows[i].el.style.opacity = t === sims[0][0] ? 1 : 0.4);
    side.innerHTML = sims.map(([t, s], i) => `<div class="bar ${i === 0 ? 'top' : ''}"><span>${t}</span><i style="width:${Math.max(0, s) * 100}%"></i><b>${fmt(s, 2)}</b></div>`).join('');
    ro.innerHTML = `Cosine similarity only looks at the angle: 1 means the same direction, 0 means unrelated, −1 means opposite. The top match, "<b>${sims[0][0]}</b>", would be pasted into the prompt of a decoder (§20). Drag the <span class="c-yellow">question</span>.`;
  };
  dragArea(svg, pt => {
    const v = [F.iX(pt.x), F.iY(pt.y)], L = Math.hypot(...v) || 1;
    q = [v[0] / L, v[1] / L]; draw();
  });
  draw();
};

FIGS.fries = body => {
  let b, log, hist, r;
  const reset = () => { b = 0; log = []; hist = [0]; r = rng(3); };
  const P = h => sig(20 * h + b);
  const svg = svgBox(body, 680, 290);
  const F = frame(svg, 50, 20, 270, 220, [0, 1], [0, 1]);
  axes(F, { xlabel: 'hunger', ylabel: 'chance of the large-portion shop', xticks: [0, 0.5, 1], yticks: [0, 0.5, 1] });
  S('line', { x1: F.X(0.5), y1: F.y, x2: F.X(0.5), y2: F.y + F.h, stroke: C.green, 'stroke-dasharray': '4 4' }, F.g);
  const pc = curve(F, null, { stroke: C.yellow, 'stroke-width': 3 });
  const dots = S('g', {}, F.g);
  const B = frame(svg, 390, 20, 260, 220, [0, 1], [-14, 2]);
  axes(B, { xlabel: 'trips', ylabel: 'bias', yticks: [-12, -8, -4, 0] });
  const bp = S('path', { fill: 'none', stroke: C.purple, 'stroke-width': 2 }, B.g);
  const ro = readout(body);
  const ctl = controls(body);
  const trip = () => {
    const h = r(), p = P(h), large = r() < p;
    const right = (h > 0.5) === large, reward = right ? 1 : -1;
    const d = large ? p - 1 : p;
    const upd = d * reward;
    const old = b;
    b -= 1.0 * upd;
    hist.push(b);
    log.unshift({ h, p, large, reward, d, upd, old });
    if (log.length > 14) log.pop();
  };
  const draw = () => {
    pc.set(P);
    dots.innerHTML = '';
    log.forEach((t, i) => S('circle', { cx: F.X(t.h), cy: F.Y(t.large ? 1 : 0), r: 4, fill: t.reward > 0 ? C.green : C.red, opacity: 1 - i / 15 }, dots));
    bp.setAttribute('d', pathD(hist.map((v, i) => [B.X(i / Math.max(1, hist.length - 1)), B.Y(clamp(v, -14, 2))])));
    const t = log[0];
    ro.innerHTML = t
      ? `trip ${hist.length - 1}: hunger ${fmt(t.h)}, chance of large ${fmt(t.p)}, went to the <b>${t.large ? 'large' : 'small'}</b>-portion shop, which was <b class="${t.reward > 0 ? 'c-green' : 'c-red'}">${t.reward > 0 ? 'right' : 'wrong'}</b> (reward ${t.reward > 0 ? '+1' : '−1'}). Slope as if it were right: ${fmt(t.d, 3)}; × reward = ${fmt(t.upd, 3)}; bias ${fmt(t.old, 2)} → <b>${fmt(b, 2)}</b>.`
      : 'Bias 0: even slightly hungry people almost always go to the large-portion shop. Press "one trip".';
  };
  const many = loop(() => { for (let s = 0; s < 3; s++) trip(); draw(); return hist.length < 301; });
  button(ctl, 'one trip', () => { trip(); draw(); }, 'primary');
  button(ctl, '300 trips', () => many.start());
  button(ctl, 'reset', () => { many.stop(); reset(); draw(); });
  reset(); draw();
};

FIGS.rlstep = body => {
  let p = 0.5, went = 'small', reward = 1, bias = 0;
  const ro = readout(body);
  const ctl = controls(body);
  const draw = () => {
    const d = went === 'large' ? p - 1 : p;
    const upd = d * reward;
    ro.innerHTML = `<div class="calc">
      <div>1. The network said the chance of the large-portion shop was <b>${fmt(p)}</b>. We went to the <b>${went}</b>-portion shop.</div>
      <div>2. Pretend that choice was correct. Cross entropy's slope with respect to the bias is ${went === 'large' ? 'p − 1' : 'p'} = <b>${fmt(d)}</b> (from §9 and the chain rule, §4).</div>
      <div>3. Find out how it went: reward <b>${reward > 0 ? '+' : ''}${reward}</b>. Multiply: ${fmt(d)} × ${reward} = <b class="c-yellow">${fmt(upd)}</b>.</div>
      <div>4. Gradient descent step (§5) with learning rate 1: bias ${fmt(bias)} − ${fmt(upd)} = <b>${fmt(bias - upd)}</b>.</div></div>`;
  };
  slider(ctl, 'chance of large', 0.01, 0.99, 0.01, p, v => { p = v; draw(); });
  select(ctl, 'went to', [['small', 'small-portion shop'], ['large', 'large-portion shop']], went, v => { went = v; draw(); });
  select(ctl, 'reward', [['1', '+1 (good)'], ['-1', '−1 (bad)'], ['2', '+2'], ['-2', '−2']], '1', v => { reward = +v; draw(); });
  slider(ctl, 'bias now', -5, 5, 0.1, bias, v => { bias = v; draw(); }, 1);
  draw();
};

FIGS.rlhf = body => {
  const stages = [
    ['Pre-train', 'Next-token prediction (§20) on a huge pile of text. The model learns language and facts, but it only continues text; it does not answer.'],
    ['Supervised fine-tune', 'Train the same way on a small set of prompts paired with good answers written by people. Small sets are easy to overfit, so this step is short.'],
    ['Collect preferences', 'Sample two answers to one prompt (sampling from SoftMax, not always taking the top token, gives different answers). A person picks the better one.'],
    ['Train a reward model', 'Copy the fine-tuned model and replace the unembedding (§18) with a single output: a score. Train it so the preferred answer scores higher. Loss = −log(sigmoid(score of better − score of worse)).'],
    ['Reinforcement learning', 'The fine-tuned model answers new prompts, the reward model scores them, and that score scales the training slope (§22, §23). No person is needed in this loop.'],
  ];
  const pipe = H('div', { class: 'pipe' }, body);
  const els = stages.map(([n]) => H('div', { class: 'pipe-s' }, pipe, n));
  const cap = readout(body);
  stepper(body, stages.length, i => {
    els.forEach((e, j) => { e.classList.toggle('on', j === i); e.classList.toggle('done', j < i); });
    cap.innerHTML = stages[i][1];
  });
  const svg = svgBox(body, 680, 230);
  const F = frame(svg, 60, 15, 560, 180, [-5, 5], [0, 5.5]);
  axes(F, { xlabel: 'score of preferred − score of other', ylabel: 'reward-model loss', xticks: [-4, -2, 0, 2, 4], yticks: [0, 1, 2, 3, 4, 5], zero: false });
  curve(F, g => -Math.log(sig(g)), { stroke: C.red });
  const dot = S('circle', { r: 7, fill: C.yellow }, F.g);
  const ro = readout(body);
  const ctl = controls(body);
  const draw = gap => {
    const l = -Math.log(sig(gap));
    dot.setAttribute('cx', F.X(gap)); dot.setAttribute('cy', F.Y(Math.min(l, 5.5)));
    ro.innerHTML = `gap ${fmt(gap)} → sigmoid ${fmt(sig(gap), 3)} (the chance the reward model agrees with the person) → loss <b>${fmt(l, 3)}</b>. A large positive gap costs almost nothing; a wrong ordering costs a lot.`;
  };
  slider(ctl, 'score gap', -5, 5, 0.01, 0.5, draw);
  draw(0.5);
};

FIGS.trainloop = body => {
  const code = [
    ['for epoch in range(epochs):', ''],
    ['    for x, y in loader:', 'data'],
    ['        pred = model(x)', 'fwd'],
    ['        loss = loss_fn(pred, y)', 'loss'],
    ['        loss.backward()', 'back'],
    ['        optimizer.step()', 'step'],
    ['        optimizer.zero_grad()', 'zero'],
  ];
  const steps = [
    ['data', 'Get a batch', 'Take a few training examples: inputs and the answers we want (§16: a batch is a tensor with one more direction).'],
    ['fwd', 'Forward pass', 'Run the inputs through the network (§2, §7). PyTorch records every operation it did.'],
    ['loss', 'Measure the miss', 'One number for how wrong the prediction is: squared misses (§5) or cross entropy (§9).'],
    ['back', 'Backward pass', 'PyTorch walks the recorded operations in reverse and applies the chain rule (§4, §6). Every weight now holds its slope in .grad.'],
    ['step', 'Step', 'Move every weight a little downhill: weight −= learning rate × slope (§5). Adam, the usual choice, also keeps a running average of each weight\'s slopes.'],
    ['zero', 'Clear the slopes', 'PyTorch adds new slopes onto old ones, so they are cleared before the next batch. Lightning writes this whole loop for you: you write training_step and configure_optimizers.'],
  ];
  const wrap = H('div', { class: 'tl' }, body);
  const pre = H('pre', { class: 'code' }, wrap);
  const lines = code.map(([t, k]) => { const l = H('span', { class: 'cl' }, pre, t + '\n'); l.dataset.k = k; return l; });
  const ring = H('div', { class: 'ring' }, wrap);
  const nodes = steps.map(([k, n], i) => {
    const a = (i / steps.length) * 2 * Math.PI - Math.PI / 2;
    const e = H('div', { class: 'ring-n', style: `left:${50 + 38 * Math.cos(a)}%;top:${50 + 38 * Math.sin(a)}%` }, ring, n);
    e.dataset.k = k;
    return e;
  });
  const ro = readout(body);
  const st = stepper(body, steps.length, i => {
    const k = steps[i][0];
    lines.forEach(l => l.classList.toggle('on', l.dataset.k === k));
    nodes.forEach(n => n.classList.toggle('on', n.dataset.k === k));
    ro.innerHTML = `<b>${steps[i][1]}.</b> ${steps[i][2]}`;
  });
  nodes.forEach((n, i) => n.addEventListener('click', () => st.go(i)));
  wireRefs(body);
};

FIGS.codemap = body => {
  const code = [
    ['class Block(nn.Module):', ''],
    ['    def __init__(self, d, heads):', ''],
    ['        self.norm1 = nn.RMSNorm(d)', 'norm'],
    ['        self.attn = nn.MultiheadAttention(d, heads)', 'att'],
    ['        self.norm2 = nn.RMSNorm(d)', 'norm'],
    ['        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.SiLU(), nn.Linear(4*d, d))', 'ffn'],
    ['    def forward(self, x, mask):', ''],
    ['        h = self.norm1(x)', 'norm'],
    ['        x = x + self.attn(h, h, h, attn_mask=mask)[0]', 'att'],
    ['        x = x + self.ffn(self.norm2(x))', 'ffn'],
    ['        return x', ''],
    ['', ''],
    ['class TinyGPT(nn.Module):', ''],
    ['    def __init__(self, vocab, d, heads, layers):', ''],
    ['        self.embed = nn.Embedding(vocab, d)', 'emb'],
    ['        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))', 'loop'],
    ['        self.unembed = nn.Linear(d, vocab)', 'out'],
    ['    def forward(self, ids):', ''],
    ['        x = self.embed(ids) + positions(ids)', 'pos'],
    ['        mask = torch.triu(torch.ones(n, n), 1).bool()', 'mask'],
    ['        for b in self.blocks: x = b(x, mask)', 'loop'],
    ['        return self.unembed(x)', 'out'],
    ['', ''],
    ['loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())', 'loss'],
  ];
  const notes = {
    emb: ['Word embedding: a lookup table of rows', 's13'], pos: ['Position information', 's18'],
    norm: ['Normalise each token\'s list', 's18'], att: ['Self-attention with query, key, value', 's19'],
    ffn: ['Feed-forward network: widen, bend, narrow', 's7'], loop: ['Stack the same block many times', 's18'],
    out: ['Unembed: score every vocabulary entry', 's8'], mask: ['Mask: no peeking at later tokens', 's20'],
    loss: ['Cross entropy on the next token: inputs shifted by one', 's9'],
  };
  const wrap = H('div', { class: 'tl one' }, body);
  const pre = H('pre', { class: 'code small' }, wrap);
  const note = H('div', { class: 'tf-info' }, wrap);
  const lines = code.map(([t, k]) => { const l = H('span', { class: 'cl' }, pre, t + '\n'); l.dataset.k = k; return l; });
  const show = k => {
    lines.forEach(l => l.classList.toggle('on', l.dataset.k === k));
    if (!notes[k]) return;
    note.innerHTML = `<h4>${notes[k][0]}</h4><p class="tk">Explained in <a class="ref" href="#${notes[k][1]}">§${notes[k][1].slice(1)}</a></p>`;
    wireRefs(note);
  };
  lines.forEach(l => { if (l.dataset.k) { l.classList.add('hov'); l.addEventListener('mouseenter', () => show(l.dataset.k)); l.addEventListener('click', () => show(l.dataset.k)); } });
  show('att');
};

const LANES = '0.400 1.000 0.462 0.330 0.190 0.040\n0.400 1.000 0.488 0.440 0.390 0.345\n0.400 1.000 0.512 0.560 0.612 0.660\n0.400 1.000 0.538 0.672 0.815 0.965';
const parseLanes = txt => {
  const rows = [];
  for (const line of txt.trim().split('\n')) {
    if (!line.trim()) continue;
    const v = line.trim().split(/\s+/).map(Number);
    if (v.length !== 6 || v.some(x => !Number.isFinite(x)) || !(v[0] >= 0 && v[0] < v[1] && v[1] <= 1)) return { error: line };
    rows.push(v);
  }
  return rows.length ? { rows } : { error: '(empty)' };
};
const laneX = (row, y) => {
  const ys = [0, 1, 2, 3].map(i => lerp(row[0], row[1], i / 3)), xs = row.slice(2);
  let s = 0;
  for (let i = 0; i < 4; i++) {
    let t = xs[i];
    for (let j = 0; j < 4; j++) if (j !== i) t *= (y - ys[j]) / (ys[i] - ys[j]);
    s += t;
  }
  return s;
};
function drawRoad(ctx, W, Hh, rows, paint = true) {
  const sky = ctx.createLinearGradient(0, 0, 0, Hh * 0.4);
  sky.addColorStop(0, '#5b7593'); sky.addColorStop(1, '#a9b8c4');
  ctx.fillStyle = sky; ctx.fillRect(0, 0, W, Hh * 0.4);
  ctx.fillStyle = '#4a5a3a'; ctx.fillRect(0, Hh * 0.4, W, Hh * 0.6);
  ctx.fillStyle = '#6c7a86';
  for (let i = 0; i < 9; i++) ctx.fillRect(W * (0.05 + i * 0.11), Hh * (0.33 - (i % 3) * 0.02), W * 0.06, Hh * (0.07 + (i % 3) * 0.02));
  const L = rows[0], R = rows[rows.length - 1];
  ctx.beginPath();
  for (let k = 0; k <= 30; k++) { const y = lerp(L[0], L[1], k / 30); ctx.lineTo(laneX(L, y) * W, y * Hh); }
  for (let k = 30; k >= 0; k--) { const y = lerp(R[0], R[1], k / 30); ctx.lineTo(laneX(R, y) * W, y * Hh); }
  ctx.closePath(); ctx.fillStyle = '#3b3d42'; ctx.fill();
  if (!paint) return;
  rows.forEach((row, i) => {
    ctx.strokeStyle = i === 0 || i === rows.length - 1 ? '#f0f0f0' : '#f2d05a';
    ctx.setLineDash(i === 0 || i === rows.length - 1 ? [] : [W * 0.03, W * 0.025]);
    ctx.lineWidth = Math.max(1.5, W * 0.006);
    ctx.beginPath();
    for (let k = 0; k <= 40; k++) { const y = lerp(row[0], row[1], k / 40); ctx.lineTo(laneX(row, y) * W, y * Hh); }
    ctx.stroke();
  });
  ctx.setLineDash([]);
}
const ROWS = parseLanes(LANES).rows;
const canvas = (box, w, h, cls = '') => {
  const c = H('canvas', { width: w, height: h, class: `fig-canvas ${cls}` }, box);
  return [c, c.getContext('2d')];
};

FIGS.patches = body => {
  const wrap = H('div', { class: 'patch' }, body);
  const [cv, ctx] = canvas(wrap, 378, 378);
  const side = H('div', { class: 'patch-side' }, wrap);
  const [zc, zx] = canvas(side, 140, 140, 'pix');
  const info = H('div', { class: 'tk' }, side);
  const ro = readout(body);
  const ctl = controls(body);
  let size = 14, at = [5, 21];
  const base = document.createElement('canvas'); base.width = 378; base.height = 378;
  drawRoad(base.getContext('2d'), 378, 378, ROWS);
  const draw = () => {
    ctx.drawImage(base, 0, 0);
    ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1;
    for (let p = 0; p <= 378; p += size) { ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 378); ctx.moveTo(0, p); ctx.lineTo(378, p); ctx.stroke(); }
    const n = Math.floor(378 / size);
    at = [clamp(at[0], 0, n - 1), clamp(at[1], 0, n - 1)];
    ctx.strokeStyle = C.yellow; ctx.lineWidth = 2.5; ctx.strokeRect(at[0] * size, at[1] * size, size, size);
    zx.imageSmoothingEnabled = false;
    zx.drawImage(base, at[0] * size, at[1] * size, size, size, 0, 0, 140, 140);
    const px = base.getContext('2d').getImageData(at[0] * size, at[1] * size, 1, 1).data;
    info.innerHTML = `patch (${at[0]}, ${at[1]}): ${size}×${size} pixels × 3 colours = <b>${size * size * 3}</b> numbers.<br>Top-left pixel: red ${px[0]}, green ${px[1]}, blue ${px[2]}.`;
    ro.innerHTML = `A ${n * size}-pixel square cut into ${size}-pixel patches gives ${n} × ${n} = <b>${comma(n * n)}</b> patches. Each patch is flattened to one list and multiplied by a table to make one vision token${size === 14 ? ' of 1,152 numbers: this is what SmolVLM2\'s vision tower does on each 384-pixel tile (378 used here so the grid divides evenly)' : ''}. Click a patch.`;
  };
  cv.addEventListener('click', e => {
    const r = cv.getBoundingClientRect();
    at = [Math.floor(((e.clientX - r.left) / r.width) * 378 / size), Math.floor(((e.clientY - r.top) / r.height) * 378 / size)];
    draw();
  });
  slider(ctl, 'patch size (pixels)', 14, 126, 14, size, v => { size = v; draw(); }, 0);
  draw();
};

const TILE_GRID = { 1: [1, 1, 384], 5: [2, 2, 768], 7: [3, 2, 1152], 13: [4, 3, 1536], 16: [5, 3, 1920], 25: [6, 4, 2304], 29: [7, 4, 2688], 41: [8, 5, 3072] };
FIGS.tiles = body => {
  const wrap = H('div', { class: 'tiles' }, body);
  const [cv, ctx] = canvas(wrap, 560, 330);
  const ro = readout(body);
  const ctl = controls(body);
  const photo = document.createElement('canvas'); photo.width = 1276; photo.height = 717;
  drawRoad(photo.getContext('2d'), 1276, 717, ROWS);
  const draw = n => {
    const [cols, rows, edge] = TILE_GRID[n];
    ctx.fillStyle = '#101014'; ctx.fillRect(0, 0, 560, 330);
    const cell = Math.min(400 / cols, 300 / rows) | 0;
    if (n === 1) {
      ctx.drawImage(photo, 20, 60, 384, 216);
      ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.strokeRect(20, 60, 384, 216);
    } else {
      ctx.drawImage(photo, 10, 10, cols * cell, rows * cell);
      ctx.strokeStyle = C.yellow; ctx.lineWidth = 2;
      for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) ctx.strokeRect(10 + i * cell + 1.5, 10 + j * cell + 1.5, cell - 3, cell - 3);
      ctx.drawImage(photo, 440, 10, 110, 110);
      ctx.strokeStyle = C.blue; ctx.strokeRect(440, 10, 110, 110);
      ctx.fillStyle = C.dim; ctx.font = '12px sans-serif'; ctx.fillText('+ whole photo', 440, 138); ctx.fillText('squashed to one tile', 440, 154);
    }
    const h = Math.round((717 * edge) / 1276);
    ro.innerHTML = n === 1
      ? 'One tile: the photo is shrunk to 384 pixels wide and seen as one picture. <b>81</b> image tokens. Lane markings far away become a pixel or two.'
      : `Long side resized to ${comma(edge)} pixels (${comma(edge)} × ${comma(h)}), then the short side is stretched up to ${comma(rows * 384)} so the photo divides into whole tiles: ${cols} × ${rows} tiles of 384 × 384, plus one whole-photo tile. <b>${n}</b> tiles × 81 = <b>${comma(n * 81)}</b> image tokens.${n === 13 ? ' This is the experiment\'s default.' : ''}`;
  };
  segmented(ctl, Object.keys(TILE_GRID).map(k => [k, `${k} tile${k === '1' ? '' : 's'}`]), '13', v => draw(+v));
  draw(13);
};

FIGS.shuffle = body => {
  const svg = svgBox(body, 680, 300);
  const g = S('g', {}, svg);
  const cs = 10, ox = 20, oy = 20;
  const cells = [];
  for (let i = 0; i < 27; i++) for (let j = 0; j < 27; j++)
    cells.push(S('rect', { x: ox + j * cs, y: oy + i * cs, width: cs - 1, height: cs - 1, fill: '#24323a' }, g));
  text(g, ox, 296, '27 × 27 = 729 patch lists, 1,152 numbers each', { class: 'tk' });
  const tx = 470, ts = 22;
  const toks = [];
  for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++)
    toks.push(S('rect', { x: tx + j * ts, y: oy + i * ts * 1.2, width: ts - 3, height: ts * 1.2 - 3, rx: 3, fill: '#2b2b20' }, g));
  text(g, tx, 296, '9 × 9 = 81 tokens, 2,048 numbers each', { class: 'tk' });
  const ar = makeArrow(g, C.yellow, 2.5);
  const ro = readout(body);
  let at = [4, 4];
  const draw = () => {
    cells.forEach((c, k) => {
      const i = Math.floor(k / 27), j = k % 27;
      c.setAttribute('fill', Math.floor(i / 3) === at[0] && Math.floor(j / 3) === at[1] ? C.blue : '#24323a');
    });
    toks.forEach((t, k) => t.setAttribute('fill', Math.floor(k / 9) === at[0] && k % 9 === at[1] ? C.yellow : '#2b2b20'));
    ar.set(ox + (at[1] * 3 + 3) * cs + 4, oy + (at[0] * 3 + 1.5) * cs, tx + at[1] * ts - 4, oy + (at[0] + 0.5) * ts * 1.2);
    ro.innerHTML = `A 3 × 3 block of neighbouring patches is laid end to end: 9 × 1,152 = <b>10,368</b> numbers. The connector, one 2,048 × 10,368 table, turns that into one token of 2,048 numbers, the size of a text token. Point at a block.`;
  };
  svg.addEventListener('pointermove', e => {
    const p = svgPoint(svg, e);
    if (p.x > ox && p.x < ox + 27 * cs && p.y > oy && p.y < oy + 27 * cs) { at = [Math.floor((p.y - oy) / (3 * cs)), Math.floor((p.x - ox) / (3 * cs))]; draw(); }
    if (p.x > tx && p.x < tx + 9 * ts && p.y > oy && p.y < oy + 9 * ts * 1.2) { at = [Math.floor((p.y - oy) / (ts * 1.2)), Math.floor((p.x - tx) / ts)]; draw(); }
  });
  draw();
};

FIGS.mask = body => {
  const toks = [
    ['<|im_start|>', 'p'], ['User:', 'p'], ['<image> × 1,053 + tile markers', 'img'], ['Generate', 'p'], [' a', 'p'], [' list', 'p'], [' of', 'p'], [' spl', 'p'], ['ines', 'p'],
    [' for', 'p'], [' all', 'p'], [' road', 'p'], [' and', 'p'], [' taxi', 'p'], [' lines', 'p'], [' in', 'p'], [' this', 'p'], [' image', 'p'],
    ['<end_of_utterance>', 'p'], ['Assistant:', 'p'], [' 0', 'a'], ['.', 'a'], ['400', 'a'], [' 1', 'a'], ['.', 'a'], ['000', 'a'], [' 0', 'a'],
    ['.', 'a'], ['462', 'a'], [' …', 'a'], ['<end_of_utterance>', 'a'],
  ];
  const strip = H('div', { class: 'tokstrip wrap' }, body);
  const ro = readout(body);
  const ctl = controls(body);
  const draw = show => {
    strip.innerHTML = toks.map(([t, k]) => `<span class="tok ${k}"><span>${t.replace(/</g, '&lt;')}</span>${show ? `<small>${k === 'a' ? 'scored' : '−100'}</small>` : ''}</span>`).join('');
    ro.innerHTML = show
      ? 'Every prompt and image position gets the label −100, which PyTorch\'s cross entropy (§9) skips. Only the answer tokens count toward the loss. The model is never asked to predict the photo or the question.'
      : 'One training example as the model sees it. The splitting of words into tokens is approximate here; the real tokenizer chooses its own pieces.';
  };
  segmented(ctl, [['0', 'tokens'], ['1', 'labels used for the loss']], '1', v => draw(v === '1'));
  draw(true);
};

FIGS.params = body => {
  const parts = [
    ['Vision tower (27 blocks)', 412987248, C.teal], ['Connector', 21233664, C.orange],
    ['Text model (24 blocks + embedding)', 1711638528, C.blue], ['Unembedding (lm_head)', 100925440, C.purple],
    ['LoRA adapters (r = 16)', 6291456, C.yellow],
  ];
  const total = 2246784880;
  const list = H('div', { class: 'rag-list' }, body);
  list.innerHTML = parts.map(([n, v, c]) => `<div class="bar"><span>${n}</span><i style="width:${Math.max(0.4, (v / 1711638528) * 100)}%;background:${c}"></i><b>${comma(v)}</b></div>`).join('');
  readout(body).innerHTML = `Frozen model total: <b>${comma(total)}</b> numbers. LoRA trains ${comma(6291456)} of its own, ${fmt((6291456 / total) * 100, 2)}% as many. Counted from the model's own weight file.`;
};

FIGS.lora = body => {
  const d = 32;
  let r = 16, trained = true;
  const rg = rng(11);
  const W = Array.from({ length: d }, () => Array.from({ length: d }, () => gauss(rg) * 0.6));
  const A0 = Array.from({ length: 64 }, () => Array.from({ length: d }, () => gauss(rg) * 0.4));
  const B1 = Array.from({ length: d }, () => Array.from({ length: 64 }, () => gauss(rg) * 0.4));
  const wrap = H('div', { class: 'lora' }, body);
  const panel = (title, w, h) => {
    const box = H('div', { class: 'lora-p' }, wrap);
    H('div', { class: 'cnn-t' }, box, title);
    const [cv, ctx] = canvas(box, w, h);
    return { box, cv, ctx };
  };
  const pW = panel('frozen table W', 160, 160);
  H('div', { class: 'op' }, wrap, '+');
  const pB = panel('B (tall, thin)', 80, 160);
  H('div', { class: 'op' }, wrap, '×');
  const pA = panel('A (short, wide)', 160, 80);
  H('div', { class: 'op' }, wrap, '=');
  const pD = panel('correction B·A', 160, 160);
  const ro = readout(body);
  const ctl = controls(body);
  const paint = (p, M, cw, ch, sc) => {
    const rows = M.length, cols = M[0].length;
    p.cv.width = cols * cw; p.cv.height = rows * ch;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { p.ctx.fillStyle = heat(M[i][j], sc); p.ctx.fillRect(j * cw, i * ch, cw - 0.5, ch - 0.5); }
  };
  const draw = () => {
    const vr = Math.min(r, 32);
    const A = A0.slice(0, vr);
    const B = B1.map(row => row.slice(0, vr).map(v => (trained ? v : 0)));
    const D = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (32 / r) * B[i].reduce((s, b, k) => s + b * A[k][j], 0) * 0.15));
    paint(pW, W, 5, 5, 1.2);
    paint(pB, B, Math.max(2, Math.floor(80 / vr)), 5, 0.8);
    paint(pA, A, 5, Math.max(2, Math.floor(80 / vr)), 0.8);
    paint(pD, D, 5, 5, 1.2);
    const per = r * (2048 + 2048);
    ro.innerHTML = `Rank ${r}: each SmolVLM2 attention table is 2,048 × 2,048 = 4,194,304 frozen numbers. B is 2,048 × ${r} and A is ${r} × 2,048: <b>${comma(per)}</b> trainable numbers, ${fmt((per / 4194304) * 100, 2)}% of the table. Across 96 tables: <b>${comma(per * 96)}</b>. The correction is multiplied by alpha ÷ r = 32 ÷ ${r} = ${fmt(32 / r)}.${trained ? '' : ' B starts as all zeros, so before training the correction is exactly zero and the model is unchanged.'} (Drawn here at 32 × 32 so the cells are visible.)`;
  };
  slider(ctl, 'rank r', 1, 64, 1, r, v => { r = v; draw(); }, 0);
  segmented(ctl, [['0', 'at the start of training'], ['1', 'after some training']], '1', v => { trained = v === '1'; draw(); });
  draw();
};

FIGS.lorawhere = body => {
  const svg = svgBox(body, 680, 300);
  const g = S('g', {}, svg);
  const names = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
  const sizes = ['2,048 × 2,048', '2,048 × 2,048', '2,048 × 2,048', '2,048 × 2,048', '8,192 × 2,048', '8,192 × 2,048', '2,048 × 8,192'];
  text(g, 20, 20, 'vision tower: 27 blocks, no adapters (names contain "vision")', { class: 'tk' });
  for (let i = 0; i < 27; i++) S('rect', { x: 20 + i * 23, y: 30, width: 19, height: 22, rx: 3, fill: '#22262b', stroke: C.faint }, g);
  text(g, 20, 84, 'text model: 24 blocks × 7 tables. Lit tables get an adapter.', { class: 'tk' });
  const cells = [];
  names.forEach((n, j) => text(g, 14, 112 + j * 24, n, { class: 'tk', 'text-anchor': 'start' }));
  for (let i = 0; i < 24; i++) for (let j = 0; j < 7; j++) {
    const lit = j < 4;
    const c = S('rect', { x: 60 + i * 25, y: 98 + j * 24, width: 21, height: 20, rx: 3, fill: lit ? C.yellow : '#23232b', opacity: lit ? 0.85 : 1 }, g);
    c.dataset.i = i; c.dataset.j = j;
    cells.push(c);
  }
  const ro = readout(body);
  const show = (i, j) => {
    cells.forEach(c => c.setAttribute('stroke', +c.dataset.i === i && +c.dataset.j === j ? C.ink : 'none'));
    ro.innerHTML = `Block ${i + 1}, <b>${names[j]}_proj</b>, ${sizes[j]}: ${j < 4 ? `adapter B·A with rank 16 adds ${comma(16 * 4096)} trainable numbers.` : 'frozen, no adapter. The settings target only names ending in q_proj, k_proj, v_proj and o_proj.'} In total, 24 blocks × 4 tables = 96 adapters.`;
  };
  cells.forEach(c => c.addEventListener('pointerenter', () => show(+c.dataset.i, +c.dataset.j)));
  show(0, 0);
};

FIGS.splines = body => {
  const wrap = H('div', { class: 'patch' }, body);
  const [cv, ctx] = canvas(wrap, 638, 358);
  const side = H('div', { class: 'patch-side' }, wrap);
  H('div', { class: 'tk' }, side, 'model answer (edit me)');
  const ta = H('textarea', { rows: 6, spellcheck: 'false' }, side);
  ta.value = LANES;
  const ro = readout(body);
  const ctl = controls(body);
  const base = document.createElement('canvas'); base.width = 638; base.height = 358;
  drawRoad(base.getContext('2d'), 638, 358, ROWS, false);
  const cols = [C.red, C.yellow, C.green, C.blue, C.purple, C.orange];
  const draw = () => {
    ctx.drawImage(base, 0, 0);
    const res = parseLanes(ta.value);
    if (res.error) { ro.innerHTML = `<span class="c-red">This line is not a valid lane, so the scorer would reject the whole answer:</span> <code>${res.error.replace(/</g, '&lt;')}</code>. Each line needs six numbers, with 0 ≤ top &lt; bottom ≤ 1.`; return; }
    res.rows.forEach((row, i) => {
      ctx.strokeStyle = cols[i % cols.length]; ctx.lineWidth = 3; ctx.beginPath();
      for (let k = 0; k <= 40; k++) { const y = lerp(row[0], row[1], k / 40); ctx.lineTo(laneX(row, y) * 638, y * 358); }
      ctx.stroke();
      ctx.fillStyle = cols[i % cols.length];
      for (let k = 0; k < 4; k++) { ctx.beginPath(); ctx.arc(row[2 + k] * 638, lerp(row[0], row[1], k / 3) * 358, 5, 0, 7); ctx.fill(); }
    });
    ro.innerHTML = `${res.rows.length} lane${res.rows.length === 1 ? '' : 's'}. Each line is <b>top bottom x₁ x₂ x₃ x₄</b> as fractions of the photo (1,276 × 717). The four x values sit at four evenly spaced heights between top and bottom, and the curve is the one cubic through those four dots.`;
  };
  ta.addEventListener('input', draw);
  button(ctl, 'restore the example', () => { ta.value = LANES; draw(); });
  button(ctl, 'bend the first lane', () => { ta.value = ta.value.replace(/^.*$/m, '0.400 1.000 0.462 0.300 0.220 0.150'); draw(); });
  draw();
};

FIGS.map = body => {
  const secs = [...document.querySelectorAll('section[data-part]')];
  const parts = [...new Set(secs.map(s => s.dataset.part))];
  const rowH = 46, W = 680, Hh = parts.length * rowH + 20;
  const svg = svgBox(body, W, Hh);
  const edgesG = S('g', {}, svg), nodesG = S('g', {}, svg);
  const pos = {};
  parts.forEach((p, ri) => {
    const row = secs.filter(s => s.dataset.part === p);
    const y = 20 + ri * rowH;
    text(nodesG, 8, y + 20, p, { class: 'tk' });
    let x = 70;
    row.forEach(s => {
      const label = `§${s.id.slice(1)} ${s.dataset.short}`;
      const w = label.length * 6.6 + 18;
      const grp = S('g', { class: 'mapn', tabindex: 0 }, nodesG);
      S('rect', { x, y, width: w, height: 30, rx: 15 }, grp);
      text(grp, x + w / 2, y + 19, label, { 'text-anchor': 'middle' });
      grp.addEventListener('click', () => { location.hash = s.id; });
      pos[s.id] = { x, y, w, grp };
      x += w + 8;
    });
  });
  const edges = [];
  secs.forEach(s => (s.dataset.builds || '').split(' ').filter(Boolean).forEach(fromId => {
    const a = pos[fromId], b = pos[s.id];
    if (!a || !b) return;
    const x1 = a.x + a.w / 2, y1 = a.y + 30, x2 = b.x + b.w / 2, y2 = b.y;
    const e = S('path', { d: `M${x1} ${y1} C ${x1} ${y1 + 30}, ${x2} ${y2 - 30}, ${x2} ${y2}`, class: 'mape' }, edgesG);
    edges.push({ from: fromId, to: s.id, e });
  }));
  const up = id => { const out = new Set([id]); const st = [id]; while (st.length) { const c = st.pop(); edges.filter(x => x.to === c).forEach(x => { if (!out.has(x.from)) { out.add(x.from); st.push(x.from); } }); } return out; };
  const ro = readout(body);
  const hl = id => {
    const a = id ? up(id) : null;
    Object.entries(pos).forEach(([k, v]) => v.grp.classList.toggle('dim', !!a && !a.has(k)));
    edges.forEach(x => x.e.classList.toggle('on', !!a && a.has(x.to) && a.has(x.from)));
    ro.innerHTML = id ? `To read <b>§${id.slice(1)}</b> you need the ${a.size - 1} lit sections above it. Click to jump there.` : 'Point at a section to light up everything it builds on.';
  };
  Object.keys(pos).forEach(k => { pos[k].grp.addEventListener('pointerenter', () => hl(k)); pos[k].grp.addEventListener('pointerleave', () => hl(null)); });
  hl(null);
};

const VIDEOS = [
  ['zxagGtF9MeU', 'Happy Halloween (Neural Networks Are Not Scary)', '0:59'],
  ['CqOfi41LfDw', 'The Essential Main Ideas of Neural Networks', '18:54'],
  ['wl1myxrtQHQ', 'The Chain Rule, Clearly Explained!!!', '18:24'],
  ['sDv4f4s2SB8', 'Gradient Descent, Step-by-Step', '23:54'],
  ['IN2XmBhILt4', 'Neural Networks Pt. 2: Backpropagation Main Ideas', '17:34'],
  ['iyn2zdALii8', 'Backpropagation Details Pt. 1: Optimizing 3 parameters simultaneously.', '18:32'],
  ['GKZoOHXGcLo', 'Backpropagation Details Pt. 2: Going bonkers with The Chain Rule', '13:09'],
  ['68BZ5f7P94E', 'Neural Networks Pt. 3: ReLU In Action!!!', '8:58'],
  ['83LYR-1IcjA', 'Neural Networks Pt. 4: Multiple Inputs and Outputs', '13:50'],
  ['KpKog-L9veg', 'Neural Networks Part 5: ArgMax and SoftMax', '14:03'],
  ['M59JElEPgIg', 'The SoftMax Derivative, Step-by-Step!!!', '7:13'],
  ['6ArSys5qHAU', 'Neural Networks Part 6: Cross Entropy', '9:31'],
  ['xBEh66V9gZo', 'Neural Networks Part 7: Cross Entropy Derivatives and Backpropagation', '22:08'],
  ['HGwBXDKFk9I', 'Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)', '15:24'],
  ['AsNTP8Kwu80', 'Recurrent Neural Networks (RNNs), Clearly Explained!!!', '16:37'],
  ['YCzL96nL7j0', 'Long Short-Term Memory (LSTM), Clearly Explained', '20:45'],
  ['viZrOnJclY0', 'Word Embedding and Word2Vec, Clearly Explained!!!', '16:12'],
  ['L8HKweZIOmg', 'Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Clearly Explained!!!', '16:50'],
  ['PSs6nxngL6k', 'Attention for Neural Networks, Clearly Explained!!!', '15:51'],
  ['zxQyTK8quyY', "Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!", '36:15'],
  ['bQ5BoolX9Ag', 'Decoder-Only Transformers, ChatGPTs specific Transformer, Clearly Explained!!!', '36:45'],
  ['GDN649X_acE', 'Encoder-Only Transformers (like BERT) for RAG, Clearly Explained!!!', '18:52'],
  ['9hbQieQh7-o', 'Reinforcement Learning with Neural Networks: Essential Concepts', '24:00'],
  ['DVGmsnxB2UQ', 'Reinforcement Learning with Neural Networks: Mathematical Details', '25:01'],
  ['qPN_XZcJf_s', 'Reinforcement Learning with Human Feedback (RLHF), Clearly Explained!!!', '18:02'],
  ['L35fFDpwIM4', 'Tensors for Neural Networks, Clearly Explained!!!', '9:40'],
  ['ZTt9gsGcdDo', 'Essential Matrix Algebra for Neural Networks, Clearly Explained!!!', '30:01'],
  ['KphmOJnLAdI', 'The matrix math behind transformer neural networks, one step at a time!!!', '23:43'],
  ['FHdlXe1bSe4', 'The StatQuest Introduction to PyTorch', '23:22'],
  ['khMzi6xPbuM', 'Introduction to Coding Neural Networks with PyTorch and Lightning', '20:43'],
  ['RHGiXPuo_pI', 'Long Short-Term Memory with PyTorch + Lightning', '33:24'],
  ['Qf06XDYXCXI', 'Word Embedding in PyTorch + Lightning', '32:02'],
  ['C9QSpl5nmrY', 'Coding a ChatGPT Like Transformer From Scratch in PyTorch', '31:11'],
];
const PLAYLIST = 'PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1';
const vidUrl = n => `https://www.youtube.com/watch?v=${VIDEOS[n - 1][0]}&list=${PLAYLIST}&index=${n}`;

function wireRefs(root) {
  root.querySelectorAll('a.ref').forEach(a => {
    if (a.dataset.wired) return;
    a.dataset.wired = 1;
    a.addEventListener('mouseenter', () => {
      const id = a.getAttribute('href').slice(1), t = document.getElementById(id);
      if (!t) return;
      const tip = document.getElementById('tip');
      const h = t.querySelector('h2, h3'), gist = t.querySelector('.gist');
      tip.innerHTML = `<b>${h ? h.textContent : id}</b>${gist ? `<p>${gist.textContent}</p>` : ''}`;
      const r = a.getBoundingClientRect();
      tip.style.left = `${Math.min(window.innerWidth - 340, r.left)}px`;
      tip.style.top = `${r.bottom + 8}px`;
      tip.classList.add('on');
    });
    a.addEventListener('mouseleave', () => document.getElementById('tip').classList.remove('on'));
  });
}

function page() {
  const secs = [...document.querySelectorAll('section[data-part]')];
  secs.forEach(s => {
    const head = s.querySelector('h2');
    const meta = H('div', { class: 'meta' });
    const builds = (s.dataset.builds || '').split(' ').filter(Boolean);
    if (builds.length) meta.innerHTML += `<span class="meta-k">Builds on</span> ${builds.map(b => `<a class="ref chip" href="#${b}">§${b.slice(1)} ${document.getElementById(b)?.dataset.short || ''}</a>`).join(' ')}`;
    const vids = (s.dataset.videos || '').split(' ').filter(Boolean).map(Number);
    if (vids.length) meta.innerHTML += `<span class="meta-k">Videos</span> ${vids.map(n => `<a class="vid chip" href="${vidUrl(n)}" target="_blank" rel="noopener" title="${VIDEOS[n - 1][1]}">▶ ${n}. ${VIDEOS[n - 1][1]} <i>${VIDEOS[n - 1][2]}</i></a>`).join(' ')}`;
    head.after(meta);
  });
  const toc = document.getElementById('toc-list');
  let part = null;
  secs.forEach(s => {
    if (s.dataset.part !== part) { part = s.dataset.part; H('li', { class: 'toc-part' }, toc, document.querySelector(`[data-part-title="${part}"]`)?.textContent || part); }
    const li = H('li', {}, toc);
    const a = H('a', { href: `#${s.id}` }, li, `§${s.id.slice(1)} ${s.dataset.short}`);
    a.dataset.id = s.id;
  });
  const used = {};
  secs.forEach(s => (s.dataset.videos || '').split(' ').filter(Boolean).forEach(n => { (used[n] = used[n] || []).push(s.id); }));
  const vt = document.getElementById('video-index');
  if (vt) vt.innerHTML = '<tr><th>#</th><th>Video</th><th>Length</th><th>Covered in</th></tr>' + VIDEOS.map((v, i) => `<tr><td>${i + 1}</td><td><a href="${vidUrl(i + 1)}" target="_blank" rel="noopener">${v[1]}</a></td><td>${v[2]}</td><td>${(used[i + 1] || []).map(id => `<a class="ref" href="#${id}">§${id.slice(1)}</a>`).join(' ') || '<b class="c-red">missing</b>'}</td></tr>`).join('');
  const gl = document.getElementById('glossary');
  if (gl) {
    const defs = [...document.querySelectorAll('.def[data-term]')].sort((a, b) => a.dataset.term.localeCompare(b.dataset.term));
    gl.innerHTML = defs.map(d => {
      const sec = d.closest('section');
      return `<dt><a href="#${d.id}">${d.dataset.term}</a> <a class="ref" href="#${sec.id}">§${sec.id.slice(1)}</a></dt><dd>${d.querySelector('p').innerHTML}</dd>`;
    }).join('');
  }
  wireRefs(document);
  const links = [...toc.querySelectorAll('a')];
  const bar = document.getElementById('progress');
  const onScroll = () => {
    const y = window.scrollY + 120;
    let cur = secs[0];
    for (const s of secs) if (s.offsetTop <= y) cur = s;
    links.forEach(a => a.classList.toggle('on', a.dataset.id === cur.id));
    bar.style.width = `${(window.scrollY / Math.max(1, document.body.scrollHeight - window.innerHeight)) * 100}%`;
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

window.__figs = { mounted: [], errors: [], videos: VIDEOS.length };
function mountAll() {
  page();
  document.querySelectorAll('[data-fig]').forEach(el => {
    const name = el.dataset.fig;
    const body = H('div', { class: 'fig-body' });
    el.insertBefore(body, el.querySelector('figcaption'));
    try {
      if (!FIGS[name]) throw new Error(`no figure named ${name}`);
      FIGS[name](body, el.dataset);
      window.__figs.mounted.push(name);
    } catch (err) {
      window.__figs.errors.push(`${name}: ${err.message}`);
      body.innerHTML = `<p class="c-red">Figure failed: ${err.message}</p>`;
    }
  });
  document.querySelectorAll('.fig').forEach((f, i) => {
    const cap = f.querySelector('figcaption');
    if (cap) cap.insertAdjacentHTML('afterbegin', `<b>Figure ${i + 1}.</b> `);
  });
  document.body.dataset.mounted = window.__figs.mounted.length;
  document.body.dataset.errors = JSON.stringify(window.__figs.errors);
  document.body.dataset.ready = '1';
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountAll); else mountAll();
})();
