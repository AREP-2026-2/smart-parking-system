"use strict";

const $ = (id) => document.getElementById(id);
const HORIZONS = [1, 3, 5, 8, 12, 24];
const BRANCH = {
  lstm: { name: "LSTM con atención", color: "var(--s1)" },
  transformer: { name: "Transformer", color: "var(--s2)" },
  baseline: { name: "Línea base", color: "var(--s3)" },
};
const WEIGHT_LABELS = {
  alpha: "α Distancia", beta: "β Tiempo", gamma: "γ Congestión",
  delta: "δ Tamaño", epsilon: "ε Accesibilidad", zeta: "ζ Saturación prevista",
};
const TERMS = [
  ["distance", "alpha", "Distancia (D)"], ["time", "beta", "Tiempo (T)"],
  ["congestion", "gamma", "Congestión (C)"], ["inadequacy", "delta", "Inadecuación (I)"],
  ["accessibility", "epsilon", "Accesibilidad (A)"], ["saturation", "zeta", "Saturación prevista (S)"],
];
const POLICIES = {
  NEAREST: "Vecino más cercano", COST_WITHOUT_SATURATION: "Costo sin saturación",
  COST_REACTIVE: "Costo con ocupación actual", COST_FORECAST: "Costo con predicción",
};
const REASONS = {
  branch_down: "rama no disponible", error_limit: "error sobre el límite",
  prediction_service_unavailable: "servicio de inferencia caído",
};

const state = { data: null, chosen: null, forecast: null };

const esc = (value) => String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;
const formatTime = (iso) => new Date(iso).toLocaleString("es-CO", {
  weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
});
const isoMinute = (iso) => iso.slice(0, 16);

async function request(url, options = {}) {
  const init = { ...options, headers: { "Content-Type": "application/json" } };
  const response = await fetch(url, init);
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try { detail = JSON.parse(text).error || JSON.parse(text).detail || text; } catch (ignored) { /* texto plano */ }
    throw new Error(typeof detail === "string" ? detail : `Error ${response.status}`);
  }
  return response.json();
}

function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.hidden = true; }, 4500);
}

const guarded = (action) => async (...args) => {
  try { return await action(...args); } catch (failure) { toast(failure.message); }
};

function badge(branch, degraded) {
  const info = BRANCH[branch];
  return `<span class="badge" style="--badge:${info.color}">${esc(info.name)}${degraded ? " (degradado)" : ""}</span>`;
}

function table(headers, rows) {
  const head = headers.map(([label, numeric]) => `<th class="${numeric ? "num" : ""}">${esc(label)}</th>`).join("");
  const body = rows.map((row) => `<tr>${row.map((cell, i) =>
    `<td class="${headers[i][1] ? "num" : ""}">${cell}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

/* ---------- Gráficas SVG ---------- */

const tooltip = $("tooltip");
function showTip(event, html) {
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const box = tooltip.getBoundingClientRect();
  tooltip.style.left = `${Math.min(event.clientX + 14, innerWidth - box.width - 8)}px`;
  tooltip.style.top = `${Math.max(event.clientY - box.height - 10, 8)}px`;
}
const hideTip = () => { tooltip.hidden = true; };

function legendHtml(items) {
  return `<div class="legend chart-legend">${items.map((item) =>
    `<span><i class="key ${item.dot ? "dot" : ""}" style="background:${item.color}"></i>${esc(item.name)}</span>`).join("")}</div>`;
}

/** Líneas y marcadores sobre un eje X numérico. Cada punto: {x, y, tip}. */
function lineChart(container, config) {
  const width = 620, height = 270, left = 44, right = 14, top = 10, bottom = 30;
  const plotW = width - left - right, plotH = height - top - bottom;
  const { xMin, xMax, yMax, ticks, xTicks, yFormat } = config;
  const sx = (x) => left + ((x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y) => top + plotH - (y / yMax) * plotH;

  let svg = "";
  for (const tick of ticks) {
    svg += `<line class="${tick === 0 ? "axisline" : "gridline"}" x1="${left}" x2="${width - right}" y1="${sy(tick)}" y2="${sy(tick)}"/>`;
    svg += `<text x="${left - 8}" y="${sy(tick) + 4}" text-anchor="end">${esc(yFormat(tick))}</text>`;
  }
  for (const tick of xTicks) {
    svg += `<text x="${sx(tick.x)}" y="${height - 10}" text-anchor="middle">${esc(tick.label)}</text>`;
  }
  if (config.nowX !== undefined) {
    svg += `<line class="now-line" x1="${sx(config.nowX)}" x2="${sx(config.nowX)}" y1="${top}" y2="${top + plotH}"/>`;
    svg += `<text x="${sx(config.nowX) + 5}" y="${top + 11}">Ahora</text>`;
  }
  for (const series of config.series) {
    if (series.line !== false && series.points.length > 1) {
      const path = series.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join(" ");
      svg += `<path class="series" d="${path}" stroke="${series.color}" ${series.opacity ? `opacity="${series.opacity}"` : ""}/>`;
    }
  }
  for (const series of config.series) {
    if (series.markers) {
      for (const p of series.points) {
        svg += `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="5" fill="${p.color || series.color}" stroke="var(--surface)" stroke-width="2"/>`;
      }
    }
  }
  svg += `<line class="crosshair" data-cross x1="0" x2="0" y1="${top}" y2="${top + plotH}" visibility="hidden"/>`;
  svg += `<rect data-hit x="${left}" y="${top}" width="${plotW}" height="${plotH}" fill="transparent"/>`;
  container.innerHTML = `${legendHtml(config.legend)}<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(config.label)}">${svg}</svg>`;

  const root = container.querySelector("svg");
  const cross = root.querySelector("[data-cross]");
  const allX = [...new Set(config.series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
  const hit = root.querySelector("[data-hit]");
  hit.addEventListener("mousemove", (event) => {
    const bounds = root.getBoundingClientRect();
    const px = ((event.clientX - bounds.left) / bounds.width) * width;
    const target = allX.reduce((best, x) => (Math.abs(sx(x) - px) < Math.abs(sx(best) - px) ? x : best), allX[0]);
    cross.setAttribute("x1", sx(target));
    cross.setAttribute("x2", sx(target));
    cross.setAttribute("visibility", "visible");
    showTip(event, config.tip(target));
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

function barChart(container, values, color, label, tip) {
  const width = 620, height = 230, left = 44, right = 14, top = 10, bottom = 30;
  const plotW = width - left - right, plotH = height - top - bottom;
  const max = Math.max(...values, 0.01) * 1.1;
  const band = plotW / values.length;
  const barW = Math.min(18, band - 2);
  let svg = "";
  for (const fraction of [0, 0.5, 1]) {
    const y = top + plotH - fraction * plotH;
    svg += `<line class="${fraction ? "gridline" : "axisline"}" x1="${left}" x2="${width - right}" y1="${y}" y2="${y}"/>`;
    svg += `<text x="${left - 8}" y="${y + 4}" text-anchor="end">${(fraction * max * 100).toFixed(0)}%</text>`;
  }
  values.forEach((value, i) => {
    const h = Math.max(1, (value / max) * plotH), x = left + i * band + (band - barW) / 2, y = top + plotH - h;
    const radius = Math.min(4, h);
    svg += `<path d="M${x} ${y + h}V${y + radius}Q${x} ${y} ${x + radius} ${y}H${x + barW - radius}Q${x + barW} ${y} ${x + barW} ${y + radius}V${y + h}Z" fill="${color}"/>`;
    svg += `<rect data-i="${i}" x="${left + i * band}" y="${top}" width="${band}" height="${plotH}" fill="transparent"/>`;
    if ((values.length - 1 - i) % 6 === 0) {
      svg += `<text x="${x + barW / 2}" y="${height - 10}" text-anchor="middle">${i === values.length - 1 ? "t" : `t−${values.length - 1 - i}`}</text>`;
    }
  });
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)}">${svg}</svg>`;
  container.querySelectorAll("rect[data-i]").forEach((rect) => {
    rect.addEventListener("mousemove", (event) => showTip(event, tip(Number(rect.dataset.i))));
    rect.addEventListener("mouseleave", hideTip);
  });
}

/* ---------- Estado y mapa ---------- */

function renderState() {
  const data = state.data;
  const clock = $("clock");
  clock.min = isoMinute(data.minAsOf);
  clock.max = isoMinute(data.maxAsOf);
  clock.value = isoMinute(data.asOf);

  const byZone = {};
  data.spots.forEach((item) => (byZone[item.spot.zone] ||= []).push(item));
  $("zones").innerHTML = data.zones.map((zone) => {
    const cells = byZone[zone.id].map((item) => {
      const classes = ["spot"];
      if (item.occupied) classes.push("occupied");
      if (item.spot.accessible) classes.push("accessible");
      if (item.reserved) classes.push("reserved");
      if (state.chosen === item.spot.id) classes.push("chosen");
      const status = item.occupied ? "ocupado" : item.reserved ? "reservado" : "libre";
      return `<i class="${classes.join(" ")}" title="${esc(item.spot.id)} · ${status} · ${item.spot.size}${item.spot.accessible ? " · accesible" : ""}"></i>`;
    }).join("");
    const full = zone.occupancy >= 0.9;
    return `<div>
      <div class="zone-title"><span>Zona ${esc(zone.id)}</span><span>${pct(zone.occupancy)}</span></div>
      <div class="zone-sub">${zone.occupied}/${zone.capacity} · congestión ${pct(zone.congestion)}${full ? " · saturada" : ""}</div>
      <div class="meter ${full ? "full" : ""}"><div style="width:${pct(Math.min(1, zone.occupancy))}"></div></div>
      <div class="spots">${cells}</div>
    </div>`;
  }).join("");
}

async function loadState() {
  state.data = await request("/api/state");
  renderState();
}

async function afterClockChange() {
  state.chosen = null;
  $("result").innerHTML = '<p class="muted">Solicita un espacio o carga un escenario.</p>';
  $("scenarioText").textContent = "";
  document.querySelectorAll("[data-scenario]").forEach((b) => b.removeAttribute("aria-pressed"));
  renderState();
  await loadForecast();
}

async function moveClock(value) {
  state.data = await request("/api/clock", { method: "PUT", body: JSON.stringify({ asOf: `${value}:00` }) });
  await afterClockChange();
}

function shiftClock(hours) {
  const moved = new Date(new Date(state.data.asOf).getTime() + hours * 3600000);
  const pad = (n) => String(n).padStart(2, "0");
  moveClock(`${moved.getFullYear()}-${pad(moved.getMonth() + 1)}-${pad(moved.getDate())}T${pad(moved.getHours())}:00`)
    .catch((failure) => toast(failure.message));
}

/* ---------- Asignación ---------- */

function renderWeights() {
  $("weights").innerHTML = Object.entries(WEIGHT_LABELS).map(([key, label]) => `
    <label>${esc(label)}
      <input type="range" min="0" max="1" step="0.05" data-weight="${key}" value="${state.data.defaultWeights[key]}">
      <output>${Number(state.data.defaultWeights[key]).toFixed(2)}</output>
    </label>`).join("");
  $("weights").addEventListener("input", (event) => {
    event.target.parentElement.querySelector("output").textContent = Number(event.target.value).toFixed(2);
  });
}

function currentWeights() {
  const weights = {};
  document.querySelectorAll("[data-weight]").forEach((input) => { weights[input.dataset.weight] = Number(input.value); });
  return weights;
}

function renderAssignment(result) {
  if (!result.assigned) {
    $("result").innerHTML = `<p class="message status-bad">✕ ${esc(result.message)}</p>`;
    return;
  }
  const { best, nearest, weights } = result;
  const same = nearest && nearest.spot.id === best.spot.id;
  const rows = TERMS.map(([term, weight, label]) => [
    esc(label), weights[weight].toFixed(2), best.terms[term].toFixed(2),
    (best.terms[term] * weights[weight]).toFixed(3),
    nearest ? nearest.terms[term].toFixed(2) : "–",
    nearest ? (nearest.terms[term] * weights[weight]).toFixed(3) : "–",
  ]);
  rows.push(["<b>Costo total</b>", "", "", `<b>${best.cost.toFixed(3)}</b>`, "", nearest ? `<b>${nearest.cost.toFixed(3)}</b>` : "–"]);

  const forecasts = [...result.forecasts].sort((a, b) => a.zone.localeCompare(b.zone)).map((f) => [
    `Zona ${esc(f.zone)}`, f.horizon_hours < 1 ? `${Math.round(f.horizon_hours * 60)} min` : `${f.horizon_hours.toFixed(1)} h`,
    pct(f.occupancy), badge(f.branch, f.degraded), f.reason ? esc(REASONS[f.reason] || f.reason) : "–",
  ]);

  $("result").innerHTML = `
    <div class="summary">
      <div class="tile"><div class="name">Espacio asignado</div><div class="value">${esc(best.spot.id)}</div>
        <div class="note">${Math.round(best.spot.distanceMeters)} m · ${best.travelMinutes.toFixed(1)} min${best.spot.accessible ? " · accesible" : ""}</div></div>
      <div class="tile"><div class="name">Vecino más cercano</div><div class="value">${nearest ? esc(nearest.spot.id) : "–"}</div>
        <div class="note">${same ? "Coincide con la asignación" : "Descartado por mayor costo"}</div></div>
      <div class="tile"><div class="name">Respuesta</div><div class="value">${result.latencyMs} ms</div>
        <div class="note">${result.degraded ? '<span class="status-warn">⚠ Con degradación controlada</span>' : '<span class="status-good">✓ Predicción normal</span>'}</div></div>
    </div>
    ${result.message ? `<p class="message status-warn">⚠ ${esc(result.message)}</p>` : ""}
    <div class="two" style="margin-top:0">
      <div><h3>Desglose de la Ecuación (2)</h3>${table([["Término"], ["Peso", 1], ["Asignado", 1], ["Aporte", 1], ["Cercano", 1], ["Aporte", 1]], rows)}</div>
      <div><h3>Quién respondió por zona</h3>${table([["Zona"], ["Horizonte", 1], ["Prevista", 1], ["Rama"], ["Motivo"]], forecasts)}</div>
    </div>`;
}

async function assign() {
  const body = {
    size: $("size").value, needsAccessible: $("accessible").checked,
    arrivalInHours: Number($("arrival").value), weights: currentWeights(), reserve: true,
  };
  const result = await request("/api/assignments", { method: "POST", body: JSON.stringify(body) });
  renderAssignment(result);
  state.chosen = result.assigned ? result.best.spot.id : null;
  await loadState();
  if (result.assigned) {
    $("forecastZone").value = result.best.spot.zone;
    await loadForecast();
  }
}

async function loadScenario(id) {
  const scenario = await request(`/api/scenarios/${id}`, { method: "POST" });
  $("size").value = scenario.size;
  $("accessible").checked = scenario.needsAccessible;
  $("arrival").value = scenario.arrivalInHours;
  await assign();
  $("scenarioText").textContent = `${formatTime(scenario.asOf)} — ${scenario.description}`;
  document.querySelectorAll("[data-scenario]").forEach((b) => {
    if (b.dataset.scenario === id) b.setAttribute("aria-pressed", "true"); else b.removeAttribute("aria-pressed");
  });
}

/* ---------- Pronóstico ---------- */

function renderForecast() {
  const view = state.forecast;
  const observedPast = view.history.map((point, i) => ({ x: i - 23, y: point.occupancy }));
  const observedFuture = [{ x: 0, y: observedPast[23].y }, ...view.future.map((point, i) => ({ x: i + 1, y: point.occupancy }))];
  const predicted = view.points.map((p) => ({ x: p.forecast.horizon_hours, y: p.forecast.occupancy, color: BRANCH[p.forecast.branch].color }));
  const usedBranches = [...new Set(view.points.map((p) => p.forecast.branch))];
  lineChart($("forecastChart"), {
    label: `Ocupación observada y prevista de la zona ${view.zone}`,
    xMin: -23, xMax: 24, yMax: 1, ticks: [0, 0.25, 0.5, 0.75, 1], yFormat: (v) => pct(v), nowX: 0,
    xTicks: [-23, -12, 5, 12, 24].map((x) => ({ x, label: x > 0 ? `+${x}h` : `${x}h` })),
    series: [
      { points: observedPast, color: "var(--observed)" },
      { points: observedFuture, color: "var(--observed)", opacity: 0.45 },
      { points: predicted, color: "var(--ink-2)", line: false, markers: true },
    ],
    legend: [
      { name: "Observado", color: "var(--observed)" },
      ...usedBranches.map((b) => ({ name: `Pronóstico ${BRANCH[b].name}`, color: BRANCH[b].color, dot: true })),
    ],
    tip: (x) => {
      const point = view.points.find((p) => p.forecast.horizon_hours === x);
      if (point) {
        return `<b>+${x} h</b><br>Previsto ${pct(point.forecast.occupancy, 1)} (${esc(BRANCH[point.forecast.branch].name)})<br>Observado ${pct(point.observed, 1)}`;
      }
      const moment = x <= 0 ? view.history[x + 23] : view.future[x - 1];
      return moment ? `<b>${esc(formatTime(moment.time))}</b><br>Observado ${pct(moment.occupancy, 1)}` : "";
    },
  });
  $("forecastTable").innerHTML = table([["Horizonte"], ["Rama"], ["Previsto", 1], ["Observado", 1], ["Error", 1]],
    view.points.map((p) => [`${p.forecast.horizon_hours} h`, badge(p.forecast.branch, p.forecast.degraded),
      pct(p.forecast.occupancy, 1), pct(p.observed, 1), pct(Math.abs(p.forecast.occupancy - p.observed), 1)]));
  renderAttention();
}

function renderAttention() {
  const horizon = Number($("attentionHorizon").value);
  const point = state.forecast.points.find((p) => p.forecast.horizon_hours === horizon);
  const container = $("attentionChart");
  if (!point.forecast.attention) {
    container.innerHTML = '<p class="muted">La línea base no usa atención: la rama no está respondiendo en este horizonte.</p>';
    $("attentionTitle").textContent = "Pesos de atención sobre la ventana de 24 h";
    return;
  }
  $("attentionTitle").textContent = `Pesos de atención · ${BRANCH[point.forecast.branch].name} a +${horizon} h`;
  barChart(container, point.forecast.attention, BRANCH[point.forecast.branch].color,
    `Pesos de atención de ${BRANCH[point.forecast.branch].name} a ${horizon} horas`,
    (i) => `<b>${esc(formatTime(state.forecast.history[i].time))}</b><br>Peso ${pct(point.forecast.attention[i], 1)}`);
}

async function loadForecast() {
  state.forecast = await request(`/api/forecasts?zone=${encodeURIComponent($("forecastZone").value)}`);
  renderForecast();
}

/* ---------- Evaluación ---------- */

function verdictTile(name, ok, note, weak = false) {
  const cls = ok && !weak ? "status-good" : "status-warn";
  const text = weak ? "⚠ Evidencia débil" : ok ? "✓ Se cumple" : "⚠ No se cumple";
  return `<div class="tile"><div class="name">${esc(name)}</div>
    <div class="value ${cls}">${text}</div><div class="note">${esc(note)}</div></div>`;
}

/** MAE medio de prueba que tendría el enrutador con un umbral dado. */
function routerMae(report, tau) {
  const errors = HORIZONS.map((h) => report.horizons[h][h <= tau ? "lstm" : "transformer"].mae);
  return errors.reduce((sum, value) => sum + value, 0) / errors.length;
}

async function loadReport() {
  const report = await request("/ml/v1/report");
  const tau = report.tau === 0 ? "0 h: todo Transformer" : report.tau === 24 ? "24 h: todo LSTM" : `${report.tau} h`;
  $("reportNote").textContent = report.dataset_kind === "synthetic"
    ? "Resultados sobre datos simulados: muestran que el método funciona de punta a punta, no el comportamiento real de la ECI."
    : `Resultados sobre datos de tipo: ${report.dataset_kind}.`;
  const m = report.means;
  const mae = (name) => pct(m[name].mae, 2);
  const crossover = report.hypotheses.h1_crossover_horizon;
  const singleBranch = report.tau === 0 || report.tau === 24;
  $("hypotheses").innerHTML = [
    verdictTile("H1 · Existe un cruce τ", report.hypotheses.h1_holds,
      crossover === 24 ? "Transformer solo supera al LSTM de forma sostenida en el último horizonte (24 h)"
        : report.hypotheses.h1_holds ? `Transformer gana desde ${crossover} h` : "Sin horizonte a partir del cual gane siempre",
      crossover === 24),
    verdictTile("H2 · Enrutador mejor que todos", report.hypotheses.h2_holds,
      `MAE ${mae("router")} · LSTM ${mae("lstm")} · Transformer ${mae("transformer")} · Base ${mae("baseline")}.`
      + `${singleBranch ? " Con este τ el enrutador equivale a una sola rama." : ""}`
      + ` Con τ = 1 h: ${pct(routerMae(report, 1), 2)}`),
    `<div class="tile"><div class="name">Umbral τ calibrado</div><div class="value">${esc(tau)}</div><div class="note">Elegido solo con validación; el reporte H2 usa este valor. El τ activo se cambia en Operación.</div></div>`,
  ].join("");

  const names = [["lstm", "LSTM con atención", "var(--s1)"], ["transformer", "Transformer", "var(--s2)"],
    ["baseline", "Línea base semana anterior", "var(--s3)"], ["router", "Enrutador", "var(--s4)"]];
  const position = (h) => HORIZONS.indexOf(h);
  const yMax = Math.ceil(Math.max(...HORIZONS.flatMap((h) => names.map(([key]) => report.horizons[h][key].mae))) * 20) / 20;
  const ticks = []; for (let t = 0; t <= yMax + 1e-9; t += yMax > 0.2 ? 0.1 : 0.05) ticks.push(Number(t.toFixed(2)));
  lineChart($("maeChart"), {
    label: "Error absoluto medio por horizonte y modelo",
    xMin: -0.25, xMax: 5.25, yMax, ticks, yFormat: (v) => `${(v * 100).toFixed(0)} pp`,
    xTicks: HORIZONS.map((h) => ({ x: position(h), label: `${h} h` })),
    series: names.map(([key, , color]) => ({
      color, markers: true, points: HORIZONS.map((h) => ({ x: position(h), y: report.horizons[h][key].mae })),
    })),
    legend: names.map(([, name, color]) => ({ name, color })),
    tip: (x) => {
      const h = HORIZONS[x];
      return `<b>${h} h</b><br>${names.map(([key, name]) => `${esc(name)}: ${(report.horizons[h][key].mae * 100).toFixed(2)} pp`).join("<br>")}`;
    },
  });
  $("maeNote").textContent = "El enrutador coincide con la rama que elige en cada horizonte, por lo que su línea cubre a esa rama.";
  $("maeTable").innerHTML = table([["Horizonte"], ["Modelo"], ["MAPE %", 1], ["MAE", 1], ["MSE", 1], ["R²", 1]],
    HORIZONS.flatMap((h) => names.map(([key, name]) => {
      const r = report.horizons[h][key];
      return [`${h} h`, esc(key === "router" ? `${name} (${BRANCH[r.branch].name})` : name), r.mape.toFixed(2), r.mae.toFixed(4), r.mse.toFixed(4), r.r2.toFixed(3)];
    })));
}

async function runExperiment() {
  const button = $("runExperiment");
  button.disabled = true;
  try {
    const result = await request("/api/experiments/allocation", { method: "POST", body: JSON.stringify(currentWeights()) });
    const signed = (d, unit) => `${d.mean >= 0 ? "+" : ""}${d.mean.toFixed(1)} ${unit} (IC95 ${d.lower95.toFixed(1)} a ${d.upper95.toFixed(1)})`;
    $("experiment").innerHTML = `
      ${table([["Política"], ["Adecuadas %", 1], ["Tiempo medio (s)", 1]],
        result.policies.map((p) => [esc(POLICIES[p.policy]), p.adequatePercent.toFixed(1), p.meanSeconds.toFixed(1)]))}
      <div class="tiles">${verdictTile("H3 · Costo con predicción vs. vecino más cercano", result.hypothesisHolds,
        `Adecuadas ${signed(result.adequateVsNearest, "pp")} · Tiempo ${signed(result.secondsVsNearest, "s")}`)}</div>
      <p class="muted small">${result.events} horas pico del periodo de prueba, ${result.vehicles} vehículos, datos: ${esc(result.datasetKind)}.
      “Adecuada”: tamaño y accesibilidad correctos en una zona que no supera el 90% al llegar.
      ${result.degradedEvents ? `<b>${result.degradedEvents} eventos con degradación.</b>` : ""}</p>`;
  } finally {
    button.disabled = false;
  }
}

/* ---------- Operación ---------- */

async function loadOperations() {
  let models;
  try {
    models = await request("/ml/v1/models");
  } catch (failure) {
    $("branches").innerHTML = '<div class="tile"><div class="name">Servicio de inferencia</div><div class="value status-bad">✕ Caído</div><div class="note">La asignación sigue funcionando con la línea base local</div></div>';
    $("healthTable").innerHTML = "";
    return;
  }
  const tauLabel = (tau) => {
    const notes = [];
    if (tau === 1) notes.push("política inicial del paper");
    if (tau === models.tau) notes.push("calibrado");
    if (tau === 0) notes.push("todo Transformer");
    if (tau === 24) notes.push("todo LSTM");
    return `${tau} h${notes.length ? ` · ${notes.join(", ")}` : ""}`;
  };
  $("tauSelect").innerHTML = [0, ...HORIZONS].map((tau) =>
    `<option value="${tau}" ${tau === models.tau_active ? "selected" : ""}>${esc(tauLabel(tau))}</option>`).join("");
  const available = (branch) => models.health.find((h) => h.branch === branch).available;
  $("branches").innerHTML = ["lstm", "transformer"].map((branch) => `
    <div class="tile"><div class="name">${esc(BRANCH[branch].name)}</div>
      <div class="value ${available(branch) ? "status-good" : "status-bad"}">${available(branch) ? "✓ Disponible" : "✕ Caída"}</div>
      <div class="row" style="margin-top:6px">
        <button class="ghost danger" data-toggle="${branch}" data-available="${!available(branch)}">${available(branch) ? "Simular caída" : "Reactivar"}</button>
        <button class="ghost" data-drift="${branch}">Inyectar deriva</button>
      </div></div>`).join("");
  $("healthTable").innerHTML = table(
    [["Rama"], ["Horizonte", 1], ["Versión"], ["MAE observado", 1], ["Límite", 1], ["Muestras", 1], ["Estado"]],
    models.health.map((h) => {
      const over = h.rolling_mae !== null && h.rolling_mae > h.error_limit;
      const meta = models.models[`${h.branch}:${h.horizon}`];
      const status = !h.available ? '<span class="status-bad">✕ Caída</span>'
        : over ? '<span class="status-warn">⚠ Sobre el límite</span>' : '<span class="status-good">✓ Sana</span>';
      return [esc(BRANCH[h.branch].name), `${h.horizon} h`, esc(meta.name),
        h.rolling_mae === null ? "–" : pct(h.rolling_mae, 2), pct(h.error_limit, 2), h.samples, status];
    }));
}

async function afterHealthChange() {
  await loadOperations();
  await loadForecast();
}

/* ---------- Inicio ---------- */

function bind() {
  $("assignForm").addEventListener("submit", guarded(async (event) => { event.preventDefault(); await assign(); }));
  $("prevHour").addEventListener("click", () => shiftClock(-1));
  $("nextHour").addEventListener("click", () => shiftClock(1));
  $("clock").addEventListener("change", guarded((event) => moveClock(event.target.value.slice(0, 13) + ":00")));
  $("resetBtn").addEventListener("click", guarded(async () => {
    state.data = await request("/api/reset", { method: "POST" });
    state.chosen = null;
    renderState();
  }));
  document.querySelectorAll("[data-scenario]").forEach((button) =>
    button.addEventListener("click", guarded(() => loadScenario(button.dataset.scenario))));
  $("forecastZone").addEventListener("change", guarded(loadForecast));
  $("attentionHorizon").addEventListener("change", () => renderAttention());
  $("runExperiment").addEventListener("click", guarded(runExperiment));
  $("backtest").addEventListener("click", guarded(async () => {
    const errors = await request("/api/observability/backtest", { method: "POST" });
    await loadOperations();
    toast(`Comparación lista: ${errors.map((e) => `${e.horizon} h ${pct(e.mae, 1)}`).join(" · ")}`);
  }));
  $("tauSelect").addEventListener("change", guarded(async (event) => {
    await request("/ml/v1/admin/router", { method: "POST", body: JSON.stringify({ tau: Number(event.target.value) }) });
    await afterHealthChange();
  }));
  $("resetHealth").addEventListener("click", guarded(async () => {
    await request("/ml/v1/admin/reset", { method: "POST" });
    await afterHealthChange();
  }));
  $("branches").addEventListener("click", guarded(async (event) => {
    const { toggle, available, drift } = event.target.dataset;
    if (toggle) {
      await request(`/ml/v1/admin/branches/${toggle}`, { method: "POST", body: JSON.stringify({ available: available === "true" }) });
      await afterHealthChange();
    } else if (drift) {
      await request(`/ml/v1/admin/branches/${drift}/drift`, { method: "POST" });
      await afterHealthChange();
    }
  }));
}

async function init() {
  bind();
  await loadState();
  const kind = state.data.datasetKind;
  if (kind === "synthetic") {
    const banner = $("dataBanner");
    banner.textContent = "⚠ Datos simulados: las ocupaciones y métricas provienen de un simulador, no de mediciones en la ECI.";
    banner.hidden = false;
  }
  $("forecastZone").innerHTML = state.data.zones.map((z) => `<option value="${esc(z.id)}">Zona ${esc(z.id)}</option>`).join("");
  $("attentionHorizon").innerHTML = HORIZONS.map((h) => `<option value="${h}">+${h} h</option>`).join("");
  renderWeights();
  await Promise.all([loadForecast(), loadReport().catch(() => { $("reportNote").textContent = "Reporte no disponible: el servicio de inferencia no responde."; }), loadOperations()]);
}

guarded(init)();
