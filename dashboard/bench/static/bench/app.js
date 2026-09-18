(function () {
    "use strict";

    // `precision` and `runtime` are both offered because they answer different
    // questions: 4bit (MLX) and q4_k_m (GGUF) share a precision but not a
    // runtime, and the two differ by ~1.8x on decode.
    const FILTER_DIMS = [
        "version", "arch", "quant", "precision", "runtime", "date",
    ];
    const SIZE_ORDER = ["short", "1000tok", "4000tok", "16000tok"];
    const TABS = ["quality", "speed", "table"];
    const PALETTE = [
        "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
        "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
        "#0d9488", "#c026d3", "#ca8a04", "#059669", "#e11d48",
    ];

    // Measures offered on the quality tab's Y axis. Quality is scored once per
    // model, not per prompt size, so it has no place on the speed tab.
    const QUALITY_SPEED_MEASURES = ["gen_tps", "prefill_tps", "total_s", "tokens_per_wh"];
    // Rates start at zero: a non-zero origin exaggerates gaps.
    const RATE_MEASURES = ["gen_tps", "prefill_tps", "tokens_per_wh"];

    // Same colours as `plot-quality-speed`: by architecture, with Apple's model
    // on its own since it is neither of the other two.
    const ARCH_GROUPS = [
        { key: "moe", label: "MoE", color: "#2b7bba" },
        { key: "dense", label: "Dense", color: "#c1452e" },
        { key: "apple", label: "Apple on-device", color: "#5b9e4a" },
    ];

    const TABLE_COLUMNS = [
        "date", "variant", "version", "arch", "quant", "precision", "runtime", "size",
        "prompt_tokens", "completion_tokens", "gen_tps", "prefill_tps",
        "ttft", "total_s", "quality_pct", "quality_gsm8k", "quality_mmlu", "quality_code",
        "ttft_spread", "watts", "tokens_per_wh",
    ];
    const TEXT_COLUMNS = new Set([
        "date", "variant", "version", "arch", "quant", "precision", "runtime", "size",
    ]);

    const state = {
        rows: [],
        meta: null,
        tab: "quality",
        filters: {},       // dim -> Set of active values
        // quality tab
        qSize: "4000tok",
        qSpeed: "gen_tps",
        qQuality: "quality_pct",
        // speed tab
        measure: "gen_tps",
        seriesBy: "variant",
        logY: false,
        compareSize: "4000tok",
        // table tab
        sortKey: "date",
        sortAsc: true,
    };

    const colorCache = new Map();
    function colorFor(key) {
        if (!colorCache.has(key)) {
            colorCache.set(key, PALETTE[colorCache.size % PALETTE.length]);
        }
        return colorCache.get(key);
    }

    function fmt(v) {
        if (v === null || v === undefined) return "–";
        if (typeof v === "number") {
            return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
        }
        return String(v);
    }

    function isNum(v) {
        return typeof v === "number" && Number.isFinite(v);
    }

    function sizeIndex(size) {
        const idx = SIZE_ORDER.indexOf(size);
        return idx === -1 ? SIZE_ORDER.length : idx;
    }

    function mean(values) {
        return values.reduce((a, b) => a + b, 0) / values.length;
    }

    function measureLabel(key) {
        const m = state.meta.measures.find((m) => m.key === key);
        return m ? m.label : key;
    }

    function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    // ---- URL state -----------------------------------------------------

    const URL_KEYS = ["tab", "qSize", "qSpeed", "qQuality", "measure", "seriesBy", "compareSize"];

    function readStateFromURL() {
        const params = new URLSearchParams(window.location.search);
        for (const dim of FILTER_DIMS) {
            const v = params.get(dim);
            if (v !== null) state.filters[dim] = new Set(v ? v.split(",") : []);
        }
        for (const key of URL_KEYS) {
            if (params.get(key)) state[key] = params.get(key);
        }
        if (params.get("logY")) state.logY = params.get("logY") === "1";
        if (!TABS.includes(state.tab)) state.tab = "quality";
    }

    function writeStateToURL() {
        const params = new URLSearchParams();
        for (const dim of FILTER_DIMS) {
            const all = state.meta[dim] || [];
            const active = state.filters[dim];
            if (active && active.size !== all.length) {
                params.set(dim, Array.from(active).join(","));
            }
        }
        for (const key of URL_KEYS) params.set(key, state[key]);
        if (state.logY) params.set("logY", "1");
        window.history.replaceState(null, "", `${window.location.pathname}?${params}`);
    }

    // ---- data load -------------------------------------------------------

    async function loadData() {
        const [metaResp, rowsResp] = await Promise.all([
            fetch("/api/meta/"),
            fetch("/api/rows/"),
        ]);
        state.meta = await metaResp.json();
        state.rows = (await rowsResp.json()).rows;

        readStateFromURL();
        for (const dim of FILTER_DIMS) {
            if (!state.filters[dim]) {
                state.filters[dim] = new Set(state.meta[dim] || []);
            }
        }
        const sizes = state.meta.size || [];
        const fallbackSize = sizes.includes("4000tok") ? "4000tok" : sizes[0];
        if (!sizes.includes(state.qSize)) state.qSize = fallbackSize;
        if (!sizes.includes(state.compareSize)) state.compareSize = fallbackSize;
    }

    // ---- tabs --------------------------------------------------------------

    function setupTabs() {
        document.querySelectorAll(".tab").forEach((btn) => {
            btn.onclick = () => {
                state.tab = btn.dataset.tab;
                renderAll();
            };
        });
    }

    function showActiveTab() {
        document.querySelectorAll(".tab").forEach((btn) => {
            const active = btn.dataset.tab === state.tab;
            btn.classList.toggle("active", active);
            btn.setAttribute("aria-selected", String(active));
        });
        for (const tab of TABS) {
            document.getElementById(`tab-${tab}`).hidden = tab !== state.tab;
        }
    }

    // ---- filter sidebar ----------------------------------------------

    function renderFilters() {
        const container = document.getElementById("filters");
        container.innerHTML = "";
        for (const dim of FILTER_DIMS) {
            const values = state.meta[dim] || [];
            const group = document.createElement("div");
            group.className = "filter-group";

            const h3 = document.createElement("h3");
            h3.textContent = dim;
            group.appendChild(h3);

            const allNone = document.createElement("div");
            allNone.className = "all-none";
            const allBtn = document.createElement("button");
            allBtn.textContent = "all";
            allBtn.onclick = () => {
                state.filters[dim] = new Set(values);
                onFiltersChanged();
            };
            const noneBtn = document.createElement("button");
            noneBtn.textContent = "none";
            noneBtn.onclick = () => {
                state.filters[dim] = new Set();
                onFiltersChanged();
            };
            allNone.appendChild(allBtn);
            allNone.appendChild(document.createTextNode(" / "));
            allNone.appendChild(noneBtn);
            group.appendChild(allNone);

            for (const value of values) {
                const label = document.createElement("label");
                const input = document.createElement("input");
                input.type = "checkbox";
                input.checked = state.filters[dim].has(value);
                input.onchange = () => {
                    if (input.checked) state.filters[dim].add(value);
                    else state.filters[dim].delete(value);
                    onFiltersChanged();
                };
                label.appendChild(input);
                label.appendChild(document.createTextNode(value));
                group.appendChild(label);
            }
            container.appendChild(group);
        }
    }

    function filteredRows() {
        return state.rows.filter((row) =>
            FILTER_DIMS.every((dim) => state.filters[dim].has(row[dim]))
        );
    }

    function onFiltersChanged() {
        renderFilters();
        renderAll();
    }

    // ---- controls ----------------------------------------------------

    function fillSelect(id, options, current) {
        const sel = document.getElementById(id);
        sel.innerHTML = "";
        for (const { value, label } of options) {
            const opt = document.createElement("option");
            opt.value = value;
            opt.textContent = label;
            sel.appendChild(opt);
        }
        sel.value = current;
        return sel;
    }

    function bindSelect(sel, key) {
        // A bookmark can carry a value the control no longer offers (e.g. a
        // quality measure on the speed tab); fall back to the first option.
        const values = Array.from(sel.options, (o) => o.value);
        if (!values.includes(state[key]) && values.length) state[key] = values[0];
        sel.value = state[key];
        sel.onchange = () => {
            state[key] = sel.value;
            renderAll();
        };
    }

    function renderControls() {
        const measures = state.meta.measures;
        const sizeOptions = (state.meta.size || []).map((s) => ({ value: s, label: s }));
        const asOptions = (list) => list.map((m) => ({ value: m.key, label: m.label }));

        bindSelect(fillSelect("qSize", sizeOptions, state.qSize), "qSize");
        bindSelect(fillSelect("qSpeed",
            asOptions(measures.filter((m) => QUALITY_SPEED_MEASURES.includes(m.key))),
            state.qSpeed), "qSpeed");
        bindSelect(fillSelect("qQuality",
            asOptions(measures.filter((m) => m.key.startsWith("quality_"))),
            state.qQuality), "qQuality");

        bindSelect(fillSelect("measure",
            asOptions(measures.filter((m) => !m.key.startsWith("quality_"))),
            state.measure), "measure");
        bindSelect(document.getElementById("seriesBy"), "seriesBy");
        bindSelect(fillSelect("compareSize", sizeOptions, state.compareSize), "compareSize");

        const logYBox = document.getElementById("logY");
        logYBox.checked = state.logY;
        logYBox.onchange = () => {
            state.logY = logYBox.checked;
            renderAll();
        };
    }

    // ---- charts ---------------------------------------------------------

    const charts = {};

    function drawChart(id, config) {
        if (charts[id]) charts[id].destroy();
        charts[id] = new Chart(document.getElementById(id).getContext("2d"), config);
    }

    function applyThemeDefaults() {
        Chart.defaults.color = cssVar("--muted") || "#666";
        Chart.defaults.borderColor = cssVar("--border") || "#e5e7eb";
        Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    }

    // ---- quality vs speed tab ----------------------------------------------
    //
    // Port of `plot-quality-speed`: one point per variant, quality on X with
    // its 95% interval as a horizontal bar, speed on Y. The interesting region
    // is the top right -- fast *and* correct.

    function archGroup(row) {
        return row.family === "apple" ? "apple" : row.arch;
    }

    function shortLabel(row) {
        if (row.family === "apple") return "Apple on-device";
        return String(row.variant).replace(/^qwen/, "");
    }

    // The composite's interval and item count predate the per-task ones and
    // keep their original names.
    function qualityFields(key) {
        if (key === "quality_pct") {
            return { low: "quality_ci_low", high: "quality_ci_high", n: "quality_n" };
        }
        return { low: `${key}_ci_low`, high: `${key}_ci_high`, n: `${key}_n` };
    }

    function qualityPoints(rows) {
        const speedKey = state.qSpeed;
        const qualityKey = state.qQuality;
        const byVariant = new Map();
        for (const row of rows) {
            if (row.size !== state.qSize) continue;
            if (!isNum(row[speedKey]) || !isNum(row[qualityKey])) continue;
            if (!byVariant.has(row.variant)) byVariant.set(row.variant, []);
            byVariant.get(row.variant).push(row);
        }

        const points = [];
        for (const variantRows of byVariant.values()) {
            // A variant re-measured on a later date supersedes the older run,
            // matching how quality scores are joined in the loader.
            const latest = variantRows.map((r) => r.date).sort().pop();
            const current = variantRows.filter((r) => r.date === latest);
            const row = current[0];
            const ci = qualityFields(qualityKey);
            const withCI = isNum(row[ci.low]) && isNum(row[ci.high]);
            points.push({
                x: row[qualityKey],
                y: mean(current.map((r) => r[speedKey])),
                lo: withCI ? row[ci.low] : null,
                hi: withCI ? row[ci.high] : null,
                n: row[ci.n],
                label: shortLabel(row),
                group: archGroup(row),
                rows: current,
            });
        }
        return points;
    }

    // Draws the confidence bars under the points and a label beside each one.
    const qualityDecorations = {
        id: "qualityDecorations",
        beforeDatasetsDraw(chart) {
            const { ctx, scales: { x, y } } = chart;
            ctx.save();
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.45;
            chart.data.datasets.forEach((ds, i) => {
                if (!chart.isDatasetVisible(i)) return;
                ctx.strokeStyle = ds.backgroundColor;
                for (const p of ds.data) {
                    if (p.lo === null) continue;
                    const py = y.getPixelForValue(p.y);
                    const x0 = x.getPixelForValue(p.lo);
                    const x1 = x.getPixelForValue(p.hi);
                    ctx.beginPath();
                    ctx.moveTo(x0, py); ctx.lineTo(x1, py);
                    ctx.moveTo(x0, py - 4); ctx.lineTo(x0, py + 4);
                    ctx.moveTo(x1, py - 4); ctx.lineTo(x1, py + 4);
                    ctx.stroke();
                }
            });
            ctx.restore();
        },
        afterDatasetsDraw(chart) {
            const { ctx, chartArea, scales: { x, y } } = chart;
            const points = [];
            chart.data.datasets.forEach((ds, i) => {
                if (chart.isDatasetVisible(i)) points.push(...ds.data);
            });
            ctx.save();
            ctx.font = `12px ${Chart.defaults.font.family}`;
            ctx.fillStyle = cssVar("--fg") || "#222";
            ctx.textBaseline = "alphabetic";
            placeLabels(ctx, points, x, y, chartArea).forEach(({ text, lx, ly }) => {
                ctx.fillText(text, lx, ly);
            });
            ctx.restore();
        },
    };

    // Greedy placement, as in `plot-quality-speed`: the first candidate that
    // stays on the plot and collides with nothing wins. Every point and its
    // confidence bar are obstacles from the start, so a label never sits on
    // top of the interval it is annotating. Highest-quality points are placed
    // first -- they are the ones the chart is mostly about.
    function placeLabels(ctx, points, x, y, area) {
        const placed = [];
        for (const p of points) {
            const px = x.getPixelForValue(p.x);
            const py = y.getPixelForValue(p.y);
            placed.push([px - 7, py - 7, px + 7, py + 7]);
            if (p.lo !== null) {
                placed.push([x.getPixelForValue(p.lo), py - 5, x.getPixelForValue(p.hi), py + 5]);
            }
        }
        const out = [];
        for (const p of points.slice().sort((a, b) => b.x - a.x)) {
            const px = x.getPixelForValue(p.x);
            const py = y.getPixelForValue(p.y);
            const w = ctx.measureText(p.label).width;
            const candidates = [
                [px + 10, py + 4], [px + 8, py - 9], [px + 8, py + 19],
                [px - 10 - w, py + 4], [px - 8 - w, py - 9], [px - 8 - w, py + 19],
                [px - w / 2, py - 12], [px - w / 2, py + 22],
            ];
            let chosen = null;
            for (const [lx, ly] of candidates) {
                if (lx < area.left || lx + w > area.right) continue;
                const box = [lx, ly - 10, lx + w, ly + 3];
                if (placed.some((o) => box[0] < o[2] && o[0] < box[2] && box[1] < o[3] && o[1] < box[3])) continue;
                chosen = [lx, ly, box];
                break;
            }
            if (!chosen) {
                // Nothing fits cleanly; keep it on the plot rather than drop it.
                const lx = Math.min(Math.max(px + 10, area.left), area.right - w);
                chosen = [lx, py + 4, [lx, py - 6, lx + w, py + 7]];
            }
            placed.push(chosen[2]);
            out.push({ text: p.label, lx: chosen[0], ly: chosen[1] });
        }
        return out;
    }

    // Pad the data range, then snap outward to whole steps, so both ends of
    // the axis land exactly on a tick.
    function niceBounds(lo, hi, padFrac = 0.12) {
        if (hi <= lo) { lo -= 1; hi += 1; }
        const span = hi - lo;
        lo -= span * padFrac;
        hi += span * padFrac;
        const rawStep = (hi - lo) / 5;
        const magnitude = 10 ** Math.floor(Math.log10(rawStep));
        let step = magnitude;
        for (const mult of [1, 2, 2.5, 5, 10]) {
            step = magnitude * mult;
            if (step >= rawStep) break;
        }
        return { min: Math.floor(lo / step) * step, max: Math.ceil(hi / step) * step, step };
    }

    function renderQualityChart(rows) {
        const points = qualityPoints(rows);
        const datasets = ARCH_GROUPS
            .map((g) => ({
                label: g.label,
                data: points.filter((p) => p.group === g.key),
                backgroundColor: g.color,
                borderColor: cssVar("--bg") || "#fff",
                borderWidth: 1.5,
                pointRadius: 6,
                pointHoverRadius: 8,
            }))
            .filter((ds) => ds.data.length);

        const xs = points.flatMap((p) => [p.x, p.lo, p.hi]).filter(isNum);
        const xb = xs.length ? niceBounds(Math.min(...xs), Math.max(...xs)) : { min: 0, max: 100, step: 10 };
        const ys = points.map((p) => p.y);
        const rate = RATE_MEASURES.includes(state.qSpeed);
        const yb = ys.length ? niceBounds(rate ? 0 : Math.min(...ys), Math.max(...ys)) : { min: 0, max: 1, step: 0.2 };
        if (rate) yb.min = 0;

        const n = Math.max(0, ...points.map((p) => p.n || 0));
        const qualityTitle = measureLabel(state.qQuality) + (n ? ` — ${n} items, measured on this machine` : "");
        const direction = rate ? "Up and to the right is better." : "Down and to the right is better.";

        drawChart("qualityChart", {
            type: "scatter",
            data: { datasets },
            plugins: [qualityDecorations],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { right: 8 } },
                scales: {
                    x: {
                        min: Math.max(0, xb.min),
                        max: Math.min(100, xb.max),
                        title: { display: true, text: qualityTitle },
                        ticks: { stepSize: xb.step, callback: (v) => `${v}%` },
                    },
                    y: {
                        min: Math.max(0, yb.min),
                        max: yb.max,
                        ticks: { stepSize: yb.step },
                        title: { display: true, text: `${measureLabel(state.qSpeed)} @ ${state.qSize}` },
                    },
                },
                plugins: {
                    title: {
                        display: true,
                        align: "start",
                        text: `Quality vs ${measureLabel(state.qSpeed).toLowerCase()} — ${state.qSize} prompt`,
                        font: { size: 14, weight: "600" },
                        color: cssVar("--fg"),
                    },
                    subtitle: {
                        display: true,
                        align: "start",
                        text: points.length
                            ? `${direction} Bars are 95% confidence intervals on the quality score.`
                            : "No variants with both a quality score and this speed measure in the current filters.",
                        padding: { bottom: 8 },
                    },
                    legend: { position: "top", align: "start", labels: { usePointStyle: true } },
                    tooltip: {
                        callbacks: {
                            title: (items) => items[0].raw.rows[0].variant,
                            label: (item) => {
                                const p = item.raw;
                                const r = p.rows[0];
                                const lines = [
                                    `${measureLabel(state.qQuality)}: ${p.x.toFixed(1)}%` +
                                        (p.lo !== null ? ` [${p.lo.toFixed(1)}–${p.hi.toFixed(1)}]` : ""),
                                    `${measureLabel(state.qSpeed)}: ${fmt(p.y)}`,
                                    ...(p.n ? [`quality items: ${p.n}`] : []),
                                    `date: ${r.date}`,
                                ];
                                if (p.rows.length > 1) lines.push(`(mean of ${p.rows.length} rows)`);
                                return lines;
                            },
                        },
                    },
                },
            },
        });
    }

    // ---- speed by dimension tab ----------------------------------------

    function buildSeries(rows) {
        // Map: seriesValue -> Map: size -> {values: [], rows: []}
        const series = new Map();
        for (const row of rows) {
            const value = row[state.measure];
            if (!isNum(value)) continue;
            const seriesValue = row[state.seriesBy] ?? "unknown";
            if (!series.has(seriesValue)) series.set(seriesValue, new Map());
            const bySize = series.get(seriesValue);
            if (!bySize.has(row.size)) bySize.set(row.size, { values: [], rows: [] });
            const bucket = bySize.get(row.size);
            bucket.values.push(value);
            bucket.rows.push(row);
        }
        return new Map([...series].sort(([a], [b]) => String(a).localeCompare(String(b))));
    }

    function renderMainChart(rows) {
        const series = buildSeries(rows);
        const sizes = (state.meta.size || []).slice().sort((a, b) => sizeIndex(a) - sizeIndex(b));

        const datasets = [];
        for (const [seriesValue, bySize] of series) {
            datasets.push({
                label: String(seriesValue),
                data: sizes.map((size) => (bySize.has(size) ? mean(bySize.get(size).values) : null)),
                meta: sizes.map((size) => bySize.get(size) || null),
                spanGaps: true,
                borderColor: colorFor(seriesValue),
                backgroundColor: colorFor(seriesValue),
                pointRadius: 4,
                pointHoverRadius: 6,
                tension: 0.15,
            });
        }

        drawChart("mainChart", {
            type: "line",
            data: { labels: sizes, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "nearest", intersect: false },
                scales: {
                    x: { title: { display: true, text: "prompt size" } },
                    y: {
                        type: state.logY ? "logarithmic" : "linear",
                        title: { display: true, text: measureLabel(state.measure) },
                    },
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const bucket = ctx.dataset.meta[ctx.dataIndex];
                                if (!bucket) return "";
                                const n = bucket.rows.length;
                                const lines = [`${ctx.dataset.label}: ${fmt(ctx.parsed.y)}`];
                                if (n > 1) lines.push(`(mean of ${n})`);
                                const variants = new Set(bucket.rows.map((r) => r.variant));
                                const dates = new Set(bucket.rows.map((r) => r.date));
                                if (variants.size <= 3) lines.push(`variants: ${Array.from(variants).join(", ")}`);
                                if (dates.size <= 3) lines.push(`dates: ${Array.from(dates).join(", ")}`);
                                const spreads = bucket.rows.map((r) => r.ttft_spread).filter(isNum);
                                if (spreads.length) lines.push(`ttft_spread: ${fmt(mean(spreads))}`);
                                return lines;
                            },
                        },
                    },
                    legend: { position: "bottom" },
                },
            },
        });
    }

    function renderCompareChart(rows) {
        const bucketRows = rows.filter((r) => r.size === state.compareSize && isNum(r[state.measure]));
        const variants = Array.from(new Set(bucketRows.map((r) => r.variant))).sort();
        const seriesValues = Array.from(new Set(bucketRows.map((r) => r[state.seriesBy] ?? "unknown"))).sort();

        const datasets = seriesValues.map((seriesValue) => ({
            label: String(seriesValue),
            data: variants.map((variant) => {
                const matching = bucketRows.filter(
                    (r) => r.variant === variant && (r[state.seriesBy] ?? "unknown") === seriesValue
                );
                return matching.length ? mean(matching.map((r) => r[state.measure])) : null;
            }),
            backgroundColor: colorFor(seriesValue),
            skipNull: true,
        }));

        document.getElementById("compareTitle").textContent =
            `${measureLabel(state.measure)} at ${state.compareSize}`;

        drawChart("compareChart", {
            type: "bar",
            data: { labels: variants, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: "variant" } },
                    y: {
                        type: state.logY ? "logarithmic" : "linear",
                        title: { display: true, text: `${measureLabel(state.measure)} @ ${state.compareSize}` },
                    },
                },
                plugins: { legend: { position: "bottom" } },
            },
        });
    }

    // ---- table -----------------------------------------------------------

    function renderTableHeader() {
        const tr = document.querySelector("#rowsTable thead tr");
        tr.innerHTML = "";
        for (const col of TABLE_COLUMNS) {
            const th = document.createElement("th");
            th.dataset.key = col;
            th.textContent = col;
            if (TEXT_COLUMNS.has(col)) th.className = "text";
            th.onclick = () => {
                if (state.sortKey === col) {
                    state.sortAsc = !state.sortAsc;
                } else {
                    state.sortKey = col;
                    state.sortAsc = true;
                }
                renderAll();
            };
            tr.appendChild(th);
        }
    }

    function renderTable(rows) {
        const sorted = rows.slice().sort((a, b) => {
            const av = a[state.sortKey];
            const bv = b[state.sortKey];
            if (av === bv) return 0;
            if (av === null || av === undefined) return 1;
            if (bv === null || bv === undefined) return -1;
            const cmp = av < bv ? -1 : av > bv ? 1 : 0;
            return state.sortAsc ? cmp : -cmp;
        });

        const tbody = document.querySelector("#rowsTable tbody");
        tbody.innerHTML = "";
        for (const row of sorted) {
            const tr = document.createElement("tr");
            const suspect = isNum(row.ttft_spread) && row.ttft_spread > 0.25;
            if (suspect) {
                tr.className = "suspect";
                tr.title = "ttft_spread > 0.25: possible cache hit, see README";
            }
            for (const col of TABLE_COLUMNS) {
                const td = document.createElement("td");
                td.textContent = fmt(row[col]);
                if (TEXT_COLUMNS.has(col)) td.classList.add("text");
                if (suspect && col === "ttft_spread") td.classList.add("flag");
                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }

        document.querySelectorAll("#rowsTable th").forEach((th) => {
            th.classList.toggle("sorted", th.dataset.key === state.sortKey);
            th.classList.toggle("asc", th.dataset.key === state.sortKey && state.sortAsc);
        });
    }

    // ---- top-level render -------------------------------------------------

    // Only the visible tab is drawn: Chart.js sizes a chart from its container,
    // and a container inside a hidden tab has no size.
    function renderAll() {
        showActiveTab();
        const rows = filteredRows();
        if (state.tab === "quality") {
            renderQualityChart(rows);
        } else if (state.tab === "speed") {
            renderMainChart(rows);
            renderCompareChart(rows);
        } else {
            renderTable(rows);
        }
        writeStateToURL();
    }

    async function init() {
        applyThemeDefaults();
        setupTabs();
        renderTableHeader();
        await loadData();
        renderFilters();
        renderControls();
        renderAll();
    }

    init().catch(console.error);
})();
