(function () {
    "use strict";

    const FILTER_DIMS = ["version", "arch", "quant", "runtime", "date"];
    const SIZE_ORDER = ["short", "1000tok", "4000tok", "16000tok"];
    const PALETTE = [
        "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
        "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
        "#0d9488", "#c026d3", "#ca8a04", "#059669", "#e11d48",
    ];

    let state = {
        rows: [],
        meta: null,
        filters: {},       // dim -> Set of active values
        measure: "gen_tps",
        seriesBy: "variant",
        logY: false,
        compareSize: "4000tok",
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

    function sizeIndex(size) {
        const idx = SIZE_ORDER.indexOf(size);
        return idx === -1 ? SIZE_ORDER.length : idx;
    }

    // ---- URL state -----------------------------------------------------

    function readStateFromURL() {
        const params = new URLSearchParams(window.location.search);
        for (const dim of FILTER_DIMS) {
            const v = params.get(dim);
            if (v) state.filters[dim] = new Set(v.split(","));
        }
        if (params.get("measure")) state.measure = params.get("measure");
        if (params.get("seriesBy")) state.seriesBy = params.get("seriesBy");
        if (params.get("logY")) state.logY = params.get("logY") === "1";
        if (params.get("compareSize")) state.compareSize = params.get("compareSize");
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
        params.set("measure", state.measure);
        params.set("seriesBy", state.seriesBy);
        params.set("compareSize", state.compareSize);
        if (state.logY) params.set("logY", "1");
        const qs = params.toString();
        const url = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
        window.history.replaceState(null, "", url);
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

    // ---- controls ----------------------------------------------------

    function renderControls() {
        const measureSel = document.getElementById("measure");
        measureSel.innerHTML = "";
        for (const m of state.meta.measures) {
            const opt = document.createElement("option");
            opt.value = m.key;
            opt.textContent = m.label;
            if (m.key === state.measure) opt.selected = true;
            measureSel.appendChild(opt);
        }
        measureSel.onchange = () => {
            state.measure = measureSel.value;
            renderAll();
        };

        const seriesSel = document.getElementById("seriesBy");
        seriesSel.value = state.seriesBy;
        seriesSel.onchange = () => {
            state.seriesBy = seriesSel.value;
            renderAll();
        };

        const logYBox = document.getElementById("logY");
        logYBox.checked = state.logY;
        logYBox.onchange = () => {
            state.logY = logYBox.checked;
            renderAll();
        };

        const compareSel = document.getElementById("compareSize");
        compareSel.innerHTML = "";
        for (const s of state.meta.size) {
            const opt = document.createElement("option");
            opt.value = s;
            opt.textContent = s;
            if (s === state.compareSize) opt.selected = true;
            compareSel.appendChild(opt);
        }
        if (!state.meta.size.includes(state.compareSize) && state.meta.size.length) {
            state.compareSize = state.meta.size.includes("4000tok") ? "4000tok" : state.meta.size[0];
            compareSel.value = state.compareSize;
        }
        compareSel.onchange = () => {
            state.compareSize = compareSel.value;
            renderAll();
        };
    }

    function onFiltersChanged() {
        renderFilters();
        renderAll();
    }

    // ---- aggregation ---------------------------------------------------

    function mean(values) {
        return values.reduce((a, b) => a + b, 0) / values.length;
    }

    function buildSeries(rows) {
        // Map: seriesValue -> Map: size -> {values: [], rows: []}
        const series = new Map();
        for (const row of rows) {
            const value = row[state.measure];
            if (value === null || value === undefined) continue;
            const seriesValue = row[state.seriesBy] ?? "unknown";
            if (!series.has(seriesValue)) series.set(seriesValue, new Map());
            const bySize = series.get(seriesValue);
            if (!bySize.has(row.size)) bySize.set(row.size, { values: [], rows: [] });
            const bucket = bySize.get(row.size);
            bucket.values.push(value);
            bucket.rows.push(row);
        }
        return series;
    }

    // ---- charts ---------------------------------------------------------

    let mainChart = null;
    let compareChart = null;

    function measureLabel() {
        const m = state.meta.measures.find((m) => m.key === state.measure);
        return m ? m.label : state.measure;
    }

    function renderMainChart(rows) {
        const series = buildSeries(rows);
        const sizes = (state.meta.size || []).slice().sort((a, b) => sizeIndex(a) - sizeIndex(b));

        const datasets = [];
        for (const [seriesValue, bySize] of series) {
            const points = sizes.map((size) => {
                const bucket = bySize.get(size);
                if (!bucket) return null;
                return mean(bucket.values);
            });
            const meta = sizes.map((size) => bySize.get(size) || null);
            datasets.push({
                label: String(seriesValue),
                data: points,
                spanGaps: true,
                borderColor: colorFor(seriesValue),
                backgroundColor: colorFor(seriesValue),
                pointRadius: 4,
                pointHoverRadius: 6,
                tension: 0.15,
                meta,
            });
        }

        const ctx = document.getElementById("mainChart").getContext("2d");
        if (mainChart) mainChart.destroy();
        mainChart = new Chart(ctx, {
            type: "line",
            data: { labels: sizes, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "nearest", intersect: false },
                scales: {
                    x: { title: { display: true, text: "size" } },
                    y: {
                        type: state.logY ? "logarithmic" : "linear",
                        title: { display: true, text: measureLabel() },
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
                                const spreads = bucket.rows.map((r) => r.ttft_spread).filter((v) => v !== null && v !== undefined);
                                if (spreads.length) lines.push(`ttft_spread: ${fmt(mean(spreads))}`);
                                const runs = bucket.rows.map((r) => r.runs).filter((v) => v !== null && v !== undefined);
                                if (runs.length) lines.push(`runs: ${runs.join(", ")}`);
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
        const bucketRows = rows.filter((r) => r.size === state.compareSize && r[state.measure] !== null && r[state.measure] !== undefined);
        const variants = Array.from(new Set(bucketRows.map((r) => r.variant))).sort();
        const seriesValues = Array.from(new Set(bucketRows.map((r) => r[state.seriesBy] ?? "unknown"))).sort();

        const datasets = seriesValues.map((seriesValue) => {
            const data = variants.map((variant) => {
                const matching = bucketRows.filter(
                    (r) => r.variant === variant && (r[state.seriesBy] ?? "unknown") === seriesValue
                );
                if (!matching.length) return null;
                return mean(matching.map((r) => r[state.measure]));
            });
            return {
                label: String(seriesValue),
                data,
                backgroundColor: colorFor(seriesValue),
            };
        });

        const ctx = document.getElementById("compareChart").getContext("2d");
        if (compareChart) compareChart.destroy();
        compareChart = new Chart(ctx, {
            type: "bar",
            data: { labels: variants, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: "variant" } },
                    y: {
                        type: state.logY ? "logarithmic" : "linear",
                        title: { display: true, text: `${measureLabel()} @ ${state.compareSize}` },
                    },
                },
                plugins: { legend: { position: "bottom" } },
            },
        });
    }

    // ---- table -----------------------------------------------------------

    const TABLE_COLUMNS = [
        "date", "variant", "version", "arch", "quant", "runtime", "size",
        "prompt_tokens", "completion_tokens", "gen_tps", "prefill_tps",
        "ttft", "ttft_spread", "watts", "tokens_per_wh",
    ];

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
            const suspect = row.ttft_spread !== null && row.ttft_spread !== undefined && row.ttft_spread > 0.25;
            if (suspect) tr.className = "suspect";
            for (const col of TABLE_COLUMNS) {
                const td = document.createElement("td");
                td.textContent = fmt(row[col]);
                tr.appendChild(td);
            }
            if (suspect) tr.title = "ttft_spread > 0.25: possible cache hit, see README";
            tbody.appendChild(tr);
        }

        document.querySelectorAll("#rowsTable th").forEach((th) => {
            th.classList.remove("sorted", "asc");
            if (th.dataset.key === state.sortKey) {
                th.classList.add("sorted");
                if (state.sortAsc) th.classList.add("asc");
            }
        });
    }

    function setupTableSorting() {
        document.querySelectorAll("#rowsTable th").forEach((th) => {
            th.onclick = () => {
                const key = th.dataset.key;
                if (state.sortKey === key) {
                    state.sortAsc = !state.sortAsc;
                } else {
                    state.sortKey = key;
                    state.sortAsc = true;
                }
                renderAll();
            };
        });
    }

    // ---- top-level render -------------------------------------------------

    function renderAll() {
        const rows = filteredRows();
        renderMainChart(rows);
        renderCompareChart(rows);
        renderTable(rows);
        writeStateToURL();
    }

    async function init() {
        await loadData();
        renderFilters();
        renderControls();
        setupTableSorting();
        renderAll();
    }

    init();
})();
