/*
 * 回測結果的共用顯示與紀錄：回測頁、需求計劃頁、回測紀錄頁都用這支。
 *   BV.renderRanking(el, results, ctx, onSelect)  方法排名表
 *   BV.renderDetail(el, result, ctx)              單一方法：結果解讀、指標、圖表、逐輪明細
 *   BV.guideHTML()                                「如何閱讀回測結果」說明
 *   BV.Records                                    我的回測紀錄（網頁版以登入密碼加密後存在瀏覽器）
 *   BV.findShared(...)                            共用回測總表（發布時預先跑好）
 * result 格式：{label, method, params, overall:{wmape,mape,bias,rmse,acc_10,acc_20}, rounds:[{months,actuals,forecasts,wmape,bias,...}]}
 */
(() => {
  const FORMULA = "WeightedMA";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => (v * 100).toFixed(1) + "%";
  const pp = (v) => Math.abs(v * 100).toFixed(1);

  function qty(v, unit) {
    if (v == null || isNaN(v)) return "-";
    if (unit === "元" && Math.abs(v) >= 1e4) return (v / 1e4).toLocaleString(undefined, { maximumFractionDigits: 1 }) + " 萬";
    return Math.round(v).toLocaleString() + (unit === "個" ? " 個" : "");
  }

  // ---------- 評等 ----------
  const RATINGS = {
    wmape: [[0.10, "優秀", "green"], [0.20, "良好", "lime"], [0.30, "可接受", "amber"], [Infinity, "需改善", "red"]],
    mape: [[0.10, "高準確度", "green"], [0.20, "良好", "lime"], [0.50, "合理", "amber"], [Infinity, "不準確", "red"]],
    bias: [[0.05, "無明顯偏差", "green"], [0.15, "輕微偏差", "amber"], [Infinity, "系統性偏差", "red"]],
    acc_10: [[0.70, "優秀", "green"], [0.50, "良好", "lime"], [0.30, "可接受", "amber"], [0, "需改善", "red"]],
    acc_20: [[0.80, "優秀", "green"], [0.60, "良好", "lime"], [0.40, "可接受", "amber"], [0, "需改善", "red"]],
  };
  function rating(metric, value) {
    const rules = RATINGS[metric];
    const hit = metric.startsWith("acc")
      ? rules.find(([min]) => value >= min)
      : rules.find(([max]) => Math.abs(value) < max);
    const [, text, color] = hit || rules[rules.length - 1];
    return { text, color };
  }
  const badge = ({ text, color }) =>
    `<span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium bg-${color}-100 text-${color}-700">${text}</span>`;

  // ---------- 整理資料 ----------
  /** API 回傳的回測結果 → 精簡格式（存紀錄用） */
  function compact(r) {
    const rounds = r.rounds || r.round_details || [];
    return {
      key: r.key, label: r.label || r.method, method: r.method, params: r.params || {},
      overall: r.overall,
      rounds: rounds.map((d) => ({ round: d.round, train_end: d.train_end, test_period: d.test_period, months: d.months,
                                   actuals: d.actuals, forecasts: d.forecasts, wmape: d.wmape, bias: d.bias })),
    };
  }

  function periodOf(rounds) {
    if (!rounds?.length || !rounds[0].months) return "";
    return `${rounds[0].months[0]}～${rounds[rounds.length - 1].months.slice(-1)[0]}`;
  }

  /** 依「提前幾個月預測」分組的誤差 */
  function horizonStats(rounds) {
    const H = rounds[0]?.actuals.length || 0;
    return Array.from({ length: H }, (_, h) => {
      let err = 0, act = 0, fc = 0;
      rounds.forEach((d) => { err += Math.abs(d.forecasts[h] - d.actuals[h]); act += d.actuals[h]; fc += d.forecasts[h]; });
      return { h: h + 1, wmape: err / Math.max(act, 1), bias: (fc - act) / Math.max(act, 1) };
    });
  }

  // ---------- 結果解讀 ----------
  function explain(r, ctx = {}) {
    const o = r.overall, rounds = r.rounds || [];
    const unit = ctx.unit || "元";
    const H = rounds[0]?.actuals.length || ctx.config?.forecast_months || 3;
    const T = ctx.config?.training_months || 24;
    const n = rounds.length * H;
    const lines = [];
    const word = unit === "個" ? "個" : "元";

    lines.push(`「<b>${esc(r.label)}</b>」在 ${periodOf(rounds) || esc(ctx.period || "")} 共 ${rounds.length} 輪回測` +
      `（每輪用前 ${T} 個月資料預測接下來 ${H} 個月），wMAPE <b>${pct(o.wmape)}</b> ${badge(rating("wmape", o.wmape))}：` +
      `平均每 100 ${word}實際銷售，預測約差 ${(o.wmape * 100).toFixed(0)} ${word}。`);

    if (Math.abs(o.bias) < 0.05) lines.push(`整體沒有明顯偏高或偏低（Bias ${pct(o.bias)}）。`);
    else if (o.bias > 0) lines.push(`整體預測<b>偏高 ${pp(o.bias)}%</b>：照這個預測備貨，容易庫存過多。`);
    else lines.push(`整體預測<b>偏低 ${pp(o.bias)}%</b>：照這個預測備貨，容易缺貨。`);

    if (n) {
      lines.push(`${n} 個預測值中，${Math.round(o.acc_20 * n)} 個（${pct(o.acc_20)}）落在實際值 ±20% 內，` +
        `${Math.round(o.acc_10 * n)} 個（${pct(o.acc_10)}）在 ±10% 內。`);
    }

    if (rounds.length && H > 1) {
      const hs = horizonStats(rounds);
      const first = hs[0].wmape, last = hs[H - 1].wmape;
      lines.push(`提前 1 個月預測的誤差 ${pct(first)}，提前 ${H} 個月 ${pct(last)}` +
        (last > first + 0.05 ? "：預測越遠越不準，建議每月滾動更新預測。" : "，遠近差異不大。"));
    }

    if (rounds.length > 1) {
      const worst = rounds.reduce((a, b) => (b.wmape > a.wmape ? b : a));
      lines.push(`誤差最大的是第 ${worst.round} 輪（預測 ${esc(periodOf([worst]))}，wMAPE ${pct(worst.wmape)}），` +
        `可檢查這段期間是否有促銷、缺貨或新品上市。`);
    }

    const f = ctx.formula, best = ctx.best;
    if (r.method !== FORMULA && f) {
      const d = o.wmape - f.overall.wmape;
      lines.push(Math.abs(d) < 0.01
        ? `與需求計劃公式（${pct(f.overall.wmape)}）差不多。`
        : `比需求計劃公式（${pct(f.overall.wmape)}）<b>${d < 0 ? "準" : "差"} ${pp(d)} 個百分點</b>。`);
    } else if (r.method === FORMULA && best) {
      if (best.method === FORMULA) lines.push("需求計劃公式是本次比較中<b>最準</b>的方法。");
      else {
        const better = (ctx.results || []).filter((x) => x.overall.wmape < o.wmape).length;
        lines.push(`有 ${better} 種方法比需求計劃公式準，最準的是「${esc(best.label)}」（${pct(best.overall.wmape)}，` +
          `少 ${pp(o.wmape - best.overall.wmape)} 個百分點）。`);
      }
    }

    const advice = o.wmape < 0.2 ? "誤差在可接受範圍，可直接作為備貨參考。"
      : o.wmape < 0.3 ? "可作為參考，但旺季與促銷月份仍需人工調整。"
      : "誤差偏大，建議搭配促銷計畫與通路回報人工判斷，不宜單獨依賴此預測。";
    lines.push(`<b>建議：</b>${advice}`);
    return lines;
  }

  // ---------- 排名表 ----------
  function renderRanking(el, results, ctx = {}, onSelect) {
    const best = results[0];
    const formula = results.find((r) => r.method === FORMULA);
    const formulaRank = formula ? results.indexOf(formula) + 1 : null;
    const maxW = Math.max(...results.map((r) => r.overall.wmape), 0.01);
    let summary = `共比較 ${results.length} 種方法，最準的是「<b>${esc(best.label)}</b>」（wMAPE ${pct(best.overall.wmape)}）`;
    if (formula && formula !== best) {
      summary += `；需求計劃公式排第 <b>${formulaRank}</b> 名（${pct(formula.overall.wmape)}，多 ${pp(formula.overall.wmape - best.overall.wmape)} 個百分點）`;
    } else if (formula) summary += "，就是需求計劃公式";
    summary += "。點選任一列可查看該方法的明細。";
    const skipped = (ctx.skipped || []).map((s) => `${esc(s.label)}：${esc(s.error)}`).join("；");

    el.innerHTML = `
      <div class="flex flex-wrap items-baseline justify-between gap-2 mb-3">
        <h3 class="text-lg font-semibold text-gray-800">方法排名 <span class="text-sm font-normal text-gray-500">依 wMAPE，越低越準</span></h3>
        ${ctx.period ? `<span class="text-xs text-gray-500">回測期間 ${esc(ctx.period)}</span>` : ""}
      </div>
      <p class="text-sm text-gray-600 mb-3">${summary}</p>
      ${skipped ? `<p class="text-xs text-amber-700 mb-3">未列入：${skipped}</p>` : ""}
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead><tr class="bg-gray-50 text-gray-600">
            <th class="px-3 py-2 text-left">#</th><th class="px-3 py-2 text-left">方法</th>
            <th class="px-3 py-2 text-left w-2/5">wMAPE</th><th class="px-3 py-2 text-right">Bias</th>
            <th class="px-3 py-2 text-right">±20% 內</th></tr></thead>
          <tbody>${results.map((r, i) => {
            const isF = r.method === FORMULA;
            const rt = rating("wmape", r.overall.wmape);
            return `<tr data-i="${i}" class="border-b cursor-pointer hover:bg-indigo-50 ${isF ? "bg-amber-50" : i === 0 ? "bg-green-50" : ""}">
              <td class="px-3 py-2 font-medium">${i + 1}</td>
              <td class="px-3 py-2">${esc(r.label)}${isF ? ' <span class="text-xs text-amber-700">◆ 需求計劃</span>' : ""}${i === 0 ? ' <span class="text-xs text-green-700">最準</span>' : ""}</td>
              <td class="px-3 py-2"><div class="flex items-center gap-2">
                <div class="flex-1 bg-gray-100 rounded h-2"><div class="h-2 rounded bg-${rt.color}-400" style="width:${(r.overall.wmape / maxW * 100).toFixed(0)}%"></div></div>
                <span class="w-14 text-right font-semibold text-${rt.color}-700">${pct(r.overall.wmape)}</span></div></td>
              <td class="px-3 py-2 text-right">${r.overall.bias > 0 ? "+" : ""}${pct(r.overall.bias)}</td>
              <td class="px-3 py-2 text-right">${pct(r.overall.acc_20)}</td></tr>`;
          }).join("")}</tbody>
        </table>
      </div>`;
    el.querySelectorAll("tr[data-i]").forEach((tr) => {
      tr.addEventListener("click", () => {
        el.querySelectorAll("tr[data-i]").forEach((x) => x.classList.remove("ring-2", "ring-indigo-400"));
        tr.classList.add("ring-2", "ring-indigo-400");
        onSelect?.(results[+tr.dataset.i]);
      });
    });
  }

  // ---------- 單一方法明細 ----------
  const METRICS = [
    ["wmape", "wMAPE", "總誤差 ÷ 總實際銷售，排名依據", (v) => pct(v)],
    ["bias", "Bias", "正＝預測偏高、負＝偏低", (v) => (v > 0 ? "+" : "") + pct(v)],
    ["acc_20", "±20% 內", "預測落在實際值 ±20% 內的比例", (v) => pct(v)],
    ["acc_10", "±10% 內", "預測落在實際值 ±10% 內的比例", (v) => pct(v)],
    ["mape", "MAPE", "每個月誤差百分比的平均，小月份影響較大", (v) => pct(v)],
    ["rmse", "RMSE", "誤差的典型大小，單位與銷售相同", null],
  ];
  const charts = new WeakMap();

  function renderDetail(el, r, ctx = {}) {
    const unit = ctx.unit || "元";
    const o = r.overall, rounds = r.rounds || [];
    el.innerHTML = `
      <div class="bg-white rounded-xl shadow-sm p-6">
        <div class="flex flex-wrap items-baseline justify-between gap-2 mb-3">
          <h3 class="text-lg font-semibold text-gray-800">${esc(r.label)}</h3>
          <span class="text-xs text-gray-500">${esc(ctx.subtitle || "")}</span>
        </div>
        <div class="bg-indigo-50 rounded-lg p-4 mb-4">
          <div class="font-medium text-indigo-800 mb-2">結果解讀</div>
          <ul class="list-disc pl-5 space-y-1 text-sm text-gray-700">${explain(r, ctx).map((l) => `<li>${l}</li>`).join("")}</ul>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3">${METRICS.map(([k, name, tip, f]) => {
          const v = o[k];
          const rt = RATINGS[k] ? rating(k, v) : null;
          return `<div class="border rounded-lg p-3" title="${tip}">
            <div class="text-xs text-gray-500">${name}</div>
            <div class="text-xl font-bold ${rt ? `text-${rt.color}-700` : "text-gray-800"}">${f ? f(v) : qty(v, unit)}</div>
            <div class="text-xs mt-0.5">${rt ? badge(rt) : '<span class="text-gray-400">越小越好</span>'}</div></div>`;
        }).join("")}</div>
      </div>
      ${rounds.length ? `
      <div class="bg-white rounded-xl shadow-sm p-6">
        <h3 class="text-lg font-semibold text-gray-800 mb-1">預測 vs 實際</h3>
        <p class="text-xs text-gray-500 mb-3">「提前 1 個月」＝每輪預測期的第 1 個月；「提前 ${rounds[0].actuals.length} 個月」＝最後 1 個月。點圖例可隱藏線條。</p>
        <div class="h-72"><canvas></canvas></div>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-6">
        <h3 class="text-lg font-semibold text-gray-800 mb-3">逐輪明細</h3>
        <div class="overflow-x-auto max-h-80">
          <table class="w-full text-sm">
            <thead class="sticky top-0 bg-gray-50 text-gray-600"><tr>
              <th class="px-3 py-2 text-left">輪</th><th class="px-3 py-2 text-left">預測月份</th>
              <th class="px-3 py-2 text-right">實際</th><th class="px-3 py-2 text-right">預測</th>
              <th class="px-3 py-2 text-right">wMAPE</th><th class="px-3 py-2 text-right">Bias</th></tr></thead>
            <tbody>${rounds.map((d) => {
              const rt = rating("wmape", d.wmape);
              const sum = (a) => a.reduce((x, y) => x + y, 0);
              const tip = d.months.map((m, i) => `${m}：實際 ${qty(d.actuals[i], unit)}／預測 ${qty(d.forecasts[i], unit)}`).join("\n");
              return `<tr class="border-b" title="${esc(tip)}">
                <td class="px-3 py-2">${d.round}</td><td class="px-3 py-2">${esc(periodOf([d]))}</td>
                <td class="px-3 py-2 text-right">${qty(sum(d.actuals), unit)}</td>
                <td class="px-3 py-2 text-right">${qty(sum(d.forecasts), unit)}</td>
                <td class="px-3 py-2 text-right font-medium text-${rt.color}-700">${pct(d.wmape)}</td>
                <td class="px-3 py-2 text-right">${d.bias > 0 ? "+" : ""}${pct(d.bias)}</td></tr>`;
            }).join("")}</tbody>
          </table>
        </div>
        <p class="text-xs text-gray-400 mt-2">實際／預測為該輪預測期的合計；滑鼠移到列上可看各月數字。</p>
      </div>` : `<div class="bg-white rounded-xl shadow-sm p-6 text-sm text-gray-500">這筆紀錄沒有逐輪明細。</div>`}`;
    if (rounds.length) drawChart(el.querySelector("canvas"), rounds, unit);
  }

  function drawChart(canvas, rounds, unit) {
    if (!window.Chart) return;
    const H = rounds[0].actuals.length;
    const actual = {}, first = {}, last = {};
    rounds.forEach((d) => d.months.forEach((m, i) => {
      actual[m] = d.actuals[i];
      if (i === 0) first[m] = d.forecasts[i];
      if (i === H - 1) last[m] = d.forecasts[i];
    }));
    const months = Object.keys(actual).sort();
    const pick = (obj) => months.map((m) => (m in obj ? obj[m] : null));
    const datasets = [
      { label: "實際", data: pick(actual), borderColor: "#4f46e5", backgroundColor: "rgba(79,70,229,.08)", fill: true, tension: 0.25 },
      { label: "預測（提前 1 個月）", data: pick(first), borderColor: "#db2777", borderDash: [6, 4], tension: 0.25, spanGaps: true },
    ];
    if (H > 1) datasets.push({ label: `預測（提前 ${H} 個月）`, data: pick(last), borderColor: "#f59e0b", borderDash: [2, 3], tension: 0.25, spanGaps: true });
    charts.get(canvas)?.destroy();
    charts.set(canvas, new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { labels: months, datasets },
      options: {
        responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "top" },
                   tooltip: { callbacks: { label: (c) => `${c.dataset.label}：${qty(c.parsed.y, unit)}` } } },
        scales: { y: { beginAtZero: true, ticks: { callback: (v) => qty(v, unit) } } },
      },
    }));
  }

  // ---------- 說明 ----------
  function guideHTML() {
    return `
      <details class="bg-white rounded-xl shadow-sm p-5 text-sm text-gray-700">
        <summary class="cursor-pointer font-semibold text-gray-800">如何閱讀回測結果</summary>
        <div class="mt-3 space-y-2">
          <p><b>回測是什麼</b>：假裝回到過去，只用當時已知的資料做預測，再和後來的實際銷售比較。每輪用前 24 個月預測接下來 3 個月，
             之後往後移一個月重做，共 12 輪；最後一輪預測到資料的最新月份。</p>
          <p><b>wMAPE（排名依據）</b>：所有月份的誤差加總 ÷ 實際銷售加總。&lt;10% 優秀、10–20% 良好、20–30% 可接受、&gt;30% 需改善。</p>
          <p><b>Bias</b>：預測總和比實際多（+）或少（−）幾 %。偏高容易庫存過多，偏低容易缺貨；±5% 內視為沒有偏差。</p>
          <p><b>±10%／±20% 內</b>：預測值落在實際值 ±10%／±20% 範圍內的比例，越高越好。</p>
          <p><b>MAPE</b>：每個月誤差百分比的平均；銷售很小的月份誤差百分比會被放大，僅供參考。<b>RMSE</b>：誤差的典型大小，單位與銷售相同。</p>
          <p><b>需求計劃公式</b>：(MA-12 + MA-6 + MA-3) ÷ 3，再依 Sell-in／Sell-out 權重加權、四捨五入到百位；未來 3 個月都用同一個數字。</p>
          <p class="text-gray-500"><b>注意</b>：回測只有 12 輪，少數促銷或缺貨月份就可能改變排名；資料少於 38 個月無法回測。
             料號層級的單月波動大，誤差普遍比通路、系列層級高。</p>
        </div>
      </details>`;
  }

  // ---------- 我的紀錄 ----------
  const STORE_KEY = "records_v1";
  const MAX_RECORDS = 30;
  async function rawGet() {
    if (window.mrpSecureStore) return window.mrpSecureStore.get(STORE_KEY);
    try { return localStorage.getItem("mrp_" + STORE_KEY); } catch { return null; }
  }
  async function rawSet(text) {
    if (window.mrpSecureStore) return window.mrpSecureStore.set(STORE_KEY, text);
    localStorage.setItem("mrp_" + STORE_KEY, text);
  }
  const Records = {
    async list() {
      try { return JSON.parse((await rawGet()) || "[]"); } catch { return []; }
    },
    async save(rec) {
      rec.id = rec.id || Date.now().toString(36);
      rec.created_at = rec.created_at || new Date().toLocaleString("zh-TW", { hour12: false });
      let list = (await this.list()).filter((x) => x.signature !== rec.signature);
      list = [rec, ...list].slice(0, MAX_RECORDS);
      for (let n = list.length; n > 0; n = Math.floor(n / 2)) {
        try { await rawSet(JSON.stringify(list.slice(0, n))); return true; } catch { /* 空間不足：減少筆數再試 */ }
      }
      return false;
    },
    async find(signature) { return (await this.list()).find((x) => x.signature === signature) || null; },
    async remove(id) { await rawSet(JSON.stringify((await this.list()).filter((x) => x.id !== id))); },
    async clear() { await rawSet("[]"); },
  };

  /** 同一個查詢（資料、對象、回測設定、方法、資料截止月）得到同一個簽章，用來找現成紀錄 */
  function signature(obj) { return JSON.stringify(obj); }

  // ---------- 共用回測總表 ----------
  let sharedPromise = null;
  function loadShared() {
    sharedPromise ||= fetch("/api/records/shared").then((r) => (r.ok ? r.json() : null)).catch(() => null);
    return sharedPromise;
  }
  const sameConfig = (a, b) => ["training_months", "forecast_months", "rolling_rounds"].every((k) => +a[k] === +b[k]);
  /** 找共用總表中同一對象、同樣回測設定、同樣資料截止月的結果 */
  async function findShared(dataset, channel, series, config, dataEnd) {
    const shared = await loadShared();
    if (!shared || !sameConfig(shared.config, config)) return null;
    if (dataEnd && shared.datasets[dataset]?.data_end !== dataEnd) return null;
    const entry = shared.entries.find((e) => e.id === `${dataset}|${channel}|${series}`);
    return entry?.results ? { entry, shared } : null;
  }

  window.BV = { rating, badge, pct, qty, esc, compact, explain, periodOf, renderRanking, renderDetail, guideHTML,
                Records, signature, loadShared, findShared, FORMULA };
})();
