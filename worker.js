/*
 * 計算用 Web Worker：在背景執行緒載入 Pyodide 並執行原本的 web_app.py，
 * 計算時不會卡住頁面。由 bridge.js 建立，訊息格式：
 *   → {type: "init", files: {檔名: CSV}}           ← {type: "ready"} / {type: "error", message}
 *   → {type: "request", id, method, path, body}   ← {type: "response", id, status, body, ctype}
 *   ← {type: "status", text}（載入進度）
 */
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const PY_PACKAGES = ["numpy", "pandas", "statsmodels", "micropip"];
const PIP_PACKAGES = [
  "flask==3.1.0", "werkzeug==3.1.3", "jinja2==3.1.5", "itsdangerous==2.2.0",
  "click==8.1.8", "blinker==1.9.0", "flask-cors==5.0.0",
];
const PY_FILES = ["web_app.py", "src/__init__.py", "src/config.py", "src/demand_planning.py"];

const status = (text) => postMessage({ type: "status", text });

async function boot() {
  status("載入計算引擎…（第一次約 20–60 秒，之後會快很多）");
  importScripts(PYODIDE_URL + "pyodide.js");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  status("安裝統計套件…");
  await pyodide.loadPackage(PY_PACKAGES);
  await pyodide.pyimport("micropip").install(PIP_PACKAGES);
  status("載入回測程式…");
  pyodide.FS.mkdirTree("/app/src");
  for (const f of PY_FILES) {
    const r = await fetch("py/" + f, { cache: "no-cache" });
    if (!r.ok) throw new Error("找不到 py/" + f);
    pyodide.FS.writeFile("/app/" + f, await r.text());
  }
  return pyodide;
}

const booting = boot(); // 使用者輸入密碼時先在背景載入
booting.catch(() => {}); // 錯誤在 init 時回報

let handle = null;

async function init(files) {
  const pyodide = await booting;
  for (const [name, text] of Object.entries(files)) pyodide.FS.writeFile("/app/" + name, text);
  pyodide.runPython(`
import os, sys
sys.path.insert(0, "/app")
os.environ["DATA_FILE"] = "/app/sales_agg.csv"
os.environ["SELLOUT_FILE"] = "/app/sellout.csv"  # 不存在時 Sell-out 選項顯示找不到資料檔
os.environ["RECORDS_FILE"] = "/app/backtest_records.json"  # 共用回測總表
os.environ["SKU_FILE"] = "/app/sku.csv"  # 料號數量（存貨表）
os.environ["SKU_STOCK_FILE"] = "/app/sku_stock.csv"
from web_app import app
_client = app.test_client()

def handle(method, path, body):
    r = _client.open(path, method=method, data=body,
                     content_type="application/json" if body else None)
    return (r.status_code, r.get_data(as_text=True), r.content_type)
`);
  handle = pyodide.globals.get("handle");
}

onmessage = async ({ data: msg }) => {
  if (msg.type === "init") {
    try {
      await init(msg.files);
      postMessage({ type: "ready" });
    } catch (e) {
      postMessage({ type: "error", message: String(e?.message || e) });
    }
  } else if (msg.type === "request") {
    try {
      const [status, body, ctype] = handle(msg.method, msg.path, msg.body).toJs();
      postMessage({ type: "response", id: msg.id, status, body, ctype });
    } catch (e) {
      postMessage({ type: "response", id: msg.id, status: 500, ctype: "application/json",
                    body: JSON.stringify({ error: String(e?.message || e) }) });
    }
  }
};
