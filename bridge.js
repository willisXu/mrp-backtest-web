/*
 * 網頁版橋接：輸入密碼解密 data.enc（PBKDF2-SHA256 → AES-256-GCM，gzip CSV），
 * 在瀏覽器以 Pyodide 執行原本的 web_app.py，頁面上的 fetch('/api/...') 改由它回應。
 * data.enc 由 site/publish.py 產生。
 */
(() => {
  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
  const PY_PACKAGES = ["numpy", "pandas", "statsmodels", "micropip"];
  const PIP_PACKAGES = [
    "flask==3.1.0", "werkzeug==3.1.3", "jinja2==3.1.5", "itsdangerous==2.2.0",
    "click==8.1.8", "blinker==1.9.0", "flask-cors==5.0.0",
  ];
  const PY_FILES = ["web_app.py", "src/__init__.py", "src/config.py", "src/demand_planning.py"];
  const PW_KEY = "mrp_backtest_pw";

  const store = {
    get(k) { try { return sessionStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { sessionStorage.setItem(k, v); } catch { /* 無痕或封鎖時略過 */ } },
    del(k) { try { sessionStorage.removeItem(k); } catch { /* 略過 */ } },
  };

  const realFetch = window.fetch.bind(window);
  let resolveReady, rejectReady;
  const ready = new Promise((res, rej) => { resolveReady = res; rejectReady = rej; });

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    const i = url.pathname.indexOf("/api/");
    if (i < 0) return realFetch(input, init);
    const pyodide = await ready;
    await new Promise((r) => setTimeout(r, 30)); // 讓 loading 畫面先畫出來
    const handle = pyodide.globals.get("handle");
    try {
      const [status, body, ctype] = handle(init.method || "GET", url.pathname.slice(i) + url.search, init.body ?? null).toJs();
      return new Response(body, { status, headers: { "Content-Type": ctype } });
    } finally {
      handle.destroy();
    }
  };

  // ---------- 解密 ----------
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

  async function decrypt(enc, password) {
    const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveKey"]);
    const key = await crypto.subtle.deriveKey(
      { name: "PBKDF2", hash: "SHA-256", salt: b64(enc.salt), iterations: enc.iter },
      base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
    const gz = await crypto.subtle.decrypt({ name: "AES-GCM", iv: b64(enc.iv) }, key, b64(enc.ct));
    const stream = new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).text();
  }

  // ---------- Pyodide ----------
  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = () => rej(new Error("無法載入 " + src));
      document.head.appendChild(s);
    });
  }

  async function bootPython(setStatus) {
    setStatus("載入計算引擎…（第一次約 20–60 秒，之後會快很多）");
    await loadScript(PYODIDE_URL + "pyodide.js");
    const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
    setStatus("安裝統計套件…");
    await pyodide.loadPackage(PY_PACKAGES);
    await pyodide.pyimport("micropip").install(PIP_PACKAGES);
    setStatus("載入回測程式…");
    pyodide.FS.mkdirTree("/app/src");
    for (const f of PY_FILES) {
      const r = await realFetch("py/" + f, { cache: "no-cache" });
      if (!r.ok) throw new Error("找不到 py/" + f);
      pyodide.FS.writeFile("/app/" + f, await r.text());
    }
    return pyodide;
  }

  // ---------- 登入畫面 ----------
  function buildOverlay() {
    const el = document.createElement("div");
    el.id = "mrp-login";
    el.innerHTML = `
      <style>
        #mrp-login{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;
          background:#f8fafc;font-family:'Noto Sans TC',system-ui,sans-serif;padding:16px}
        #mrp-login .box{background:#fff;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.08);padding:32px;width:100%;max-width:360px}
        #mrp-login h1{font-size:20px;font-weight:700;margin:0 0 4px;color:#1e293b}
        #mrp-login p{font-size:13px;color:#64748b;margin:0 0 20px}
        #mrp-login input{width:100%;box-sizing:border-box;border:1px solid #cbd5e1;border-radius:8px;padding:10px 12px;font-size:15px}
        #mrp-login button{width:100%;margin-top:12px;background:#4f46e5;color:#fff;border:0;border-radius:8px;padding:10px;font-size:15px;cursor:pointer}
        #mrp-login button:disabled{opacity:.6;cursor:default}
        #mrp-login .err{color:#dc2626;font-size:13px;min-height:18px;margin-top:8px}
        #mrp-login .status{color:#475569;font-size:13px;margin-top:8px}
      </style>
      <form class="box">
        <h1>MRP 生產預測回測</h1>
        <p>請輸入密碼以解密資料</p>
        <input type="password" autocomplete="current-password" placeholder="密碼" required>
        <button type="submit">進入</button>
        <div class="err"></div>
        <div class="status"></div>
      </form>`;
    document.body.appendChild(el);
    return el;
  }

  async function start() {
    const overlay = buildOverlay();
    const form = overlay.querySelector("form");
    const input = overlay.querySelector("input");
    const btn = overlay.querySelector("button");
    const err = overlay.querySelector(".err");
    const status = overlay.querySelector(".status");
    const setStatus = (t) => { status.textContent = t; };

    const pyPromise = bootPython(() => {}); // 等密碼時先在背景載入
    pyPromise.catch(() => {});
    let enc;
    try {
      enc = await (await realFetch("data.enc", { cache: "no-cache" })).json();
    } catch (e) {
      err.textContent = "無法讀取資料檔 data.enc";
      rejectReady(e);
      return;
    }

    async function unlock(password) {
      btn.disabled = true; input.disabled = true; err.textContent = "";
      let csv;
      try {
        csv = await decrypt(enc, password);
      } catch {
        store.del(PW_KEY);
        err.textContent = "密碼錯誤";
        btn.disabled = false; input.disabled = false; input.focus();
        return;
      }
      store.set(PW_KEY, password);
      try {
        setStatus("載入計算引擎…（第一次約 20–60 秒，之後會快很多）");
        const pyodide = await pyPromise;
        pyodide.FS.writeFile("/app/sales_agg.csv", csv);
        pyodide.runPython(`
import os, sys
sys.path.insert(0, "/app")
os.environ["DATA_FILE"] = "/app/sales_agg.csv"
from web_app import app
_client = app.test_client()

def handle(method, path, body):
    r = _client.open(path, method=method, data=body,
                     content_type="application/json" if body else None)
    return (r.status_code, r.get_data(as_text=True), r.content_type)
`);
        overlay.remove();
        resolveReady(pyodide);
      } catch (e) {
        console.error(e);
        err.textContent = "計算引擎載入失敗，請重新整理頁面再試一次";
        setStatus(String(e.message || e));
      }
    }

    form.addEventListener("submit", (e) => { e.preventDefault(); unlock(input.value); });
    const saved = store.get(PW_KEY);
    if (saved) unlock(saved); else input.focus();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
