/*
 * 網頁版橋接：輸入密碼解密 data.enc（PBKDF2-SHA256 → AES-256-GCM，gzip CSV），
 * 交給 worker.js 在背景執行緒以 Pyodide 執行原本的 web_app.py，
 * 頁面上的 fetch('/api/...') 改由它回應。data.enc 由 site/publish.py 產生。
 */
(() => {
  const PW_KEY = "mrp_backtest_pw";

  const store = {
    get(k) { try { return sessionStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { sessionStorage.setItem(k, v); } catch { /* 無痕或封鎖時略過 */ } },
    del(k) { try { sessionStorage.removeItem(k); } catch { /* 略過 */ } },
  };

  // ---------- 計算 Worker ----------
  // 沿用 bridge.js 的版本號（publish.py 加上的 ?v=），避免載入快取裡的舊 worker.js
  const VERSION = document.currentScript ? new URL(document.currentScript.src).search : "";
  const worker = new Worker("worker.js" + VERSION); // 立即開始載入，使用者輸入密碼時同步進行
  const pending = new Map();
  let nextId = 1;
  let lastStatus = "";
  let onStatus = () => {};
  let onInit = null;

  worker.onmessage = ({ data: msg }) => {
    if (msg.type === "status") { lastStatus = msg.text; onStatus(msg.text); }
    else if (msg.type === "ready" || msg.type === "error") { onInit?.(msg); }
    else if (msg.type === "response") {
      pending.get(msg.id)?.(msg);
      pending.delete(msg.id);
    }
  };

  function initWorker(csv) {
    return new Promise((resolve, reject) => {
      onInit = (msg) => (msg.type === "ready" ? resolve() : reject(new Error(msg.message)));
      worker.postMessage({ type: "init", csv });
    });
  }

  const realFetch = window.fetch.bind(window);
  let resolveReady, rejectReady;
  const ready = new Promise((res, rej) => { resolveReady = res; rejectReady = rej; });
  ready.catch(() => {}); // 失敗已顯示在登入畫面；個別 fetch 仍會收到錯誤

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    const i = url.pathname.indexOf("/api/");
    if (i < 0) return realFetch(input, init);
    await ready;
    const id = nextId++;
    const msg = await new Promise((resolve) => {
      pending.set(id, resolve);
      worker.postMessage({ type: "request", id, method: init.method || "GET",
                           path: url.pathname.slice(i) + url.search, body: init.body ?? null });
    });
    return new Response(msg.body, { status: msg.status, headers: { "Content-Type": msg.ctype } });
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

    // 送出事件要在下載 data.enc 之前就攔住，否則太早按「進入」會讓表單以預設方式送出並重新整理頁面
    form.addEventListener("submit", (e) => { e.preventDefault(); unlock(input.value); });

    const encPromise = realFetch("data.enc", { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
    encPromise.catch((e) => {
      err.textContent = "無法讀取資料檔 data.enc";
      btn.disabled = true; input.disabled = true;
      rejectReady(e);
    });

    async function unlock(password) {
      btn.disabled = true; input.disabled = true; err.textContent = "";
      let enc, csv;
      try {
        enc = await encPromise;
      } catch {
        return; // 錯誤訊息已由 encPromise.catch 顯示
      }
      try {
        csv = await decrypt(enc, password);
      } catch {
        store.del(PW_KEY);
        err.textContent = "密碼錯誤";
        btn.disabled = false; input.disabled = false; input.focus();
        return;
      }
      store.set(PW_KEY, password);
      onStatus = (t) => { status.textContent = t; };
      onStatus(lastStatus || "載入計算引擎…");
      try {
        await initWorker(csv);
        overlay.remove();
        resolveReady();
      } catch (e) {
        console.error(e);
        err.textContent = "計算引擎載入失敗，請重新整理頁面再試一次";
        status.textContent = String(e.message || e);
      }
    }

    const saved = store.get(PW_KEY);
    if (saved) unlock(saved); else input.focus();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
