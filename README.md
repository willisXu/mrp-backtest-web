# MRP 生產預測回測（網頁版）

靜態網站，部署在 GitHub Pages。資料以密碼加密（PBKDF2-SHA256 → AES-256-GCM），repo 內只有加密後的 `data.enc`；回測計算以 Pyodide 在瀏覽器內執行。

此 repo 由原始專案的 `site/publish.py` 自動產生，請勿直接修改。
