"""
Flask web app for the MRP backtest workspace.

The original project references forecast modules that are not present in this
workspace, so this app provides a compatible, self-contained runtime using the
existing demand planning code and lightweight forecasting implementations.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import SystemConfig
from src.demand_planning import (
    DemandPlanningCalculator,
    DemandPlanningConfig,
    InventoryStatus,
    SalesData,
    WeightedMAForecaster,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

DATA_CACHE: dict[str, pd.DataFrame] = {}

# 資料集：sellin＝通路×系列的銷貨淨額（元）；sellout＝通路×料號的銷貨數量（個）；
# sku＝料號的 Sell-in 數量（個，存貨可用月數表），另帶同料號的 Sell-out 數量供需求計劃公式使用。
# 檔案可為 .xlsx 或彙總後的 .csv（網頁版由 site/publish.py 產生，以環境變數指定）
DATASETS = {
    "sellin": {
        "file": Path(os.environ.get("DATA_FILE", BASE_DIR / "Sellin_數據_扁平化.xlsx")),
        "columns": {"年月": "date", "通路": "channel", "系列": "series", "銷貨淨額": "sales"},
        "label": "Sell-in 銷貨淨額", "item_label": "系列", "unit": "元",
    },
    "sellout": {
        "file": Path(os.environ.get("SELLOUT_FILE", BASE_DIR / "Sellout_數據_扁平化.xlsx")),
        "columns": {"年月": "date", "通路": "channel", "料號": "series", "銷貨數量": "sales"},
        "label": "Sell-out 銷貨數量", "item_label": "料號", "unit": "個",
    },
    "sku": {
        "file": Path(os.environ.get("SKU_FILE", BASE_DIR / "SKU_數量_扁平化.xlsx")),
        "columns": {"年月": "date", "料號": "series", "Sellin數量": "sales"},
        "label": "Sell-in 數量（料號）", "item_label": "料號", "unit": "個",
    },
}
# 料號庫存參考（存貨可用月數表最新一期）：xlsx 為「庫存參考」工作表；網頁版為獨立 CSV
SKU_STOCK_FILE = Path(os.environ["SKU_STOCK_FILE"]) if os.environ.get("SKU_STOCK_FILE") else None
DATA_SHEET = "銷售明細"
SKU_LIST_SIZE = 150  # Sell-out 料號選單只列近 12 個月銷量最高的前 N 個（sku 資料集列出全部）

FORMULA_INFO = {
    "Naive": {"name": "Naive", "params": []},
    "Seasonal_Naive": {
        "name": "Seasonal Naive",
        "params": [
            {"name": "seasonal_period", "label": "季節週期", "default": 12, "min": 2, "max": 24, "step": 1}
        ],
    },
    "SMA": {
        "name": "SMA",
        "params": [
            {"name": "window", "label": "移動平均期間", "default": 3, "min": 2, "max": 24, "step": 1}
        ],
    },
    "SES": {
        "name": "SES",
        "params": [
            {"name": "alpha", "label": "Alpha", "default": 0.2, "min": 0.05, "max": 0.95, "step": 0.05}
        ],
    },
    "Holt": {
        "name": "Holt",
        "params": [
            {"name": "alpha", "label": "Alpha", "default": 0.2, "min": 0.05, "max": 0.95, "step": 0.05},
            {"name": "beta", "label": "Beta", "default": 0.1, "min": 0.01, "max": 0.95, "step": 0.01},
        ],
    },
    "HoltWinters": {
        "name": "Holt-Winters",
        "params": [
            {"name": "alpha", "label": "Alpha", "default": 0.2, "min": 0.05, "max": 0.95, "step": 0.05},
            {"name": "beta", "label": "Beta", "default": 0.1, "min": 0.01, "max": 0.95, "step": 0.01},
            {"name": "gamma", "label": "Gamma", "default": 0.2, "min": 0.01, "max": 0.95, "step": 0.01},
            {
                "name": "seasonal_periods",
                "label": "季節週期",
                "default": 12,
                "min": 2,
                "max": 24,
                "step": 1,
            },
            {
                "name": "seasonal_type",
                "label": "季節型態",
                "default": "add",
                "options": ["add", "mul"],
            },
        ],
    },
    "WeightedMA": {
        "name": "Weighted MA",
        "params": [
            {
                "name": "sell_in_weight",
                "label": "Sell-in 權重",
                "default": 0.7,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            {
                "name": "sell_out_weight",
                "label": "Sell-out 權重",
                "default": 0.3,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            {"name": "ma_period_1", "label": "MA-1", "default": 12, "min": 1, "max": 24, "step": 1},
            {"name": "ma_period_2", "label": "MA-2", "default": 6, "min": 1, "max": 24, "step": 1},
            {"name": "ma_period_3", "label": "MA-3", "default": 3, "min": 1, "max": 24, "step": 1},
            {
                "name": "use_sell_in_as_sell_out",
                "label": "以 Sell-in 代替 Sell-out",
                "default": True,
                "type": "checkbox",
            },
        ],
    },
}


# 各方法的說明（白話、公式、適合／注意、參數意義）在 static/method_docs.js

FORMULA_LABEL = "需求計劃公式（現行）"
DEFAULT_FORMULA_PARAMS = {"ma_period_1": 12, "ma_period_2": 6, "ma_period_3": 3,
                          "sell_in_weight": 0.7, "sell_out_weight": 0.3, "use_sell_in_as_sell_out": True}


def method_label(method: str, params: dict[str, Any] | None = None) -> str:
    """方法的中文顯示名稱，回測頁、需求計劃頁與回測紀錄共用。"""
    params = params or {}
    if method == "Naive":
        return "Naive（上月）"
    if method == "Seasonal_Naive":
        return "Seasonal Naive（去年同期）"
    if method == "SMA":
        return f"SMA-{params.get('window', 3):g}"
    if method == "SES":
        return f"SES（α={params.get('alpha', 0.2):g}）"
    if method == "Holt":
        return f"Holt（α={params.get('alpha', 0.2):g}, β={params.get('beta', 0.1):g}）"
    if method == "HoltWinters":
        return "Holt-Winters（乘法）" if params.get("seasonal_type") == "mul" else "Holt-Winters（加法）"
    if method == "WeightedMA":
        periods = [params.get(f"ma_period_{i}", DEFAULT_FORMULA_PARAMS[f"ma_period_{i}"]) for i in (1, 2, 3)]
        weight = params.get("sell_in_weight", 0.7)
        if periods != [12, 6, 3] or weight != 0.7:
            return f"需求計劃公式（MA {'/'.join(f'{p:g}' for p in periods)}，Sell-in {weight:g}）"
        return FORMULA_LABEL
    return method


# 「快速比較」與共用回測紀錄使用的方法組合
DEFAULT_COMPARE_METHODS = [
    {"method": "WeightedMA", "params": dict(DEFAULT_FORMULA_PARAMS)},
    {"method": "Naive", "params": {}},
    {"method": "Seasonal_Naive", "params": {"seasonal_period": 12}},
    {"method": "SMA", "params": {"window": 3}},
    {"method": "SMA", "params": {"window": 6}},
    {"method": "SMA", "params": {"window": 12}},
    {"method": "SES", "params": {"alpha": 0.2}},
    {"method": "SES", "params": {"alpha": 0.3}},
    {"method": "Holt", "params": {"alpha": 0.2, "beta": 0.1}},
    {"method": "HoltWinters", "params": {"alpha": 0.2, "beta": 0.1, "gamma": 0.2, "seasonal_periods": 12, "seasonal_type": "add"}},
    {"method": "HoltWinters", "params": {"alpha": 0.2, "beta": 0.1, "gamma": 0.2, "seasonal_periods": 12, "seasonal_type": "mul"}},
]

# 共用回測紀錄（build_backtest_records.py 產生；網頁版由 publish.py 一併加密）
RECORDS_FILE = Path(os.environ.get("RECORDS_FILE", BASE_DIR / "output" / "backtest_records.json"))


@dataclass
class StandardColumns:
    date: str = "date"
    channel: str = "channel"
    series: str = "series"
    sales: str = "sales"


COLS = StandardColumns()


class NaiveForecaster:
    def fit(self, data: pd.Series) -> "NaiveForecaster":
        self.data = pd.Series(data).astype(float)
        return self

    def predict(self, steps: int) -> np.ndarray:
        last = float(self.data.iloc[-1]) if len(self.data) else 0.0
        return np.repeat(last, steps)


class SeasonalNaiveForecaster:
    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = max(1, int(seasonal_period))

    def fit(self, data: pd.Series) -> "SeasonalNaiveForecaster":
        self.data = pd.Series(data).astype(float)
        return self

    def predict(self, steps: int) -> np.ndarray:
        if len(self.data) < self.seasonal_period:
            return NaiveForecaster().fit(self.data).predict(steps)
        history = self.data.iloc[-self.seasonal_period :].to_numpy()
        return np.array([history[i % len(history)] for i in range(steps)], dtype=float)


class SMAForecaster:
    def __init__(self, window: int = 3):
        self.window = max(1, int(window))

    def fit(self, data: pd.Series) -> "SMAForecaster":
        self.data = pd.Series(data).astype(float)
        return self

    def predict(self, steps: int) -> np.ndarray:
        window_data = self.data.iloc[-self.window :]
        value = float(window_data.mean()) if len(window_data) else 0.0
        return np.repeat(value, steps)


class SESForecaster:
    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)

    def fit(self, data: pd.Series) -> "SESForecaster":
        values = pd.Series(data).astype(float).to_numpy()
        if len(values) == 0:
            self.level = 0.0
            return self
        level = values[0]
        for value in values[1:]:
            level = self.alpha * value + (1 - self.alpha) * level
        self.level = float(level)
        return self

    def predict(self, steps: int) -> np.ndarray:
        return np.repeat(self.level, steps)


class HoltForecaster:
    def __init__(self, alpha: float = 0.2, beta: float = 0.1):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def fit(self, data: pd.Series) -> "HoltForecaster":
        values = pd.Series(data).astype(float).to_numpy()
        if len(values) == 0:
            self.level = 0.0
            self.trend = 0.0
            return self
        if len(values) == 1:
            self.level = float(values[0])
            self.trend = 0.0
            return self
        level = values[0]
        trend = values[1] - values[0]
        for value in values[1:]:
            prev_level = level
            level = self.alpha * value + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
        self.level = float(level)
        self.trend = float(trend)
        return self

    def predict(self, steps: int) -> np.ndarray:
        return np.array([self.level + (step + 1) * self.trend for step in range(steps)], dtype=float)


class HoltWintersForecaster:
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.1,
        gamma: float = 0.2,
        seasonal_periods: int = 12,
        seasonal_type: str = "add",
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.seasonal_periods = max(2, int(seasonal_periods))
        self.seasonal_type = seasonal_type

    def fit(self, data: pd.Series) -> "HoltWintersForecaster":
        self.data = pd.Series(data).astype(float)
        self.model = None
        if len(self.data) >= self.seasonal_periods * 2:
            try:
                self.model = ExponentialSmoothing(
                    self.data,
                    trend="add",
                    seasonal=self.seasonal_type,
                    seasonal_periods=self.seasonal_periods,
                    initialization_method="estimated",
                ).fit(
                    smoothing_level=self.alpha,
                    smoothing_trend=self.beta,
                    smoothing_seasonal=self.gamma,
                    optimized=False,
                )
            except Exception:
                self.model = None
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.forecast(steps), dtype=float)
        return SMAForecaster(window=min(self.seasonal_periods, max(1, len(self.data)))).fit(self.data).predict(steps)


def _normalize_month(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    if values.notna().any():
        return values.dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime(series.astype(str), format="%Y-%m", errors="coerce")


def _standardize_dataframe(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    missing = [source for source in rename_map if source not in df.columns]
    if missing:
        raise ValueError(f"資料欄位不足: {missing}")

    standardized = df.rename(columns=rename_map)[list(rename_map.values())].copy()
    if COLS.channel not in standardized.columns:  # 沒有通路維度的資料集（sku）
        standardized[COLS.channel] = "ALL"
    standardized[COLS.date] = _normalize_month(standardized[COLS.date])
    standardized[COLS.channel] = standardized[COLS.channel].astype(str).str.strip()
    standardized[COLS.series] = standardized[COLS.series].fillna("未分類").astype(str).str.strip()
    standardized[COLS.sales] = pd.to_numeric(standardized[COLS.sales], errors="coerce").fillna(0.0)
    standardized = standardized.dropna(subset=[COLS.date])
    if "品名" in df.columns:  # 料號的顯示名稱：規格 品名（停產）
        name = df["品名"].fillna("").astype(str).str.strip()
        if "規格" in df.columns:
            name = df["規格"].fillna("").astype(str).str.strip() + " " + name
        if "備註" in df.columns:
            name = name + np.where(df["備註"].astype(str).str.contains("停產"), "（停產）", "")
        standardized["name"] = name.str.strip()
    if "Sellout數量" in df.columns:  # 同料號的 Sell-out，需求計劃公式的 Sell-out 部分
        standardized["sales_out"] = pd.to_numeric(df["Sellout數量"], errors="coerce").fillna(0.0)
    return standardized


def load_data(dataset: str = "sellin") -> pd.DataFrame | None:
    if dataset in DATA_CACHE:
        return DATA_CACHE[dataset]

    spec = DATASETS.get(dataset)
    if spec is None or not spec["file"].exists():
        return None

    try:
        dtype = {"年月": str, "料號": str}
        if spec["file"].suffix.lower() == ".csv":
            raw = pd.read_csv(spec["file"], dtype=dtype)
        else:
            raw = pd.read_excel(spec["file"], sheet_name=DATA_SHEET, dtype=dtype)
        df = _standardize_dataframe(raw, spec["columns"])
        DATA_CACHE[dataset] = df
        return df
    except Exception as exc:
        print(f"load_data({dataset}) failed: {exc}")
        return None


def _dataset_from(value: Any) -> str:
    return value if value in DATASETS else "sellin"


def get_channel_time_series(df: pd.DataFrame, channel: str, series: str | None = None,
                            column: str = COLS.sales) -> pd.Series:
    filtered = df
    if channel and channel != "ALL":
        filtered = filtered[filtered[COLS.channel] == channel]
    if series and series != "ALL":
        filtered = filtered[filtered[COLS.series] == series]

    ts = filtered.groupby(COLS.date)[column].sum().sort_index()
    if len(ts):
        # 補到資料集最後一個月：已停賣的料號／通路，近期應視為 0 而不是截在最後有銷售的月份
        full_range = pd.date_range(ts.index.min(), df[COLS.date].max(), freq="MS")
        ts = ts.reindex(full_range, fill_value=0.0)
    return ts.astype(float)


def get_sell_out_series(df: pd.DataFrame, channel: str, series: str | None, index: pd.Index) -> pd.Series | None:
    """資料集有 Sell-out 欄（sku）時，回傳與 Sell-in 對齊的 Sell-out 時間序列；沒有則回傳 None。"""
    if "sales_out" not in df.columns:
        return None
    ts = get_channel_time_series(df, channel, series, column="sales_out")
    return ts.reindex(index, fill_value=0.0)


def create_forecaster(method: str, params: dict[str, Any]):
    if method == "Naive":
        return NaiveForecaster()
    if method == "Seasonal_Naive":
        return SeasonalNaiveForecaster(params.get("seasonal_period", 12))
    if method == "SMA":
        return SMAForecaster(params.get("window", 3))
    if method == "SES":
        return SESForecaster(params.get("alpha", 0.2))
    if method == "Holt":
        return HoltForecaster(params.get("alpha", 0.2), params.get("beta", 0.1))
    if method == "HoltWinters":
        return HoltWintersForecaster(
            alpha=params.get("alpha", 0.2),
            beta=params.get("beta", 0.1),
            gamma=params.get("gamma", 0.2),
            seasonal_periods=params.get("seasonal_periods", 12),
            seasonal_type=params.get("seasonal_type", "add"),
        )
    if method == "WeightedMA":
        return WeightedMAForecaster(
            sell_in_weight=params.get("sell_in_weight", 0.7),
            sell_out_weight=params.get("sell_out_weight", 0.3),
            ma_period_1=params.get("ma_period_1", 12),
            ma_period_2=params.get("ma_period_2", 6),
            ma_period_3=params.get("ma_period_3", 3),
            use_sell_in_as_sell_out=params.get("use_sell_in_as_sell_out", True),
        )
    return SMAForecaster(window=3)


def run_backtest(
    ts: pd.Series,
    forecaster,
    training_months: int = 24,
    forecast_months: int = 3,
    rolling_rounds: int = 12,
    sell_out: pd.Series | None = None,
) -> dict[str, Any]:
    """滾動回測。sell_out：與 ts 對齊的 Sell-out 序列，需求計劃公式（WeightedMA）會用來計算 Sell-out 部分。"""
    min_required = training_months + forecast_months + rolling_rounds - 1
    if len(ts) < min_required:
        return {"error": f"資料不足，至少需要 {min_required} 個月"}

    all_actuals: list[float] = []
    all_forecasts: list[float] = []
    round_results: list[dict[str, Any]] = []

    # 回測期間對齊資料最後一個月：最後一輪的預測期結束在最新月份；
    # 每輪以預測期前 training_months 個月訓練，逐輪往後滑動一個月
    offset = len(ts) - min_required
    for round_idx in range(rolling_rounds):
        train_start = offset + round_idx
        train_end = train_start + training_months
        test_start = train_end
        test_end = test_start + forecast_months

        train_data = ts.iloc[train_start:train_end]
        test_data = ts.iloc[test_start:test_end]
        try:
            if sell_out is not None and isinstance(forecaster, WeightedMAForecaster):
                forecaster.fit(train_data, sell_out.iloc[train_start:train_end])
            else:
                forecaster.fit(train_data)
            predictions = np.asarray(forecaster.predict(forecast_months), dtype=float)
        except Exception as exc:
            round_results.append({"round": round_idx + 1, "error": str(exc)})
            continue

        actuals = test_data.to_numpy(dtype=float)
        predictions = np.maximum(predictions, 0.0)
        denom = np.maximum(actuals, 1.0)
        mape = float(np.mean(np.abs(predictions - actuals) / denom))
        wmape = float(np.sum(np.abs(predictions - actuals)) / max(np.sum(actuals), 1.0))
        bias = float((np.sum(predictions) - np.sum(actuals)) / max(np.sum(actuals), 1.0))

        round_results.append(
            {
                "round": round_idx + 1,
                "train_end": str(ts.index[train_end - 1].date()),
                "test_period": f"{ts.index[test_start].date()} ~ {ts.index[test_end - 1].date()}",
                "months": [d.strftime("%Y-%m") for d in ts.index[test_start:test_end]],
                "actuals": [float(x) for x in actuals],
                "forecasts": [float(x) for x in predictions],
                "mape": mape,
                "wmape": wmape,
                "bias": bias,
            }
        )
        all_actuals.extend(actuals.tolist())
        all_forecasts.extend(predictions.tolist())

    if not all_actuals:
        return {"error": "回測失敗，沒有產生可用結果"}

    actual_arr = np.asarray(all_actuals, dtype=float)
    pred_arr = np.asarray(all_forecasts, dtype=float)
    if np.sum(actual_arr) <= 0:
        # 例如已停止出貨的通路：實際與預測都是 0，指標會誤顯示為 0% 誤差
        return {"error": f"回測期間（{ts.index[offset + training_months].date()} ~ {ts.index[-1].date()}）沒有銷售，無法評估"}
    denom = np.maximum(actual_arr, 1.0)
    errors = np.abs(pred_arr - actual_arr) / denom

    overall = {
        "wmape": float(np.sum(np.abs(pred_arr - actual_arr)) / max(np.sum(actual_arr), 1.0)),
        "mape": float(np.mean(errors)),
        "bias": float((np.sum(pred_arr) - np.sum(actual_arr)) / max(np.sum(actual_arr), 1.0)),
        "rmse": float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2))),
        "acc_10": float(np.mean(errors <= 0.10)),
        "acc_20": float(np.mean(errors <= 0.20)),
    }
    return {
        # 前端（templates/index.html）讀取的欄位
        "overall": overall,
        "all_actuals": all_actuals,
        "all_forecasts": all_forecasts,
        "round_details": [item for item in round_results if "error" not in item],
        "backtest_period": f"{ts.index[offset + training_months].date()} ~ {ts.index[-1].date()}",
        "round_results": round_results,
        "overall_wmape": float(np.sum(np.abs(pred_arr - actual_arr)) / max(np.sum(actual_arr), 1.0)),
        "overall_mape": float(np.mean(errors)),
        "overall_bias": float((np.sum(pred_arr) - np.sum(actual_arr)) / max(np.sum(actual_arr), 1.0)),
        "overall_rmse": float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2))),
        "accuracy_10": float(np.mean(errors <= 0.10)),
        "accuracy_20": float(np.mean(errors <= 0.20)),
        "n_predictions": int(len(pred_arr)),
    }


def _build_series_list(df: pd.DataFrame, dataset: str) -> list[dict[str, str]]:
    if dataset in ("sellout", "sku"):
        recent = df[df[COLS.date] > df[COLS.date].max() - pd.DateOffset(months=12)]
        top = recent.groupby(COLS.series)[COLS.sales].sum().sort_values(ascending=False)
        top = top.reindex(top.index.union(df[COLS.series].unique()), fill_value=0).sort_values(ascending=False)
        if dataset == "sellout":
            top = top.head(SKU_LIST_SIZE)
        names = df.drop_duplicates(COLS.series, keep="last").set_index(COLS.series)["name"]
        return [{"key": "ALL", "name": "全部料號"}] + [
            {"key": sku, "name": f"{sku} {names.get(sku, '')}".strip()} for sku in top.index
        ]
    series_sales = df.groupby(COLS.series)[COLS.sales].sum().sort_values(ascending=False)
    top_series = series_sales.head(15).index.tolist()
    return [{"key": "ALL", "name": "全部系列"}] + [{"key": item, "name": item} for item in top_series]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demand-planning")
def demand_planning_page():
    return render_template("demand_planning.html")


@app.route("/records")
def records_page():
    return render_template("records.html")


@app.route("/api/data/info")
def get_data_info():
    dataset = _dataset_from(request.args.get("dataset"))
    df = load_data(dataset)
    if df is None:
        return jsonify({"error": "找不到資料檔"}), 500

    channels = sorted(c for c in df[COLS.channel].dropna().unique().tolist() if c != "ALL")
    return jsonify(
        {
            "channels": ["ALL"] + channels,
            "channel_names": {"ALL": "全部通路", **{channel: channel for channel in channels}},
            "series": _build_series_list(df, dataset),
            "year_range": [int(df[COLS.date].dt.year.min()), int(df[COLS.date].dt.year.max())],
            "total_records": int(len(df)),
            "data_source": DATASETS[dataset]["file"].name,
            "dataset": dataset,
            "datasets": {key: {"label": spec["label"], "item_label": spec["item_label"], "unit": spec["unit"],
                               "available": spec["file"].exists()} for key, spec in DATASETS.items()},
            "data_end": df[COLS.date].max().strftime("%Y-%m"),
        }
    )


@app.route("/api/config/default")
def get_default_config():
    config = SystemConfig()
    return jsonify(
        {
            "backtest": {
                "training_months": config.backtest.training_months,
                "forecast_months": config.backtest.forecast_months,
                "rolling_rounds": config.backtest.rolling_rounds,
                "seasonal_period": config.backtest.seasonal_period,
            },
            "sma": {"periods": config.sma.periods},
            "wma": {"periods": config.wma.periods, "weights": config.wma.weights},
            "ses": {"alphas": config.ses.alphas},
            "holt": {"alphas": config.holt.alphas, "betas": config.holt.betas},
            "holt_winters": {
                "alphas": config.holt_winters.alphas,
                "betas": config.holt_winters.betas,
                "gammas": config.holt_winters.gammas,
                "seasonal_periods": config.holt_winters.seasonal_periods,
                "seasonal_types": config.holt_winters.seasonal_types,
            },
            "promotion": config.promotion.promotion_coefficients,
        }
    )


@app.route("/api/formula/info")
def get_formula_info():
    return jsonify(FORMULA_INFO)


@app.route("/api/backtest/run", methods=["POST"])
def run_backtest_api():
    payload = request.get_json(silent=True) or {}
    df = load_data(_dataset_from(payload.get("dataset")))
    if df is None:
        return jsonify({"error": "找不到資料檔"}), 500

    channel = payload.get("channel", "ALL")
    series = payload.get("series", "ALL")
    method = payload.get("method", "SMA")
    params = payload.get("params", {})
    backtest_config = payload.get("backtest_config", {})

    ts = get_channel_time_series(df, channel, series)
    if len(ts) == 0:
        return jsonify({"error": f"沒有資料: {channel} / {series}"}), 400

    result = run_backtest(
        ts,
        create_forecaster(method, params),
        training_months=backtest_config.get("training_months", 24),
        forecast_months=backtest_config.get("forecast_months", 3),
        rolling_rounds=backtest_config.get("rolling_rounds", 12),
        sell_out=get_sell_out_series(df, channel, series, ts.index),
    )
    result.update(
        {
            "channel": channel,
            "series": series,
            "method": method,
            "params": params,
            "label": method_label(method, params),
            "data_length": int(len(ts)),
            "data_range": f"{ts.index.min().date()} ~ {ts.index.max().date()}",
        }
    )
    return jsonify(result)


@app.route("/api/backtest/compare", methods=["POST"])
def compare_methods_api():
    payload = request.get_json(silent=True) or {}
    df = load_data(_dataset_from(payload.get("dataset")))
    if df is None:
        return jsonify({"error": "找不到資料檔"}), 500

    channel = payload.get("channel", "ALL")
    series = payload.get("series", "ALL")
    methods_config = payload.get("methods") or DEFAULT_COMPARE_METHODS
    backtest_config = payload.get("backtest_config", {})

    ts = get_channel_time_series(df, channel, series)
    if len(ts) == 0:
        return jsonify({"error": f"沒有資料: {channel} / {series}"}), 400
    sell_out = get_sell_out_series(df, channel, series, ts.index)

    results = []
    for item in methods_config:
        method = item.get("method", "SMA")
        params = item.get("params", {})
        result = run_backtest(
            ts,
            create_forecaster(method, params),
            training_months=backtest_config.get("training_months", 24),
            forecast_months=backtest_config.get("forecast_months", 3),
            rolling_rounds=backtest_config.get("rolling_rounds", 12),
            sell_out=sell_out,
        )
        result.update({"method": method, "params": params, "label": method_label(method, params)})
        results.append(result)

    ranking = sorted(
        [item for item in results if "overall_wmape" in item],
        key=lambda item: item["overall_wmape"],
    )
    if not ranking:
        return jsonify({"error": next((item["error"] for item in results if "error" in item), "沒有可比較的結果")}), 400
    return jsonify({"results": ranking, "ranking": ranking,
                    "skipped": [{"label": item["label"], "error": item["error"]} for item in results if "error" in item]})


@app.route("/api/demand-planning/calculate", methods=["POST"])
def calculate_demand_planning():
    payload = request.get_json(silent=True) or {}
    config_data = payload.get("config", {})
    config = DemandPlanningConfig(
        sell_in_weight=config_data.get("sell_in_weight", 0.7),
        sell_out_weight=config_data.get("sell_out_weight", 0.3),
        lead_time_months=config_data.get("lead_time_months", 3.5),
        safety_stock_months=config_data.get("safety_stock_months", 2.0),
        production_coefficient=config_data.get("production_coefficient", 7.0),
        order_lead_days=config_data.get("order_lead_days", 60),
    )
    sales_input = payload.get("sales", {})
    inventory_input = payload.get("inventory", {})
    calculator = DemandPlanningCalculator(config)
    result = calculator.calculate_full_plan(
        SalesData(
            sell_in_ma12=sales_input.get("sell_in_ma12", 0),
            sell_in_ma6=sales_input.get("sell_in_ma6", 0),
            sell_in_ma3=sales_input.get("sell_in_ma3", 0),
            sell_out_ma12=sales_input.get("sell_out_ma12", 0),
            sell_out_ma6=sales_input.get("sell_out_ma6", 0),
            sell_out_ma3=sales_input.get("sell_out_ma3", 0),
        ),
        InventoryStatus(
            warehouse_stock=inventory_input.get("warehouse_stock", 0),
            in_production=inventory_input.get("in_production", 0),
            combo_reserved=inventory_input.get("combo_reserved", 0),
            promo_reserved=inventory_input.get("promo_reserved", 0),
            marketing_reserved=inventory_input.get("marketing_reserved", 0),
        ),
        payload.get("confirmed_quantity"),
    )
    return jsonify(_serialize_demand_result(result))


def _calculate_ma(series: pd.Series, period: int) -> float:
    period = max(1, int(period))
    if len(series) >= period:
        return float(series.iloc[-period:].mean())
    return float(series.mean()) if len(series) else 0.0


def _serialize_demand_result(result) -> dict[str, Any]:
    return {
        "success": True,
        "sales_analysis": {
            "sell_in_avg": result.sell_in_avg,
            "sell_out_avg": result.sell_out_avg,
            "estimated_monthly_sales": result.estimated_monthly_sales,
        },
        "inventory_demand": {
            "rop": result.rop,
            "available_stock": result.available_stock,
            "total_demand": result.total_demand,
        },
        "production_decision": {
            "need_production": bool(result.need_production),  # numpy.bool_ 無法直接轉 JSON
            "suggested_quantity": result.suggested_quantity,
            "confirmed_quantity": result.confirmed_quantity,
        },
        "inventory_status": {
            "current_available_months": result.current_available_months,
            "stock_depletion_date": result.stock_depletion_date.strftime("%Y-%m-%d"),
            "planned_available_months": result.planned_available_months,
            "planned_depletion_date": result.planned_depletion_date.strftime("%Y-%m-%d"),
        },
        "order_info": {
            "order_deadline": result.order_deadline.strftime("%Y-%m-%d") if result.order_deadline else None,
            "post_replenish_days": result.post_replenish_days,
        },
    }


# 需求計劃可選用的預測方法（key → 顯示名稱、方法、參數）
FORECAST_CHOICES = {
    "WeightedMA": (FORMULA_LABEL, "WeightedMA", {}),
    "Seasonal_Naive": ("Seasonal Naive（去年同期）", "Seasonal_Naive", {"seasonal_period": 12}),
    "HW-add": ("Holt-Winters（加法）", "HoltWinters", {"seasonal_type": "add"}),
    "HW-mul": ("Holt-Winters（乘法）", "HoltWinters", {"seasonal_type": "mul"}),
    "SES": ("SES（α=0.2）", "SES", {"alpha": 0.2}),
    "Holt": ("Holt", "Holt", {"alpha": 0.2, "beta": 0.1}),
    "SMA-3": ("SMA-3", "SMA", {"window": 3}),
    "SMA-12": ("SMA-12", "SMA", {"window": 12}),
    "Naive": ("Naive（上月）", "Naive", {}),
}
PLAN_TRAINING_MONTHS = 24  # 與回測預設相同
PLAN_FORECAST_MONTHS = 3   # 對應製程週期


def _backtest_forecast_choices(ts: pd.Series, formula_params: dict[str, Any] | None = None,
                               sell_out: pd.Series | None = None) -> list[dict[str, Any]]:
    """以預設回測設定（訓練 24／預測 3／12 輪）回測各方法，依 wMAPE 由低到高排序，保留完整結果。
    需求計劃公式使用頁面上設定的 MA 期數與權重。"""
    results = []
    for key, (label, method, params) in FORECAST_CHOICES.items():
        if key == "WeightedMA":
            params = {**DEFAULT_FORMULA_PARAMS, **(formula_params or {})}
            label = method_label(method, params)
        result = run_backtest(ts, create_forecaster(method, params), sell_out=sell_out)
        if "overall" in result:
            results.append({**result, "key": key, "label": label, "method": method, "params": params})
    return sorted(results, key=lambda item: item["overall"]["wmape"])


def _rank_forecast_choices(ts: pd.Series, formula_params: dict[str, Any] | None = None,
                          sell_out: pd.Series | None = None) -> list[dict[str, Any]]:
    return [{"key": r["key"], "label": r["label"], "wmape": r["overall"]["wmape"], "bias": r["overall"]["bias"]}
            for r in _backtest_forecast_choices(ts, formula_params, sell_out)]


def _plan_target(payload: dict[str, Any]):
    """需求計劃的計算對象：通路（Sell-in 金額）或料號（sku 資料集，數量）。回傳 (dataset, df, channel, series) 或錯誤訊息。"""
    dataset = "sku" if payload.get("dataset") == "sku" else "sellin"
    df = load_data(dataset)
    if df is None:
        return "找不到資料檔"
    if dataset == "sku":
        series = payload.get("series")
        if not series or series == "ALL":
            return "請選擇料號"
        return dataset, df, "ALL", series
    return dataset, df, payload.get("channel") or "ALL", payload.get("series", "ALL")


def _formula_params_from(payload: dict[str, Any]) -> dict[str, Any]:
    """需求計劃頁送來的 MA 期數、權重 → WeightedMA 參數。"""
    ma = payload.get("ma_periods", {})
    config = payload.get("config", {})
    weight = config.get("sell_in_weight", DEFAULT_FORMULA_PARAMS["sell_in_weight"])
    return {
        "ma_period_1": ma.get("period_1", 12), "ma_period_2": ma.get("period_2", 6), "ma_period_3": ma.get("period_3", 3),
        "sell_in_weight": weight, "sell_out_weight": config.get("sell_out_weight", 1 - weight),
        "use_sell_in_as_sell_out": payload.get("use_sell_in_as_sell_out", True),
    }


def _forecast_monthly_average(ts: pd.Series, key: str) -> dict[str, Any]:
    """用最近 PLAN_TRAINING_MONTHS 個月訓練，預測未來 PLAN_FORECAST_MONTHS 個月並取平均。"""
    _, method, params = FORECAST_CHOICES[key]
    forecaster = create_forecaster(method, params).fit(ts.iloc[-PLAN_TRAINING_MONTHS:])
    predictions = np.maximum(np.asarray(forecaster.predict(PLAN_FORECAST_MONTHS), dtype=float), 0.0)
    months = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=PLAN_FORECAST_MONTHS, freq="MS")
    return {
        "months": [{"month": m.strftime("%Y-%m"), "value": float(v)} for m, v in zip(months, predictions)],
        "monthly_average": float(predictions.mean()),
    }


@app.route("/api/demand-planning/calculate-from-data", methods=["POST"])
def calculate_demand_planning_from_data():
    payload = request.get_json(silent=True) or {}
    target = _plan_target(payload)
    if isinstance(target, str):
        return jsonify({"success": False, "error": target})
    dataset, df, channel, series = target

    ts = get_channel_time_series(df, channel, series)
    if len(ts) == 0:
        return jsonify({"success": False, "error": "沒有資料"})
    sell_out = get_sell_out_series(df, channel, series, ts.index)

    ma_periods = payload.get("ma_periods", {})
    period_1 = ma_periods.get("period_1", 12)
    period_2 = ma_periods.get("period_2", 6)
    period_3 = ma_periods.get("period_3", 3)

    sell_in_ma1 = _calculate_ma(ts, period_1)
    sell_in_ma2 = _calculate_ma(ts, period_2)
    sell_in_ma3 = _calculate_ma(ts, period_3)

    # Sell-out：有資料（sku）時用實際值；沒有時依設定以 Sell-in 代替，或不計入
    use_sell_in_as_sell_out = payload.get("use_sell_in_as_sell_out", True)
    has_sell_out = sell_out is not None and sell_out.sum() > 0
    if has_sell_out:
        sell_out_ma1, sell_out_ma2, sell_out_ma3 = (_calculate_ma(sell_out, p) for p in (period_1, period_2, period_3))
    elif use_sell_in_as_sell_out:
        sell_out_ma1, sell_out_ma2, sell_out_ma3 = sell_in_ma1, sell_in_ma2, sell_in_ma3
    else:
        sell_out_ma1 = sell_out_ma2 = sell_out_ma3 = 0.0

    config_data = payload.get("config", {})
    config = DemandPlanningConfig(
        sell_in_weight=config_data.get("sell_in_weight", 0.7),
        sell_out_weight=config_data.get("sell_out_weight", 0.3),
        lead_time_months=config_data.get("lead_time_months", 3.5),
        safety_stock_months=config_data.get("safety_stock_months", 2.0),
        production_coefficient=config_data.get("production_coefficient", 7.0),
        order_lead_days=config_data.get("order_lead_days", 60),
    )
    # 預估月均銷的方法：現行公式（預設）、自動（回測最準）或指定方法
    ranking = _rank_forecast_choices(ts, _formula_params_from(payload), sell_out)
    requested = payload.get("forecast_method", "WeightedMA")
    if requested == "auto":
        chosen = ranking[0]["key"] if ranking else "WeightedMA"
    elif requested in FORECAST_CHOICES:
        chosen = requested
    else:
        return jsonify({"success": False, "error": f"未知的預測方法：{requested}"})
    forecast = None if chosen == "WeightedMA" else _forecast_monthly_average(ts, chosen)

    inventory_data = payload.get("inventory", {})
    calculator = DemandPlanningCalculator(config)
    result = calculator.calculate_full_plan(
        SalesData(
            sell_in_ma12=sell_in_ma1,
            sell_in_ma6=sell_in_ma2,
            sell_in_ma3=sell_in_ma3,
            sell_out_ma12=sell_out_ma1,
            sell_out_ma6=sell_out_ma2,
            sell_out_ma3=sell_out_ma3,
        ),
        InventoryStatus(
            warehouse_stock=inventory_data.get("warehouse_stock", 0),
            in_production=inventory_data.get("in_production", 0),
            combo_reserved=inventory_data.get("combo_reserved", 0),
            promo_reserved=inventory_data.get("promo_reserved", 0),
            marketing_reserved=inventory_data.get("marketing_reserved", 0),
        ),
        payload.get("confirmed_quantity"),
        estimated_monthly_sales_override=forecast["monthly_average"] if forecast else None,
    )

    response = _serialize_demand_result(result)
    response["forecast_method"] = {
        "requested": requested,
        "key": chosen,
        "label": FORECAST_CHOICES[chosen][0],
        "forecast": forecast,
        "ranking": ranking,
        "data_end": str(ts.index[-1].date()),
    }
    response["target"] = {"dataset": dataset, "unit": DATASETS[dataset]["unit"], "has_sell_out": bool(has_sell_out)}
    response["calculated_ma"] = {
        "sell_in_ma1": sell_in_ma1,
        "sell_in_ma2": sell_in_ma2,
        "sell_in_ma3": sell_in_ma3,
        "sell_out_ma1": sell_out_ma1,
        "sell_out_ma2": sell_out_ma2,
        "sell_out_ma3": sell_out_ma3,
    }
    return jsonify(response)


@app.route("/api/demand-planning/backtest", methods=["POST"])
def backtest_demand_planning():
    """回測需求計劃頁的預估月均銷公式（使用頁面上的 MA 與權重），並與其他方法比較。"""
    payload = request.get_json(silent=True) or {}
    target = _plan_target(payload)
    if isinstance(target, str):
        return jsonify({"error": target}), 400
    dataset, df, channel, series = target
    ts = get_channel_time_series(df, channel, series)
    if len(ts) == 0:
        return jsonify({"error": "沒有資料"}), 400
    sell_out = get_sell_out_series(df, channel, series, ts.index)
    results = _backtest_forecast_choices(ts, _formula_params_from(payload), sell_out)
    if not results:
        min_required = PLAN_TRAINING_MONTHS + PLAN_FORECAST_MONTHS + 12 - 1
        name = series if dataset == "sku" else channel
        return jsonify({"error": f"「{name}」資料只有 {len(ts)} 個月，回測至少需要 {min_required} 個月"}), 400
    return jsonify({"dataset": dataset, "channel": channel, "series": series, "results": results,
                    "data_end": ts.index[-1].strftime("%Y-%m")})


def load_sku_stock() -> pd.DataFrame | None:
    """存貨可用月數表最新一期的庫存參考（料號 → 再製中、各倉、預估月均銷X、ROP…）。"""
    if "sku_stock" in DATA_CACHE:
        return DATA_CACHE["sku_stock"]
    try:
        if SKU_STOCK_FILE is not None:
            stock = pd.read_csv(SKU_STOCK_FILE, dtype={"料號": str, "庫存日期": str}) if SKU_STOCK_FILE.exists() else None
        elif DATASETS["sku"]["file"].exists():
            stock = pd.read_excel(DATASETS["sku"]["file"], sheet_name="庫存參考", dtype={"料號": str, "庫存日期": str})
        else:
            stock = None
    except Exception as exc:
        print(f"load_sku_stock failed: {exc}")
        stock = None
    DATA_CACHE["sku_stock"] = stock
    return stock


AGING_COLS = ["2年以上", "1.5年~2年", "1年~1.5年", "0.5年~1年", "0.5年以下"]


@app.route("/api/demand-planning/sku-list")
def get_sku_list():
    """料號選單：依近 12 個月 Sell-in 數量排序，附庫存參考。"""
    df = load_data("sku")
    if df is None:
        return jsonify({"error": "找不到料號數量資料（執行 python parse_sku_quantity.py）"}), 404
    stock = load_sku_stock()
    stock_map = {} if stock is None else {
        row["料號"]: row for row in stock.astype(object).where(stock.notna(), None).to_dict("records")}  # NaN 不是合法 JSON
    items = []
    for item in _build_series_list(df, "sku")[1:]:
        ref = stock_map.get(item["key"])
        if ref is not None:
            num = lambda v: float(v) if isinstance(v, (int, float)) else 0.0
            ref = {
                "date": str(ref.get("庫存日期") or "").zfill(4),
                "warehouse_stock": sum(num(ref.get(c)) for c in AGING_COLS),  # 總倉庫存＝各效期區間加總
                "in_production": num(ref.get("再製中")),
                "momo": num(ref.get("MOMO")), "boxful": num(ref.get("BOXFUL")),
                "sheet_forecast": num(ref.get("預估月均銷X")), "sheet_rop": num(ref.get("ROP")),
                "sheet_decision": ref.get("是否再製"), "sheet_months": num(ref.get("存貨可用月數")),
                "lead_time_note": ref.get("交期"),
            }
        items.append({**item, "stock": ref})
    return jsonify({"items": items, "data_end": df[COLS.date].max().strftime("%Y-%m")})


@app.route("/api/records/shared")
def get_shared_records():
    if not RECORDS_FILE.exists():
        return jsonify({"error": "尚未產生共用回測紀錄（執行 python build_backtest_records.py）"}), 404
    return app.response_class(RECORDS_FILE.read_text(encoding="utf-8"), mimetype="application/json")


@app.route("/api/demand-planning/config")
def get_demand_planning_config():
    return jsonify(
        {
            "forecast_choices": [{"key": key, "label": label} for key, (label, _, _) in FORECAST_CHOICES.items()],
            "config": {
                "sell_in_weight": {"value": 0.7, "min": 0, "max": 1, "step": 0.05, "label": "Sell-in 權重"},
                "sell_out_weight": {"value": 0.3, "min": 0, "max": 1, "step": 0.05, "label": "Sell-out 權重"},
                "lead_time_months": {"value": 3.5, "min": 1, "max": 12, "step": 0.5, "label": "Lead Time 月"},
                "safety_stock_months": {"value": 2.0, "min": 0, "max": 6, "step": 0.5, "label": "Safety Stock 月"},
                "production_coefficient": {
                    "value": 7.0,
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "label": "生產係數",
                },
                "order_lead_days": {"value": 60, "min": 30, "max": 120, "step": 10, "label": "下單提前天數"},
            }
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("MRP Web app")
    print("http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)
