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

DATA_CACHE: dict[str, Any] = {"df": None, "loaded": False, "source": None}

# DATA_FILE 可指向 .xlsx 或彙總後的 .csv（網頁版由 site/publish.py 產生）
PRIMARY_DATA_FILE = Path(os.environ.get("DATA_FILE", BASE_DIR / "Sellin_數據_扁平化.xlsx"))
PRIMARY_SHEET = "銷售明細"

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


FORMULA_TEXT = {
    "Naive": ("預測值 = 上一期銷量", "直接以最近一期實際值作為預測", "穩定、無明顯趨勢的品項，或作為比較基準"),
    "Seasonal_Naive": ("預測值 = 去年同期銷量", "以一個季節週期前的實際值作為預測", "季節性明顯、每年型態相近的品項"),
    "SMA": ("預測值 = (過去 N 期銷量加總) / N", "對所有歷史數據權重相同，平滑短期波動", "需求平穩、波動不大的品項"),
    "SES": ("預測值 = α × 本期實際 + (1 − α) × 本期預測", "近期資料權重較高，權重隨時間指數遞減", "無明顯趨勢與季節性的品項"),
    "Holt": ("預測值 = 水準 + 趨勢 × 期數", "在指數平滑之外加入趨勢項", "有上升或下降趨勢的品項"),
    "HoltWinters": ("預測值 = (水準 + 趨勢 × 期數) ± 季節因子", "同時考慮水準、趨勢與季節性", "有趨勢且有季節性的品項"),
    "WeightedMA": ("預測值 = Sell-in 權重 × MA 組合 + Sell-out 權重 × MA 組合", "結合 12／6／3 期移動平均，並依 Sell-in／Sell-out 加權", "現行需求計劃公式"),
}
for _key, (_formula, _desc, _suitable) in FORMULA_TEXT.items():
    FORMULA_INFO[_key].update({"formula": _formula, "description": _desc, "suitable_for": _suitable})

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


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "年月": COLS.date,
        "通路": COLS.channel,
        "系列": COLS.series,
        "銷貨淨額": COLS.sales,
    }
    missing = [source for source in rename_map if source not in df.columns]
    if missing:
        raise ValueError(f"資料欄位不足: {missing}")

    standardized = df.rename(columns=rename_map)[list(rename_map.values())].copy()
    standardized[COLS.date] = _normalize_month(standardized[COLS.date])
    standardized[COLS.channel] = standardized[COLS.channel].astype(str).str.strip()
    standardized[COLS.series] = standardized[COLS.series].fillna("未分類").astype(str).str.strip()
    standardized[COLS.sales] = pd.to_numeric(standardized[COLS.sales], errors="coerce").fillna(0.0)
    standardized = standardized.dropna(subset=[COLS.date])
    return standardized


def load_data() -> pd.DataFrame | None:
    if DATA_CACHE["loaded"]:
        return DATA_CACHE["df"]

    if not PRIMARY_DATA_FILE.exists():
        return None

    try:
        if PRIMARY_DATA_FILE.suffix.lower() == ".csv":
            raw = pd.read_csv(PRIMARY_DATA_FILE, dtype={"年月": str})
        else:
            raw = pd.read_excel(PRIMARY_DATA_FILE, sheet_name=PRIMARY_SHEET)
        df = _standardize_dataframe(raw)
        DATA_CACHE["df"] = df
        DATA_CACHE["loaded"] = True
        DATA_CACHE["source"] = PRIMARY_DATA_FILE.name
        return df
    except Exception as exc:
        print(f"load_data failed: {exc}")
        return None


def get_channel_time_series(df: pd.DataFrame, channel: str, series: str | None = None) -> pd.Series:
    filtered = df.copy()
    if channel and channel != "ALL":
        filtered = filtered[filtered[COLS.channel] == channel]
    if series and series != "ALL":
        filtered = filtered[filtered[COLS.series] == series]

    ts = filtered.groupby(COLS.date)[COLS.sales].sum().sort_index()
    if len(ts):
        full_range = pd.date_range(ts.index.min(), ts.index.max(), freq="MS")
        ts = ts.reindex(full_range, fill_value=0.0)
    return ts.astype(float)


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
) -> dict[str, Any]:
    min_required = training_months + forecast_months + rolling_rounds - 1
    if len(ts) < min_required:
        return {"error": f"資料不足，至少需要 {min_required} 個月"}

    all_actuals: list[float] = []
    all_forecasts: list[float] = []
    round_results: list[dict[str, Any]] = []

    for round_idx in range(rolling_rounds):
        train_end = training_months + round_idx
        test_start = train_end
        test_end = test_start + forecast_months
        if test_end > len(ts):
            break

        train_data = ts.iloc[:train_end]
        test_data = ts.iloc[test_start:test_end]
        try:
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
        "round_results": round_results,
        "overall_wmape": float(np.sum(np.abs(pred_arr - actual_arr)) / max(np.sum(actual_arr), 1.0)),
        "overall_mape": float(np.mean(errors)),
        "overall_bias": float((np.sum(pred_arr) - np.sum(actual_arr)) / max(np.sum(actual_arr), 1.0)),
        "overall_rmse": float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2))),
        "accuracy_10": float(np.mean(errors <= 0.10)),
        "accuracy_20": float(np.mean(errors <= 0.20)),
        "n_predictions": int(len(pred_arr)),
    }


def _build_series_list(df: pd.DataFrame) -> list[dict[str, str]]:
    series_sales = df.groupby(COLS.series)[COLS.sales].sum().sort_values(ascending=False)
    top_series = series_sales.head(15).index.tolist()
    return [{"key": "ALL", "name": "全部系列"}] + [{"key": item, "name": item} for item in top_series]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demand-planning")
def demand_planning_page():
    return render_template("demand_planning.html")


@app.route("/api/data/info")
def get_data_info():
    df = load_data()
    if df is None:
        return jsonify({"error": "找不到資料檔"}), 500

    channels = sorted(df[COLS.channel].dropna().unique().tolist())
    return jsonify(
        {
            "channels": ["ALL"] + channels,
            "channel_names": {"ALL": "全部通路", **{channel: channel for channel in channels}},
            "series": _build_series_list(df),
            "year_range": [int(df[COLS.date].dt.year.min()), int(df[COLS.date].dt.year.max())],
            "total_records": int(len(df)),
            "data_source": DATA_CACHE.get("source", "unknown"),
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
    df = load_data()
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
    )
    result.update(
        {
            "channel": channel,
            "series": series,
            "method": method,
            "params": params,
            "data_length": int(len(ts)),
            "data_range": f"{ts.index.min().date()} ~ {ts.index.max().date()}",
        }
    )
    return jsonify(result)


@app.route("/api/backtest/compare", methods=["POST"])
def compare_methods_api():
    payload = request.get_json(silent=True) or {}
    df = load_data()
    if df is None:
        return jsonify({"error": "找不到資料檔"}), 500

    channel = payload.get("channel", "ALL")
    series = payload.get("series", "ALL")
    methods_config = payload.get("methods", [])
    backtest_config = payload.get("backtest_config", {})

    ts = get_channel_time_series(df, channel, series)
    if len(ts) == 0:
        return jsonify({"error": f"沒有資料: {channel} / {series}"}), 400

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
        )
        result.update({"method": method, "params": params})
        results.append(result)

    ranking = sorted(
        [item for item in results if "overall_wmape" in item],
        key=lambda item: item["overall_wmape"],
    )
    return jsonify({"results": ranking, "ranking": ranking})


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


@app.route("/api/demand-planning/calculate-from-data", methods=["POST"])
def calculate_demand_planning_from_data():
    payload = request.get_json(silent=True) or {}
    df = load_data()
    if df is None:
        return jsonify({"success": False, "error": "找不到資料檔"})

    channel = payload.get("channel")
    if not channel:
        return jsonify({"success": False, "error": "請提供通路"})

    ts = get_channel_time_series(df, channel, payload.get("series", "ALL"))
    if len(ts) == 0:
        return jsonify({"success": False, "error": "該通路沒有資料"})

    ma_periods = payload.get("ma_periods", {})
    period_1 = ma_periods.get("period_1", 12)
    period_2 = ma_periods.get("period_2", 6)
    period_3 = ma_periods.get("period_3", 3)

    sell_in_ma1 = _calculate_ma(ts, period_1)
    sell_in_ma2 = _calculate_ma(ts, period_2)
    sell_in_ma3 = _calculate_ma(ts, period_3)

    use_sell_in_as_sell_out = payload.get("use_sell_in_as_sell_out", True)
    if use_sell_in_as_sell_out:
        sell_out_ma1, sell_out_ma2, sell_out_ma3 = sell_in_ma1, sell_in_ma2, sell_in_ma3
    else:
        sell_out_ma1, sell_out_ma2, sell_out_ma3 = sell_in_ma1, sell_in_ma2, sell_in_ma3

    config_data = payload.get("config", {})
    config = DemandPlanningConfig(
        sell_in_weight=config_data.get("sell_in_weight", 0.7),
        sell_out_weight=config_data.get("sell_out_weight", 0.3),
        lead_time_months=config_data.get("lead_time_months", 3.5),
        safety_stock_months=config_data.get("safety_stock_months", 2.0),
        production_coefficient=config_data.get("production_coefficient", 7.0),
        order_lead_days=config_data.get("order_lead_days", 60),
    )
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
    )

    response = _serialize_demand_result(result)
    response["calculated_ma"] = {
        "sell_in_ma1": sell_in_ma1,
        "sell_in_ma2": sell_in_ma2,
        "sell_in_ma3": sell_in_ma3,
        "sell_out_ma1": sell_out_ma1,
        "sell_out_ma2": sell_out_ma2,
        "sell_out_ma3": sell_out_ma3,
    }
    return jsonify(response)


@app.route("/api/demand-planning/config")
def get_demand_planning_config():
    return jsonify(
        {
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
