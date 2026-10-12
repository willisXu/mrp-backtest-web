"""
MRP 回測系統配置參數
"""
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class BacktestConfig:
    """回測參數配置"""
    # 訓練期與預測期
    training_months: int = 24  # 訓練期月數（確保足夠學習季節性，至少兩個完整年度）
    forecast_months: int = 3  # 預測期月數（對應製程週期）
    rolling_rounds: int = 12  # 滾動輪數
    min_data_months: int = 36  # 最少數據需求 (24 + 12)
    
    # 季節週期
    seasonal_period: int = 12  # 月度數據的季節週期


@dataclass
class SMAConfig:
    """簡單移動平均參數"""
    periods: List[int] = field(default_factory=lambda: [3, 6, 12])


@dataclass
class WMAConfig:
    """加權移動平均參數"""
    periods: int = 3
    weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])


@dataclass
class SESConfig:
    """簡單指數平滑參數"""
    alphas: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


@dataclass
class HoltConfig:
    """Holt 雙參數指數平滑參數"""
    alphas: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    betas: List[float] = field(default_factory=lambda: [0.1, 0.2])


@dataclass
class HoltWintersConfig:
    """Holt-Winters 三參數指數平滑參數"""
    alphas: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    betas: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    gammas: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    seasonal_periods: int = 12
    seasonal_types: List[str] = field(default_factory=lambda: ['add', 'mul'])


@dataclass
class PromotionConfig:
    """促銷調整參數"""
    # 促銷類型與預估提升係數
    promotion_coefficients: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'B1G1': {'min': 1.5, 'max': 2.5, 'default': 2.0},
        '背櫃購': {'min': 0.3, 'max': 0.8, 'default': 0.5},
        '第2件5折': {'min': 0.8, 'max': 1.5, 'default': 1.0},
        '單件8折': {'min': 0.3, 'max': 0.6, 'default': 0.4},
        '單件75折': {'min': 0.4, 'max': 0.8, 'default': 0.6},
    })


@dataclass
class ChannelConfig:
    """通路配置"""
    channels: Dict[str, Dict] = field(default_factory=lambda: {
        'WSN': {
            'name': 'Watsons',
            'has_sellout': True,
            'promotion_types': ['B1G1', '背櫃購', '品類換購']
        },
        'CSM': {
            'name': 'Cosmed',
            'has_sellout': True,
            'promotion_types': ['B1G1', '背櫃購', '第2件5折']
        },
        'POYA': {
            'name': '寶雅',
            'has_sellout': False,
            'promotion_types': ['B1G1', '背櫃購']
        }
    })


@dataclass
class MetricsConfig:
    """評估指標配置"""
    accuracy_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.2])  # ±10%, ±20%
    mape_benchmarks: Dict[str, float] = field(default_factory=lambda: {
        'excellent': 0.10,  # < 10%
        'good': 0.20,       # 10-20%
        'acceptable': 0.30, # 20-30%
    })


@dataclass
class SystemConfig:
    """系統總配置"""
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    sma: SMAConfig = field(default_factory=SMAConfig)
    wma: WMAConfig = field(default_factory=WMAConfig)
    ses: SESConfig = field(default_factory=SESConfig)
    holt: HoltConfig = field(default_factory=HoltConfig)
    holt_winters: HoltWintersConfig = field(default_factory=HoltWintersConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


# 預設配置實例
DEFAULT_CONFIG = SystemConfig()
