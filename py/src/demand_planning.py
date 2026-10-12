"""
需求計劃系統 - 生產公式實作

根據需求計劃系統_公式說明文檔.docx 實作以下公式：
1. 預估月均銷量 (AB) = Sell-in MA 平均 × 0.7 + Sell-out MA 平均 × 0.3
2. 再訂購點 ROP (G) = 預估月均銷 × (前置時間 + 安全庫存月數)
3. 是否再製判斷 (J) = IF(可用庫存 < 總需求, "Y", "N")
4. 再製量 (L) = 預估月均銷 × 生產係數
5. 需求交期 (M) = 計劃銷完日 - 60天
6. 補倉後可用日 (O) = (決議再製量 ÷ 預估月均銷) × 30 + 計劃銷完日
7. 庫存狀態計算 (AF-AI)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta


@dataclass
class DemandPlanningConfig:
    """需求計劃配置參數"""
    # 銷售權重
    sell_in_weight: float = 0.7      # Sell-in 權重
    sell_out_weight: float = 0.3     # Sell-out 權重
    
    # 前置時間與安全庫存
    lead_time_months: float = 3.5    # 前置時間（月）
    safety_stock_months: float = 2.0  # 安全庫存月數
    
    # 生產係數
    production_coefficient: float = 7.0  # 生產係數（批量倍數）
    
    # 需求交期提前天數
    order_lead_days: int = 60        # 提前下單天數
    
    # 四捨五入精度
    round_precision: int = -2        # 四捨五入至百位數


@dataclass
class InventoryStatus:
    """庫存狀態"""
    warehouse_stock: float = 0       # 總倉庫存 (R)
    in_production: float = 0         # 再製中 (P)
    combo_reserved: float = 0        # 組包預留 (AC)
    promo_reserved: float = 0        # 深促預留 (AD)
    marketing_reserved: float = 0    # 行銷預留 (AE)


@dataclass
class SalesData:
    """銷售數據"""
    sell_in_ma12: float = 0          # Sell-in MA-12 (V)
    sell_in_ma6: float = 0           # Sell-in MA-6 (W)
    sell_in_ma3: float = 0           # Sell-in MA-3 (X)
    sell_out_ma12: float = 0         # Sell-out MA-12 (Y)
    sell_out_ma6: float = 0          # Sell-out MA-6 (Z)
    sell_out_ma3: float = 0          # Sell-out MA-3 (AA)


@dataclass
class DemandPlanningResult:
    """需求計劃計算結果"""
    # 基礎計算
    sell_in_avg: float               # Sell-in 平均
    sell_out_avg: float              # Sell-out 平均
    estimated_monthly_sales: float   # 預估月均銷 (AB)
    
    # ROP 計算
    rop: float                       # 再訂購點 (G)
    
    # 庫存狀態
    available_stock: float           # 可用庫存 (P + R)
    total_demand: float              # 總需求 (G + AE + AC + AD)
    
    # 再製判斷
    need_production: bool            # 是否再製 (J)
    suggested_quantity: float        # 建議再製量 (L)
    
    # 庫存天數計算
    current_available_months: float  # 現有可用月數 (AF)
    stock_depletion_date: datetime   # 庫存銷完日 (AG)
    planned_available_months: float  # 計劃可用月數 (AH)
    planned_depletion_date: datetime # 計劃銷完日 (AI)
    
    # 需求交期與補倉
    order_deadline: Optional[datetime] = None  # 需求交期 (M)
    post_replenish_days: Optional[float] = None  # 補倉後可用日 (O)
    
    # 決議再製量（人工輸入）
    confirmed_quantity: Optional[float] = None  # 決議再製量 (N)


class DemandPlanningCalculator:
    """
    需求計劃計算器
    
    實作需求計劃系統的所有公式計算
    """
    
    def __init__(self, config: Optional[DemandPlanningConfig] = None):
        """
        初始化計算器
        
        Args:
            config: 配置參數，若為 None 則使用預設值
        """
        self.config = config or DemandPlanningConfig()
    
    def calculate_estimated_monthly_sales(self, sales: SalesData) -> Tuple[float, float, float]:
        """
        計算預估月均銷量 (AB欄)
        
        公式: =ROUND(AVERAGE(V:X)*0.7 + AVERAGE(Y:AA)*0.3, -2)
        
        Args:
            sales: 銷售數據
            
        Returns:
            (sell_in_avg, sell_out_avg, estimated_monthly_sales)
        """
        # Sell-in 移動平均的平均值
        sell_in_avg = np.mean([sales.sell_in_ma12, sales.sell_in_ma6, sales.sell_in_ma3])
        
        # Sell-out 移動平均的平均值
        sell_out_avg = np.mean([sales.sell_out_ma12, sales.sell_out_ma6, sales.sell_out_ma3])
        
        # 加權計算
        weighted_avg = (sell_in_avg * self.config.sell_in_weight + 
                       sell_out_avg * self.config.sell_out_weight)
        
        # 四捨五入至百位數
        estimated = round(weighted_avg, self.config.round_precision)
        
        return sell_in_avg, sell_out_avg, estimated
    
    def calculate_rop(self, estimated_monthly_sales: float) -> float:
        """
        計算再訂購點 ROP (G欄)
        
        公式: =AB*(H+I)
        ROP = 預估月均銷 × (前置時間 + 安全庫存月數)
        
        Args:
            estimated_monthly_sales: 預估月均銷
            
        Returns:
            再訂購點
        """
        rop = estimated_monthly_sales * (
            self.config.lead_time_months + self.config.safety_stock_months
        )
        return rop
    
    def calculate_available_stock(self, inventory: InventoryStatus) -> float:
        """
        計算可用庫存
        
        公式: P + R (再製中 + 總倉庫存)
        
        Args:
            inventory: 庫存狀態
            
        Returns:
            可用庫存
        """
        return inventory.in_production + inventory.warehouse_stock
    
    def calculate_total_demand(self, rop: float, inventory: InventoryStatus) -> float:
        """
        計算總需求
        
        公式: G + AE + AC + AD (ROP + 行銷 + 組包 + 深促)
        
        Args:
            rop: 再訂購點
            inventory: 庫存狀態
            
        Returns:
            總需求
        """
        return (rop + inventory.marketing_reserved + 
                inventory.combo_reserved + inventory.promo_reserved)
    
    def check_need_production(self, available_stock: float, total_demand: float) -> bool:
        """
        判斷是否需要再製 (J欄)
        
        公式: =IF(P+R < G+AE+AC+AD, "Y", "N")
        
        Args:
            available_stock: 可用庫存
            total_demand: 總需求
            
        Returns:
            是否需要再製
        """
        return available_stock < total_demand
    
    def calculate_production_quantity(self, estimated_monthly_sales: float) -> float:
        """
        計算建議再製量 (L欄)
        
        公式: =IF(J="Y", AB*K, " ")
        再製量 = 預估月均銷 × 生產係數
        
        Args:
            estimated_monthly_sales: 預估月均銷
            
        Returns:
            建議再製量
        """
        return estimated_monthly_sales * self.config.production_coefficient
    
    def calculate_inventory_status(
        self, 
        inventory: InventoryStatus, 
        estimated_monthly_sales: float,
        base_date: Optional[datetime] = None
    ) -> Tuple[float, datetime, float, datetime]:
        """
        計算庫存狀態 (AF-AI欄)
        
        AF 現有可用月數 = R / AB
        AG 庫存銷完日 = AF * 30 + TODAY()
        AH 計劃可用月數 = (R + P - AE - AD - AC) / AB
        AI 計劃銷完日 = AH * 30 + TODAY()
        
        Args:
            inventory: 庫存狀態
            estimated_monthly_sales: 預估月均銷
            base_date: 基準日期，預設為今天
            
        Returns:
            (current_available_months, stock_depletion_date, 
             planned_available_months, planned_depletion_date)
        """
        if base_date is None:
            base_date = datetime.now()
        
        # 避免除以零
        if estimated_monthly_sales <= 0:
            estimated_monthly_sales = 1
        
        # AF: 現有可用月數
        current_available_months = inventory.warehouse_stock / estimated_monthly_sales
        
        # AG: 庫存銷完日
        stock_depletion_date = base_date + timedelta(days=current_available_months * 30)
        
        # AH: 計劃可用月數（扣除預留後的淨庫存）
        net_stock = (inventory.warehouse_stock + inventory.in_production - 
                    inventory.marketing_reserved - inventory.promo_reserved - 
                    inventory.combo_reserved)
        planned_available_months = net_stock / estimated_monthly_sales
        
        # AI: 計劃銷完日
        planned_depletion_date = base_date + timedelta(days=planned_available_months * 30)
        
        return (current_available_months, stock_depletion_date, 
                planned_available_months, planned_depletion_date)
    
    def calculate_order_deadline(self, planned_depletion_date: datetime) -> datetime:
        """
        計算需求交期 (M欄)
        
        公式: =IF(J="Y", AI-60, " ")
        需求交期 = 計劃銷完日 - 60天
        
        Args:
            planned_depletion_date: 計劃銷完日
            
        Returns:
            需求交期
        """
        return planned_depletion_date - timedelta(days=self.config.order_lead_days)
    
    def calculate_post_replenish_days(
        self, 
        confirmed_quantity: float, 
        estimated_monthly_sales: float,
        planned_depletion_date: datetime
    ) -> datetime:
        """
        計算補倉後可用日 (O欄)
        
        公式: =N/AB*30 + AI
        補倉後可用日 = (決議再製量 ÷ 預估月均銷) × 30天 + 計劃銷完日
        
        Args:
            confirmed_quantity: 決議再製量
            estimated_monthly_sales: 預估月均銷
            planned_depletion_date: 計劃銷完日
            
        Returns:
            補倉後可用日
        """
        if estimated_monthly_sales <= 0:
            estimated_monthly_sales = 1
        
        additional_days = (confirmed_quantity / estimated_monthly_sales) * 30
        return planned_depletion_date + timedelta(days=additional_days)
    
    def calculate_full_plan(
        self, 
        sales: SalesData, 
        inventory: InventoryStatus,
        confirmed_quantity: Optional[float] = None,
        base_date: Optional[datetime] = None
    ) -> DemandPlanningResult:
        """
        執行完整的需求計劃計算
        
        Args:
            sales: 銷售數據
            inventory: 庫存狀態
            confirmed_quantity: 決議再製量（可選）
            base_date: 基準日期
            
        Returns:
            完整的需求計劃結果
        """
        if base_date is None:
            base_date = datetime.now()
        
        # 1. 計算預估月均銷
        sell_in_avg, sell_out_avg, estimated_monthly_sales = \
            self.calculate_estimated_monthly_sales(sales)
        
        # 2. 計算 ROP
        rop = self.calculate_rop(estimated_monthly_sales)
        
        # 3. 計算可用庫存與總需求
        available_stock = self.calculate_available_stock(inventory)
        total_demand = self.calculate_total_demand(rop, inventory)
        
        # 4. 判斷是否再製
        need_production = self.check_need_production(available_stock, total_demand)
        
        # 5. 計算建議再製量
        suggested_quantity = self.calculate_production_quantity(estimated_monthly_sales) \
            if need_production else 0
        
        # 6. 計算庫存狀態
        (current_available_months, stock_depletion_date, 
         planned_available_months, planned_depletion_date) = \
            self.calculate_inventory_status(inventory, estimated_monthly_sales, base_date)
        
        # 7. 計算需求交期（若需再製）
        order_deadline = self.calculate_order_deadline(planned_depletion_date) \
            if need_production else None
        
        # 8. 計算補倉後可用日（若有決議再製量）
        post_replenish_days = None
        if confirmed_quantity and confirmed_quantity > 0:
            post_replenish_date = self.calculate_post_replenish_days(
                confirmed_quantity, estimated_monthly_sales, planned_depletion_date
            )
            post_replenish_days = (post_replenish_date - base_date).days
        
        return DemandPlanningResult(
            sell_in_avg=sell_in_avg,
            sell_out_avg=sell_out_avg,
            estimated_monthly_sales=estimated_monthly_sales,
            rop=rop,
            available_stock=available_stock,
            total_demand=total_demand,
            need_production=need_production,
            suggested_quantity=suggested_quantity,
            current_available_months=current_available_months,
            stock_depletion_date=stock_depletion_date,
            planned_available_months=planned_available_months,
            planned_depletion_date=planned_depletion_date,
            order_deadline=order_deadline,
            post_replenish_days=post_replenish_days,
            confirmed_quantity=confirmed_quantity
        )


class WeightedMAForecaster:
    """
    加權移動平均預測器（需求計劃系統專用）
    
    使用 Sell-in 與 Sell-out 的加權組合進行預測
    支援自訂 MA 月份數、權重比例、以及無 Sell-out 時的替代邏輯
    """
    
    def __init__(
        self, 
        sell_in_weight: float = 0.7, 
        sell_out_weight: float = 0.3,
        ma_period_1: int = 12,
        ma_period_2: int = 6,
        ma_period_3: int = 3,
        use_sell_in_as_sell_out: bool = True,
        round_precision: int = -2
    ):
        """
        初始化預測器
        
        Args:
            sell_in_weight: Sell-in 權重 (0-1)
            sell_out_weight: Sell-out 權重 (0-1)
            ma_period_1: 第一個移動平均月份數（預設 12）
            ma_period_2: 第二個移動平均月份數（預設 6）
            ma_period_3: 第三個移動平均月份數（預設 3）
            use_sell_in_as_sell_out: 無 Sell-out 數據時，是否用 Sell-in 替代
            round_precision: 四捨五入精度（預設 -2 為百位數）
        """
        self.sell_in_weight = sell_in_weight
        self.sell_out_weight = sell_out_weight
        self.ma_periods = [ma_period_1, ma_period_2, ma_period_3]
        self.use_sell_in_as_sell_out = use_sell_in_as_sell_out
        self.round_precision = round_precision
        self._fitted = False
        self._forecast_value = None
        self._sell_in_avg = None
        self._sell_out_avg = None
    
    def fit(self, sell_in_data: pd.Series, sell_out_data: pd.Series = None) -> 'WeightedMAForecaster':
        """
        訓練模型
        
        Args:
            sell_in_data: Sell-in 時間序列
            sell_out_data: Sell-out 時間序列（可選）
            
        Returns:
            self
        """
        # 計算 Sell-in 各期移動平均
        sell_in_mas = []
        for period in self.ma_periods:
            if period > 0 and len(sell_in_data) >= period:
                ma = sell_in_data.iloc[-period:].mean()
                sell_in_mas.append(ma)
        
        self._sell_in_avg = np.mean(sell_in_mas) if sell_in_mas else sell_in_data.mean()
        
        # 處理 Sell-out 數據
        has_sell_out = sell_out_data is not None and len(sell_out_data) > 0 and sell_out_data.sum() > 0
        
        if has_sell_out:
            # 有 Sell-out 數據，正常計算
            sell_out_mas = []
            for period in self.ma_periods:
                if period > 0 and len(sell_out_data) >= period:
                    ma = sell_out_data.iloc[-period:].mean()
                    sell_out_mas.append(ma)
            
            self._sell_out_avg = np.mean(sell_out_mas) if sell_out_mas else sell_out_data.mean()
            
            # 加權計算
            self._forecast_value = (self._sell_in_avg * self.sell_in_weight + 
                                   self._sell_out_avg * self.sell_out_weight)
        elif self.use_sell_in_as_sell_out:
            # 無 Sell-out 數據，但設定用 Sell-in 替代
            self._sell_out_avg = self._sell_in_avg
            self._forecast_value = self._sell_in_avg  # 100% Sell-in
        else:
            # 無 Sell-out 數據，且不替代，僅使用 Sell-in 部分
            self._sell_out_avg = 0
            self._forecast_value = self._sell_in_avg * self.sell_in_weight
        
        self._fitted = True
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """
        預測未來數值
        
        Args:
            steps: 預測步數
            
        Returns:
            預測值陣列
        """
        if not self._fitted:
            raise ValueError("模型尚未訓練，請先呼叫 fit()")
        
        # 四捨五入
        rounded_value = round(self._forecast_value, self.round_precision)
        return np.full(steps, rounded_value)
    
    def get_details(self) -> dict:
        """取得計算細節"""
        return {
            'sell_in_avg': self._sell_in_avg,
            'sell_out_avg': self._sell_out_avg,
            'forecast_value': self._forecast_value,
            'ma_periods': self.ma_periods,
            'sell_in_weight': self.sell_in_weight,
            'sell_out_weight': self.sell_out_weight,
            'use_sell_in_as_sell_out': self.use_sell_in_as_sell_out
        }


def demo_calculation():
    """示範計算（使用文檔中的範例數據）"""
    # 建立計算器
    config = DemandPlanningConfig(
        lead_time_months=3.5,
        safety_stock_months=2.0,
        production_coefficient=7.0
    )
    calculator = DemandPlanningCalculator(config)
    
    # 輸入銷售數據（文檔範例）
    sales = SalesData(
        sell_in_ma12=2983,
        sell_in_ma6=2987,
        sell_in_ma3=4521,
        sell_out_ma12=2878,
        sell_out_ma6=2659,
        sell_out_ma3=3584
    )
    
    # 輸入庫存狀態（文檔範例）
    inventory = InventoryStatus(
        warehouse_stock=13800,
        in_production=19002,
        combo_reserved=8000,
        promo_reserved=5000,
        marketing_reserved=5000
    )
    
    # 執行計算
    result = calculator.calculate_full_plan(sales, inventory)
    
    # 輸出結果
    print("=" * 60)
    print(" 需求計劃計算結果")
    print("=" * 60)
    print(f"\n【銷售數據分析】")
    print(f"  Sell-in 平均: {result.sell_in_avg:,.0f}")
    print(f"  Sell-out 平均: {result.sell_out_avg:,.0f}")
    print(f"  預估月均銷 (AB): {result.estimated_monthly_sales:,.0f}")
    
    print(f"\n【庫存與需求】")
    print(f"  再訂購點 ROP (G): {result.rop:,.0f}")
    print(f"  可用庫存 (P+R): {result.available_stock:,.0f}")
    print(f"  總需求 (G+AE+AC+AD): {result.total_demand:,.0f}")
    
    print(f"\n【再製判斷】")
    print(f"  是否再製 (J): {'Y (需要)' if result.need_production else 'N (不需要)'}")
    print(f"  建議再製量 (L): {result.suggested_quantity:,.0f}")
    
    print(f"\n【庫存狀態】")
    print(f"  現有可用月數 (AF): {result.current_available_months:.1f}")
    print(f"  庫存銷完日 (AG): {result.stock_depletion_date.strftime('%Y-%m-%d')}")
    print(f"  計劃可用月數 (AH): {result.planned_available_months:.1f}")
    print(f"  計劃銷完日 (AI): {result.planned_depletion_date.strftime('%Y-%m-%d')}")
    
    if result.order_deadline:
        print(f"\n【需求交期】")
        print(f"  需求交期 (M): {result.order_deadline.strftime('%Y-%m-%d')}")
    
    return result


if __name__ == '__main__':
    demo_calculation()
