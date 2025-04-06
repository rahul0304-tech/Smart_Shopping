from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from prophet import Prophet
from dataclasses import dataclass

@dataclass
class InventoryStatus:
    """Inventory status information"""
    product_id: str
    current_stock: int
    reorder_point: int
    lead_time_days: float
    safety_stock: int
    stock_status: str  # 'in_stock', 'low_stock', 'out_of_stock'
    predicted_stockout_date: Optional[datetime] = None
    recommended_action: Optional[str] = None

@dataclass
class DemandForecast:
    """Demand forecast information"""
    product_id: str
    forecast_periods: List[datetime]
    predicted_demand: List[float]
    confidence_intervals: List[Dict[str, float]]
    seasonal_patterns: Dict[str, float]

class InventoryForecaster:
    """Time series forecasting for inventory management"""
    def __init__(self):
        self.models: Dict[str, Prophet] = {}
        self.seasonal_patterns = {
            'daily': True,
            'weekly': True,
            'yearly': True
        }

    def train_model(self, product_id: str, historical_data: List[Dict[str, Any]]):
        """Train Prophet model for a product"""
        # Prepare data for Prophet
        df = pd.DataFrame([
            {
                'ds': record['timestamp'],
                'y': record['demand']
            }
            for record in historical_data
        ])

        # Initialize and train Prophet model
        model = Prophet(
            daily_seasonality=self.seasonal_patterns['daily'],
            weekly_seasonality=self.seasonal_patterns['weekly'],
            yearly_seasonality=self.seasonal_patterns['yearly']
        )
        model.fit(df)
        self.models[product_id] = model

    def get_forecast(self, product_id: str, days: int = 30) -> Optional[DemandForecast]:
        """Generate demand forecast for a product"""
        if product_id not in self.models:
            return None

        model = self.models[product_id]
        future_dates = model.make_future_dataframe(periods=days)
        forecast = model.predict(future_dates)

        # Extract seasonal patterns
        seasonal_patterns = {}
        if 'yearly' in forecast.columns:
            seasonal_patterns['yearly'] = forecast['yearly'].mean()
        if 'weekly' in forecast.columns:
            seasonal_patterns['weekly'] = forecast['weekly'].mean()
        if 'daily' in forecast.columns:
            seasonal_patterns['daily'] = forecast['daily'].mean()

        return DemandForecast(
            product_id=product_id,
            forecast_periods=forecast['ds'].tolist()[-days:],
            predicted_demand=forecast['yhat'].tolist()[-days:],
            confidence_intervals=[{
                'lower': lower,
                'upper': upper
            } for lower, upper in zip(
                forecast['yhat_lower'].tolist()[-days:],
                forecast['yhat_upper'].tolist()[-days:]
            )],
            seasonal_patterns=seasonal_patterns
        )

class InventoryManager:
    """Inventory management and optimization"""
    def __init__(self):
        self.forecaster = InventoryForecaster()
        self.inventory_cache: Dict[str, InventoryStatus] = {}

    def update_inventory_status(self, product: Dict[str, Any]):
        """Update inventory status for a product"""
        inventory = product['inventory']
        current_stock = inventory['stock_level']
        reorder_point = inventory['reorder_point']

        # Calculate safety stock based on lead time and demand variability
        lead_time = inventory['lead_time_days']
        safety_stock = int(reorder_point * 0.2)  # 20% of reorder point as safety stock

        # Determine stock status
        if current_stock <= 0:
            stock_status = 'out_of_stock'
        elif current_stock <= reorder_point:
            stock_status = 'low_stock'
        else:
            stock_status = 'in_stock'

        # Get demand forecast
        forecast = self.forecaster.get_forecast(product['product_id'])
        predicted_stockout_date = None
        if forecast:
            cumulative_demand = np.cumsum(forecast.predicted_demand)
            stockout_days = np.where(cumulative_demand > current_stock)[0]
            if len(stockout_days) > 0:
                predicted_stockout_date = forecast.forecast_periods[stockout_days[0]]

        # Determine recommended action
        recommended_action = self._get_recommended_action(
            current_stock, reorder_point, safety_stock, stock_status, predicted_stockout_date
        )

        # Update cache
        self.inventory_cache[product['product_id']] = InventoryStatus(
            product_id=product['product_id'],
            current_stock=current_stock,
            reorder_point=reorder_point,
            lead_time_days=lead_time,
            safety_stock=safety_stock,
            stock_status=stock_status,
            predicted_stockout_date=predicted_stockout_date,
            recommended_action=recommended_action
        )

    def _get_recommended_action(self, current_stock: int, reorder_point: int,
                              safety_stock: int, stock_status: str,
                              predicted_stockout_date: Optional[datetime]) -> str:
        """Determine recommended action based on inventory status"""
        if stock_status == 'out_of_stock':
            return 'immediate_reorder'
        elif stock_status == 'low_stock':
            if current_stock <= safety_stock:
                return 'urgent_reorder'
            return 'plan_reorder'
        elif predicted_stockout_date and (predicted_stockout_date - datetime.now()).days < 7:
            return 'schedule_reorder'
        return 'monitor'

    def check_recommendation_availability(self, recommendations: List[Dict[str, Any]]) \
            -> List[Dict[str, Any]]:
        """Filter and adjust recommendations based on inventory status"""
        filtered_recommendations = []

        for rec in recommendations:
            product_id = rec['product_id']
            if product_id in self.inventory_cache:
                status = self.inventory_cache[product_id]
                
                # Skip out of stock items
                if status.stock_status == 'out_of_stock':
                    continue

                # Adjust score based on inventory status
                inventory_factor = 1.0
                if status.stock_status == 'low_stock':
                    inventory_factor = 0.8
                elif status.predicted_stockout_date and \
                     (status.predicted_stockout_date - datetime.now()).days < 7:
                    inventory_factor = 0.9

                filtered_rec = rec.copy()
                filtered_rec['score'] = rec.get('score', 1.0) * inventory_factor
                filtered_rec['inventory_status'] = status.stock_status
                filtered_recommendations.append(filtered_rec)

        return sorted(filtered_recommendations, key=lambda x: x['score'], reverse=True)

    def get_inventory_alerts(self) -> List[Dict[str, Any]]:
        """Get current inventory alerts"""
        alerts = []
        for product_id, status in self.inventory_cache.items():
            if status.stock_status != 'in_stock' or \
               (status.predicted_stockout_date and \
                (status.predicted_stockout_date - datetime.now()).days < 7):
                
                alert = {
                    'product_id': product_id,
                    'status': status.stock_status,
                    'current_stock': status.current_stock,
                    'reorder_point': status.reorder_point,
                    'recommended_action': status.recommended_action,
                    'priority': 'high' if status.stock_status == 'out_of_stock' else \
                               'medium' if status.stock_status == 'low_stock' else 'low'
                }
                
                if status.predicted_stockout_date:
                    alert['predicted_stockout'] = status.predicted_stockout_date
                
                alerts.append(alert)

        return sorted(alerts, key=lambda x: ['high', 'medium', 'low'].index(x['priority']))