from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class ProductPrice(BaseModel):
    """Product pricing information"""
    base_price: float = Field(..., gt=0)
    current_price: float = Field(..., gt=0)
    currency: str = Field(default="USD")
    discount_percentage: float = Field(default=0, ge=0, le=100)
    price_history: List[Dict[str, float]] = Field(default_factory=list)  # [{timestamp: price}]

class ProductInventory(BaseModel):
    """Inventory and supply chain information"""
    stock_level: int = Field(..., ge=0)
    reorder_point: int = Field(..., ge=0)
    supplier_id: str = Field(...)
    lead_time_days: float = Field(..., ge=0)  # average delivery time
    warehouse_location: str = Field(...)
    last_restock_date: datetime = Field(default_factory=datetime.utcnow)

class ProductMetrics(BaseModel):
    """Product performance and popularity metrics"""
    view_count: int = Field(default=0, ge=0)
    purchase_count: int = Field(default=0, ge=0)
    rating: float = Field(default=0, ge=0, le=5)
    review_count: int = Field(default=0, ge=0)
    add_to_cart_count: int = Field(default=0, ge=0)
    conversion_rate: float = Field(default=0, ge=0, le=1)

class ProductAttributes(BaseModel):
    """Product characteristics and features"""
    brand: str = Field(...)
    category: str = Field(...)
    subcategories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    features: Dict[str, str] = Field(default_factory=dict)
    dimensions: Optional[Dict[str, float]] = None  # length, width, height
    weight: Optional[float] = None

class Product(BaseModel):
    """Main product model"""
    product_id: str = Field(...)
    name: str = Field(...)
    description: str = Field(...)
    attributes: ProductAttributes = Field(...)
    price_info: ProductPrice = Field(...)
    inventory: ProductInventory = Field(...)
    metrics: ProductMetrics = Field(default_factory=ProductMetrics)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    recommendations: List[str] = Field(default_factory=list)  # related product IDs
    seasonal_flags: List[str] = Field(default_factory=list)  # e.g., ["summer", "holiday"]

    class Config:
        schema_extra = {
            "example": {
                "product_id": "p123456789",
                "name": "Premium Wireless Headphones",
                "description": "High-quality wireless headphones with noise cancellation",
                "attributes": {
                    "brand": "AudioTech",
                    "category": "Electronics",
                    "subcategories": ["Audio", "Headphones"],
                    "tags": ["wireless", "noise-cancelling", "bluetooth"],
                    "features": {
                        "battery_life": "20 hours",
                        "connectivity": "Bluetooth 5.0"
                    }
                },
                "price_info": {
                    "base_price": 199.99,
                    "current_price": 179.99,
                    "currency": "USD",
                    "discount_percentage": 10
                },
                "inventory": {
                    "stock_level": 50,
                    "reorder_point": 20,
                    "supplier_id": "s789",
                    "lead_time_days": 5,
                    "warehouse_location": "WH-001"
                },
                "seasonal_flags": ["holiday", "back-to-school"]
            }
        }