from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class UserPreferences(BaseModel):
    """User preferences model for personalization"""
    favorite_categories: List[str] = Field(default_factory=list)
    preferred_brands: List[str] = Field(default_factory=list)
    price_sensitivity: float = Field(ge=0, le=1, default=0.5)  # 0: price insensitive, 1: very price sensitive
    shopping_frequency: str = Field(default="medium")  # low, medium, high

class UserDemographics(BaseModel):
    """Anonymized user demographics for GDPR compliance"""
    age_group: str = Field(...)  # e.g., '18-24', '25-34', etc.
    location_region: str = Field(...)  # Region/city level, not exact location
    device_types: List[str] = Field(default_factory=list)
    language_preferences: List[str] = Field(default_factory=list)

class UserInteraction(BaseModel):
    """User interaction events model"""
    event_type: str = Field(...)  # click, add_to_cart, purchase, etc.
    product_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = Field(...)
    dwell_time: Optional[float] = None  # time spent on product in seconds
    context: dict = Field(default_factory=dict)  # weather, time of day, etc.

class UserProfile(BaseModel):
    """Main user profile model"""
    user_id: str = Field(...)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    demographics: Optional[UserDemographics] = None
    last_active: datetime = Field(default_factory=datetime.utcnow)
    segment_labels: List[str] = Field(default_factory=list)  # dynamic user segments
    interaction_history: List[UserInteraction] = Field(default_factory=list)
    consent_flags: dict = Field(default_factory=dict)  # GDPR consent settings
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "u123456789",
                "preferences": {
                    "favorite_categories": ["electronics", "books"],
                    "preferred_brands": ["brand1", "brand2"],
                    "price_sensitivity": 0.7,
                    "shopping_frequency": "high"
                },
                "demographics": {
                    "age_group": "25-34",
                    "location_region": "West Coast",
                    "device_types": ["mobile", "desktop"],
                    "language_preferences": ["en-US"]
                },
                "segment_labels": ["frequent_buyer", "tech_savvy"],
                "consent_flags": {
                    "marketing_emails": True,
                    "personalization": True,
                    "data_collection": True
                }
            }
        }