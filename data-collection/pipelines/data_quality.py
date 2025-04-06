from typing import List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from connectors.database import DatabaseConnector

@dataclass
class DataQualityMetrics:
    """Data quality metrics container"""
    completeness: float  # Percentage of required fields present
    validity: float     # Percentage of valid values
    timeliness: float   # Percentage of data within acceptable time range
    consistency: float  # Percentage of consistent data across sources

class DataQualityChecker:
    """Data quality checking implementation"""
    def __init__(self):
        self.db = DatabaseConnector()

    def check_completeness(self, data: Dict[str, Any], required_fields: List[str]) -> float:
        """Check if all required fields are present"""
        if not data:
            return 0.0
        present_fields = sum(1 for field in required_fields if field in data and data[field] is not None)
        return present_fields / len(required_fields)

    def check_validity(self, data: Dict[str, Any], validation_rules: Dict[str, callable]) -> float:
        """Check if data values are valid according to rules"""
        if not data or not validation_rules:
            return 0.0
        valid_count = sum(1 for field, rule in validation_rules.items()
                         if field in data and rule(data[field]))
        return valid_count / len(validation_rules)

    def check_timeliness(self, timestamp: datetime, max_age: timedelta) -> float:
        """Check if data is within acceptable time range"""
        if not timestamp:
            return 0.0
        age = datetime.utcnow() - timestamp
        return 1.0 if age <= max_age else 0.0

    def check_consistency(self, data: Dict[str, Any], reference_data: Dict[str, Any]) -> float:
        """Check data consistency across sources"""
        if not data or not reference_data:
            return 0.0
        common_fields = set(data.keys()) & set(reference_data.keys())
        if not common_fields:
            return 0.0
        consistent_count = sum(1 for field in common_fields
                             if data[field] == reference_data[field])
        return consistent_count / len(common_fields)

def check_data_quality(**context):
    """Main data quality check function for Airflow DAG"""
    checker = DataQualityChecker()
    quality_issues = []

    # Get database connections
    mongo_users = checker.db.get_mongo_collection('users')
    mongo_products = checker.db.get_mongo_collection('products')

    # Check user data quality
    user_required_fields = ['user_id', 'preferences', 'last_active']
    user_validation_rules = {
        'user_id': lambda x: isinstance(x, str) and len(x) > 0,
        'last_active': lambda x: isinstance(x, datetime),
        'preferences': lambda x: isinstance(x, dict)
    }

    # Sample recent users for quality checks
    recent_users = mongo_users.find({'last_active': {'$gte': datetime.utcnow() - timedelta(days=1)}})
    
    for user in recent_users:
        metrics = DataQualityMetrics(
            completeness=checker.check_completeness(user, user_required_fields),
            validity=checker.check_validity(user, user_validation_rules),
            timeliness=checker.check_timeliness(user.get('last_active'), timedelta(days=30)),
            consistency=1.0  # Placeholder for cross-source consistency check
        )

        if any(metric < 0.9 for metric in [metrics.completeness, metrics.validity, metrics.timeliness]):
            quality_issues.append({
                'entity_type': 'user',
                'entity_id': user.get('user_id'),
                'metrics': metrics.__dict__,
                'timestamp': datetime.utcnow()
            })

    # Check product data quality
    product_required_fields = ['product_id', 'name', 'price_info', 'inventory']
    product_validation_rules = {
        'product_id': lambda x: isinstance(x, str) and len(x) > 0,
        'name': lambda x: isinstance(x, str) and len(x) > 0,
        'price_info': lambda x: isinstance(x, dict) and 'current_price' in x,
        'inventory': lambda x: isinstance(x, dict) and 'stock_level' in x
    }

    recent_products = mongo_products.find({'updated_at': {'$gte': datetime.utcnow() - timedelta(days=1)}})

    for product in recent_products:
        metrics = DataQualityMetrics(
            completeness=checker.check_completeness(product, product_required_fields),
            validity=checker.check_validity(product, product_validation_rules),
            timeliness=checker.check_timeliness(product.get('updated_at'), timedelta(days=7)),
            consistency=1.0  # Placeholder for cross-source consistency check
        )

        if any(metric < 0.9 for metric in [metrics.completeness, metrics.validity, metrics.timeliness]):
            quality_issues.append({
                'entity_type': 'product',
                'entity_id': product.get('product_id'),
                'metrics': metrics.__dict__,
                'timestamp': datetime.utcnow()
            })

    # Store quality issues in MongoDB for monitoring
    if quality_issues:
        checker.db.get_mongo_collection('data_quality_issues').insert_many(quality_issues)

    # Raise error if too many quality issues
    if len(quality_issues) > 100:  # Threshold can be adjusted
        raise ValueError(f"Too many data quality issues found: {len(quality_issues)}")

    return len(quality_issues) == 0