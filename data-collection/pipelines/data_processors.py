from datetime import datetime, timedelta
from typing import List, Dict, Any
from collections import defaultdict

from connectors.database import DatabaseConnector

class DataProcessor:
    """Base class for data processing operations"""
    def __init__(self):
        self.db = DatabaseConnector()

    def _get_timestamp(self) -> datetime:
        return datetime.utcnow()

class UserInteractionProcessor(DataProcessor):
    """Process user interaction data"""
    def process_interactions(self, interactions: List[Dict[str, Any]]):
        """Process batch of user interactions"""
        mongo_users = self.db.get_mongo_collection('users')
        redis_client = self.db.get_redis_client()

        for interaction in interactions:
            user_id = interaction.get('user_id')
            if not user_id:
                continue

            # Update user's last activity
            mongo_users.update_one(
                {'user_id': user_id},
                {'$set': {'last_active': self._get_timestamp()},
                 '$push': {'interaction_history': interaction}}
            )

            # Cache recent interactions for real-time recommendations
            redis_key = f'user:{user_id}:recent_interactions'
            redis_client.lpush(redis_key, str(interaction))
            redis_client.ltrim(redis_key, 0, 99)  # Keep last 100 interactions

class ProductProcessor(DataProcessor):
    """Process product updates and metrics"""
    def process_updates(self, products: List[Dict[str, Any]]):
        """Process batch of product updates"""
        mongo_products = self.db.get_mongo_collection('products')
        redis_client = self.db.get_redis_client()

        for product in products:
            product_id = product.get('product_id')
            if not product_id:
                continue

            # Update product information
            product['updated_at'] = self._get_timestamp()
            mongo_products.update_one(
                {'product_id': product_id},
                {'$set': product},
                upsert=True
            )

            # Cache product data for quick access
            redis_client.setex(
                f'product:{product_id}',
                timedelta(hours=24),
                str(product)
            )

class UserSegmentProcessor(DataProcessor):
    """Process user segmentation"""
    def update_segments(self):
        """Update user segments based on recent behavior"""
        mongo_users = self.db.get_mongo_collection('users')
        recent_users = mongo_users.find({
            'last_active': {'$gte': datetime.utcnow() - timedelta(days=30)}
        })

        for user in recent_users:
            segments = self._calculate_segments(user)
            mongo_users.update_one(
                {'user_id': user['user_id']},
                {'$set': {'segment_labels': segments}}
            )

    def _calculate_segments(self, user: Dict[str, Any]) -> List[str]:
        """Calculate user segments based on behavior and preferences"""
        segments = []
        interactions = user.get('interaction_history', [])
        
        if not interactions:
            return ['new_user']

        # Frequency-based segmentation
        monthly_interactions = sum(1 for i in interactions
                                 if i['timestamp'] >= datetime.utcnow() - timedelta(days=30))
        if monthly_interactions > 20:
            segments.append('frequent_buyer')
        elif monthly_interactions > 10:
            segments.append('regular_buyer')
        else:
            segments.append('occasional_buyer')

        # Price sensitivity segmentation
        if user.get('preferences', {}).get('price_sensitivity', 0.5) > 0.7:
            segments.append('price_sensitive')
        
        return segments

class ProductMetricsProcessor(DataProcessor):
    """Process product metrics"""
    def update_metrics(self):
        """Update product metrics based on recent interactions"""
        mongo_products = self.db.get_mongo_collection('products')
        mongo_interactions = self.db.get_mongo_collection('interactions')

        # Get recent interactions
        recent_interactions = mongo_interactions.find({
            'timestamp': {'$gte': datetime.utcnow() - timedelta(days=7)}
        })

        # Aggregate metrics by product
        metrics = defaultdict(lambda: {
            'view_count': 0,
            'purchase_count': 0,
            'add_to_cart_count': 0
        })

        for interaction in recent_interactions:
            product_id = interaction.get('product_id')
            event_type = interaction.get('event_type')
            
            if not (product_id and event_type):
                continue

            if event_type == 'view':
                metrics[product_id]['view_count'] += 1
            elif event_type == 'purchase':
                metrics[product_id]['purchase_count'] += 1
            elif event_type == 'add_to_cart':
                metrics[product_id]['add_to_cart_count'] += 1

        # Update product metrics
        for product_id, product_metrics in metrics.items():
            # Calculate conversion rate
            views = product_metrics['view_count']
            purchases = product_metrics['purchase_count']
            conversion_rate = purchases / views if views > 0 else 0

            mongo_products.update_one(
                {'product_id': product_id},
                {'$set': {
                    'metrics': {
                        **product_metrics,
                        'conversion_rate': conversion_rate
                    }
                }}
            )

# Airflow DAG task functions
def process_user_interactions(**context):
    processor = UserInteractionProcessor()
    # Get interactions from context or source system
    interactions = context.get('task_instance').xcom_pull(task_ids='fetch_interactions')
    processor.process_interactions(interactions or [])

def process_product_updates(**context):
    processor = ProductProcessor()
    # Get product updates from context or source system
    products = context.get('task_instance').xcom_pull(task_ids='fetch_products')
    processor.process_updates(products or [])

def update_user_segments(**context):
    processor = UserSegmentProcessor()
    processor.update_segments()

def update_product_metrics(**context):
    processor = ProductMetricsProcessor()
    processor.update_metrics()