import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

class IncrementalDBSCAN:
    """Incremental DBSCAN implementation for online customer segmentation"""
    def __init__(self, eps: float = 0.5, min_samples: int = 5, feature_weights: Dict[str, float] = None):
        self.eps = eps
        self.min_samples = min_samples
        self.feature_weights = feature_weights or {}
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.fitted_data = None
        self.labels = None

    def _extract_features(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from user data"""
        features = []
        
        # Interaction frequency (normalized by time)
        interactions = user_data.get('interaction_history', [])
        recent_interactions = [i for i in interactions
                             if i['timestamp'] >= datetime.utcnow() - timedelta(days=30)]
        features.append(len(recent_interactions) / 30)  # Daily interaction rate

        # Purchase value
        purchase_interactions = [i for i in recent_interactions if i['event_type'] == 'purchase']
        avg_purchase_value = np.mean([i.get('value', 0) for i in purchase_interactions]) if purchase_interactions else 0
        features.append(avg_purchase_value)

        # Price sensitivity
        features.append(user_data.get('preferences', {}).get('price_sensitivity', 0.5))

        # Category diversity
        categories = set(i.get('category') for i in recent_interactions if i.get('category'))
        features.append(len(categories) / 10)  # Normalized by assumed max categories

        # Time-based features
        if recent_interactions:
            times = [i['timestamp'].hour for i in recent_interactions]
            features.append(np.mean(times) / 24)  # Average shopping hour
            features.append(np.std(times) / 12)   # Shopping time variability
        else:
            features.extend([0, 0])

        return np.array(features)

    def partial_fit(self, new_users: List[Dict[str, Any]]):
        """Incrementally update the clustering model with new user data"""
        if not new_users:
            return

        # Extract features from new users
        new_features = np.array([self._extract_features(user) for user in new_users])

        # Update scaler with new data
        if self.fitted_data is None:
            self.fitted_data = new_features
            self.scaler.fit(new_features)
        else:
            # Combine old and new data for scaling
            combined_features = np.vstack([self.fitted_data, new_features])
            self.scaler.fit(combined_features)
            self.fitted_data = combined_features

        # Scale the combined data
        scaled_data = self.scaler.transform(self.fitted_data)

        # Apply feature weights
        for i, feature in enumerate(scaled_data.T):
            weight = self.feature_weights.get(str(i), 1.0)
            scaled_data[:, i] *= weight

        # Rerun DBSCAN on all data
        self.labels = self.dbscan.fit_predict(scaled_data)

    def predict(self, users: List[Dict[str, Any]]) -> List[int]:
        """Predict clusters for new users"""
        if not users:
            return []

        # Extract and scale features
        features = np.array([self._extract_features(user) for user in users])
        scaled_features = self.scaler.transform(features)

        # Apply feature weights
        for i, feature in enumerate(scaled_features.T):
            weight = self.feature_weights.get(str(i), 1.0)
            scaled_features[:, i] *= weight

        return self.dbscan.fit_predict(scaled_features)

    def get_segment_characteristics(self) -> Dict[int, Dict[str, float]]:
        """Get characteristics of each segment"""
        if self.fitted_data is None or self.labels is None:
            return {}

        segment_stats = {}
        scaled_data = self.scaler.transform(self.fitted_data)

        for label in np.unique(self.labels):
            if label == -1:  # Noise points
                continue

            segment_mask = self.labels == label
            segment_data = scaled_data[segment_mask]

            segment_stats[int(label)] = {
                'size': np.sum(segment_mask),
                'interaction_rate': np.mean(self.fitted_data[segment_mask, 0]),
                'avg_purchase_value': np.mean(self.fitted_data[segment_mask, 1]),
                'price_sensitivity': np.mean(self.fitted_data[segment_mask, 2]),
                'category_diversity': np.mean(self.fitted_data[segment_mask, 3]),
                'avg_shopping_hour': np.mean(self.fitted_data[segment_mask, 4] * 24),
                'shopping_time_variability': np.mean(self.fitted_data[segment_mask, 5] * 12)
            }

        return segment_stats

class CustomerSegmentationAgent:
    """Agent responsible for customer segmentation and analysis"""
    def __init__(self):
        self.model = IncrementalDBSCAN(
            eps=0.3,
            min_samples=10,
            feature_weights={
                '0': 2.0,  # Higher weight for interaction frequency
                '1': 1.5,  # Higher weight for purchase value
                '2': 1.0,  # Normal weight for price sensitivity
                '3': 1.2,  # Slightly higher weight for category diversity
                '4': 0.8,  # Lower weight for shopping hour
                '5': 0.8   # Lower weight for time variability
            }
        )

    def update_segments(self, users: List[Dict[str, Any]]):
        """Update customer segments with new user data"""
        self.model.partial_fit(users)

    def get_user_segment(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """Get segment information for a specific user"""
        segment_id = self.model.predict([user])[0]
        segment_stats = self.model.get_segment_characteristics()

        if segment_id == -1:
            return {
                'segment_id': 'undefined',
                'characteristics': 'Unique behavior pattern',
                'recommendations': ['personalized_approach', 'needs_more_data']
            }

        stats = segment_stats.get(segment_id, {})
        
        # Interpret segment characteristics
        characteristics = []
        if stats.get('interaction_rate', 0) > 0.5:
            characteristics.append('high_engagement')
        if stats.get('avg_purchase_value', 0) > 100:
            characteristics.append('high_value')
        if stats.get('price_sensitivity', 0) > 0.7:
            characteristics.append('price_sensitive')
        if stats.get('category_diversity', 0) > 0.5:
            characteristics.append('diverse_interests')

        return {
            'segment_id': f'segment_{segment_id}',
            'characteristics': characteristics,
            'statistics': stats,
            'recommendations': self._generate_recommendations(characteristics)
        }

    def _generate_recommendations(self, characteristics: List[str]) -> List[str]:
        """Generate recommendations based on segment characteristics"""
        recommendations = []

        if 'high_engagement' in characteristics:
            recommendations.extend([
                'loyalty_program_upgrade',
                'early_access_new_products'
            ])

        if 'high_value' in characteristics:
            recommendations.extend([
                'premium_product_recommendations',
                'concierge_service'
            ])

        if 'price_sensitive' in characteristics:
            recommendations.extend([
                'price_drop_alerts',
                'bulk_purchase_discounts'
            ])

        if 'diverse_interests' in characteristics:
            recommendations.extend([
                'cross_category_promotions',
                'discovery_features'
            ])

        return recommendations[:3]  # Return top 3 recommendations