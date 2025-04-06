import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class BanditArm:
    """Represents a recommendation option (arm) in the multi-armed bandit"""
    item_id: str
    alpha: float  # Success count (e.g., clicks, purchases)
    beta: float   # Failure count
    context_weights: Dict[str, float]  # Weights for different contexts

class ContextualThompsonSampling:
    """Contextual Thompson Sampling implementation for personalized recommendations"""
    def __init__(self, context_features: List[str]):
        self.arms: Dict[str, BanditArm] = {}
        self.context_features = context_features
        self.default_weight = 1.0 / len(context_features)

    def add_arm(self, item_id: str):
        """Add a new item (arm) to the bandit"""
        if item_id not in self.arms:
            self.arms[item_id] = BanditArm(
                item_id=item_id,
                alpha=1.0,  # Prior success count
                beta=1.0,   # Prior failure count
                context_weights={feature: self.default_weight for feature in self.context_features}
            )

    def get_context_score(self, arm: BanditArm, context: Dict[str, float]) -> float:
        """Calculate contextual score for an arm"""
        return sum(arm.context_weights.get(feature, 0) * value
                  for feature, value in context.items())

    def sample(self, context: Dict[str, float]) -> List[Tuple[str, float]]:
        """Sample arms based on Thompson Sampling with context"""
        scores = []
        for arm in self.arms.values():
            # Sample from beta distribution
            base_score = np.random.beta(arm.alpha, arm.beta)
            # Adjust score with context
            context_score = self.get_context_score(arm, context)
            final_score = base_score * (1 + context_score)
            scores.append((arm.item_id, final_score))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def update(self, item_id: str, reward: float, context: Dict[str, float]):
        """Update arm statistics based on reward"""
        if item_id not in self.arms:
            self.add_arm(item_id)

        arm = self.arms[item_id]
        
        # Update success/failure counts
        if reward > 0:
            arm.alpha += reward
        else:
            arm.beta += 1

        # Update context weights
        learning_rate = 0.1
        for feature, value in context.items():
            if feature in arm.context_weights:
                current_weight = arm.context_weights[feature]
                # Increase weight if reward is positive
                arm.context_weights[feature] += learning_rate * reward * value

class RealtimePersonalizationAgent:
    """Agent for real-time personalization using contextual bandits"""
    def __init__(self):
        self.context_features = [
            'time_of_day',
            'day_of_week',
            'device_type',
            'location_type',
            'weather'
        ]
        self.bandit = ContextualThompsonSampling(self.context_features)
        self.session_data = defaultdict(list)

    def _normalize_context(self, raw_context: Dict[str, Any]) -> Dict[str, float]:
        """Normalize context values to float between 0 and 1"""
        normalized = {}

        # Time of day (0-23 hours) -> 0-1
        if 'time_of_day' in raw_context:
            normalized['time_of_day'] = raw_context['time_of_day'] / 23.0

        # Day of week (0-6) -> 0-1
        if 'day_of_week' in raw_context:
            normalized['day_of_week'] = raw_context['day_of_week'] / 6.0

        # Device type encoding
        device_scores = {'mobile': 1.0, 'tablet': 0.7, 'desktop': 0.4}
        if 'device_type' in raw_context:
            normalized['device_type'] = device_scores.get(raw_context['device_type'], 0.5)

        # Location type encoding
        location_scores = {'home': 1.0, 'work': 0.8, 'shopping': 0.6, 'other': 0.4}
        if 'location_type' in raw_context:
            normalized['location_type'] = location_scores.get(raw_context['location_type'], 0.5)

        # Weather encoding
        weather_scores = {'sunny': 1.0, 'cloudy': 0.7, 'rainy': 0.4, 'snowy': 0.3}
        if 'weather' in raw_context:
            normalized['weather'] = weather_scores.get(raw_context['weather'], 0.5)

        return normalized

    def _calculate_reward(self, interaction: Dict[str, Any]) -> float:
        """Calculate reward based on user interaction"""
        reward_weights = {
            'view': 0.1,
            'click': 0.3,
            'add_to_cart': 0.7,
            'purchase': 1.0
        }
        return reward_weights.get(interaction.get('event_type'), 0.0)

    def initialize_products(self, products: List[Dict[str, Any]]):
        """Initialize bandit arms with available products"""
        for product in products:
            self.bandit.add_arm(product['product_id'])

    def track_interaction(self, session_id: str, interaction: Dict[str, Any]):
        """Track user interaction in real-time"""
        self.session_data[session_id].append({
            'timestamp': datetime.utcnow(),
            'interaction': interaction
        })

        # Update bandit with reward
        if 'context' in interaction and 'product_id' in interaction:
            normalized_context = self._normalize_context(interaction['context'])
            reward = self._calculate_reward(interaction)
            self.bandit.update(interaction['product_id'], reward, normalized_context)

    def get_personalized_recommendations(self, session_id: str, context: Dict[str, Any],
                                       n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get real-time personalized recommendations"""
        normalized_context = self._normalize_context(context)
        recommendations = self.bandit.sample(normalized_context)[:n_recommendations]

        # Format recommendations with scores and context influence
        return [{
            'product_id': product_id,
            'confidence_score': float(score),  # Convert numpy float to Python float
            'context_factors': [
                factor for factor, value in normalized_context.items()
                if value > 0.6  # Only include significant context factors
            ]
        } for product_id, score in recommendations]

    def get_session_insights(self, session_id: str) -> Dict[str, Any]:
        """Get insights about the current session"""
        session_interactions = self.session_data.get(session_id, [])
        if not session_interactions:
            return {'status': 'no_data'}

        # Analyze session behavior
        interaction_types = [i['interaction']['event_type'] for i in session_interactions]
        products_viewed = len(set(i['interaction'].get('product_id') for i in session_interactions))
        session_duration = (datetime.utcnow() - session_interactions[0]['timestamp']).seconds

        return {
            'status': 'active',
            'interaction_count': len(interaction_types),
            'products_viewed': products_viewed,
            'session_duration_seconds': session_duration,
            'primary_activity': max(set(interaction_types), key=interaction_types.count),
            'engagement_level': 'high' if len(interaction_types) > 10 else 'medium' if len(interaction_types) > 5 else 'low'
        }