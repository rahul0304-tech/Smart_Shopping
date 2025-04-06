import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertModel, DistilBertTokenizer

class ProductEncoder(nn.Module):
    """Encode product information using DistilBERT"""
    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.feature_projection = nn.Linear(768, 128)  # Project to smaller dimension

    def forward(self, product_info: str) -> torch.Tensor:
        tokens = self.tokenizer(product_info, return_tensors='pt', padding=True, truncation=True)
        outputs = self.encoder(**tokens)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return self.feature_projection(pooled_output)

class SessionTransformer(nn.Module):
    """Transformer model for session-based recommendations"""
    def __init__(self, n_products: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.product_embedding = nn.Embedding(n_products, d_model)
        self.position_embedding = nn.Embedding(50, d_model)  # Max 50 items per session
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.output_layer = nn.Linear(d_model, n_products)

    def forward(self, session_items: torch.Tensor) -> torch.Tensor:
        # session_items shape: (batch_size, seq_length)
        positions = torch.arange(session_items.size(1)).unsqueeze(0).expand_as(session_items)
        item_embeds = self.product_embedding(session_items)
        pos_embeds = self.position_embedding(positions)
        x = item_embeds + pos_embeds
        x = x.permute(1, 0, 2)  # TransformerEncoder expects seq_length first
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Restore batch_size first
        return self.output_layer(x[:, -1, :])  # Predict next item

class ContentBasedRecommender:
    """Content-based recommendation using product features"""
    def __init__(self):
        self.product_encoder = ProductEncoder()
        self.product_features = {}

    def update_product_features(self, products: List[Dict[str, Any]]):
        """Update product feature vectors"""
        for product in products:
            product_info = f"{product['name']} {product['description']} {' '.join(product['attributes']['tags'])}"
            with torch.no_grad():
                features = self.product_encoder(product_info)
                self.product_features[product['product_id']] = features.numpy()

    def get_similar_products(self, product_id: str, n: int = 5) -> List[str]:
        """Find similar products based on content features"""
        if product_id not in self.product_features:
            return []

        query_features = self.product_features[product_id]
        similarities = {}
        
        for pid, features in self.product_features.items():
            if pid != product_id:
                sim = cosine_similarity(query_features.reshape(1, -1), features.reshape(1, -1))[0][0]
                similarities[pid] = sim

        return sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)[:n]

class CollaborativeFilter:
    """Collaborative filtering using user-item interactions"""
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors = {}
        self.item_factors = {}

    def train(self, interactions: List[Dict[str, Any]]):
        """Train collaborative filtering model"""
        # Simplified matrix factorization implementation
        # In production, use a proper matrix factorization library
        user_items = {}
        for interaction in interactions:
            user_id = interaction['user_id']
            item_id = interaction['product_id']
            if user_id not in user_items:
                user_items[user_id] = []
            user_items[user_id].append(item_id)

        # Initialize random factors
        for user_id in user_items:
            self.user_factors[user_id] = np.random.normal(0, 0.1, self.n_factors)

        for interaction in interactions:
            item_id = interaction['product_id']
            if item_id not in self.item_factors:
                self.item_factors[item_id] = np.random.normal(0, 0.1, self.n_factors)

    def get_recommendations(self, user_id: str, n: int = 5) -> List[str]:
        """Get item recommendations for a user"""
        if user_id not in self.user_factors:
            return []

        user_vector = self.user_factors[user_id]
        scores = {}

        for item_id, item_vector in self.item_factors.items():
            score = np.dot(user_vector, item_vector)
            scores[item_id] = score

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:n]

class RecommendationEngine:
    """Main recommendation engine combining multiple approaches"""
    def __init__(self):
        self.content_based = ContentBasedRecommender()
        self.collaborative = CollaborativeFilter()
        self.session_model = None
        self.n_products = 0

    def initialize_session_model(self, n_products: int):
        """Initialize session-based model"""
        self.n_products = n_products
        self.session_model = SessionTransformer(n_products)

    def update_models(self, products: List[Dict[str, Any]], interactions: List[Dict[str, Any]]):
        """Update all recommendation models"""
        # Update content-based features
        self.content_based.update_product_features(products)

        # Train collaborative filtering
        self.collaborative.train(interactions)

        # Session model would be trained separately with PyTorch training loop

    def get_recommendations(self, user_id: str, current_session: List[str],
                           context: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
        """Get personalized recommendations considering all factors"""
        recommendations = []

        # Get recommendations from each model
        if current_session:
            # Session-based recommendations (short-term intent)
            session_items = torch.tensor([int(pid) for pid in current_session]).unsqueeze(0)
            with torch.no_grad():
                session_scores = self.session_model(session_items)
                session_recs = torch.topk(session_scores[0], n).indices.tolist()
        else:
            session_recs = []

        # Collaborative filtering recommendations (user similarity)
        collab_recs = self.collaborative.get_recommendations(user_id, n)

        # Content-based recommendations from last interacted item
        if current_session:
            content_recs = self.content_based.get_similar_products(current_session[-1], n)
        else:
            content_recs = []

        # Combine recommendations with weights
        scores = {}
        weights = {
            'session': 0.5,    # Higher weight for current session intent
            'collab': 0.3,     # Medium weight for collaborative filtering
            'content': 0.2     # Lower weight for content-based
        }

        # Adjust weights based on context
        hour = context.get('time_of_day', datetime.now().hour)
        if 6 <= hour <= 9:  # Morning routine
            weights['session'] = 0.6  # Increase session importance

        # Combine scores
        for i, rec in enumerate(session_recs):
            scores[str(rec)] = scores.get(str(rec), 0) + weights['session'] * (n - i) / n

        for i, rec in enumerate(collab_recs):
            scores[rec] = scores.get(rec, 0) + weights['collab'] * (n - i) / n

        for i, rec in enumerate(content_recs):
            scores[rec] = scores.get(rec, 0) + weights['content'] * (n - i) / n

        # Sort and return top recommendations
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for product_id, score in sorted_recs[:n]:
            recommendations.append({
                'product_id': product_id,
                'score': float(score),
                'reason': self._get_recommendation_reason(product_id, user_id, current_session)
            })

        return recommendations

    def _get_recommendation_reason(self, product_id: str, user_id: str,
                                 current_session: List[str]) -> str:
        """Generate explanation for recommendation"""
        if current_session and product_id in self.content_based.get_similar_products(current_session[-1], 1):
            return "Similar to items you're currently viewing"
        elif user_id in self.collaborative.user_factors and product_id in self.collaborative.get_recommendations(user_id, 1):
            return "Popular among users with similar taste"
        else:
            return "Matches your interests"