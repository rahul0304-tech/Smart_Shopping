from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats

class TestVariant(Enum):
    """A/B test variant types"""
    CONTROL = 'control'
    TREATMENT = 'treatment'

@dataclass
class TestResult:
    """Results of an A/B test"""
    variant: TestVariant
    sample_size: int
    conversion_rate: float
    confidence_interval: tuple
    p_value: float
    is_significant: bool

class BayesianABTesting:
    """Bayesian A/B testing implementation"""
    def __init__(self):
        self.prior_alpha = 1
        self.prior_beta = 1

    def update_beliefs(self, successes: int, trials: int) -> tuple:
        """Update beta distribution parameters"""
        posterior_alpha = self.prior_alpha + successes
        posterior_beta = self.prior_beta + (trials - successes)
        return posterior_alpha, posterior_beta

    def calculate_probability(self, control_successes: int, control_trials: int,
                            treatment_successes: int, treatment_trials: int) -> float:
        """Calculate probability that treatment is better than control"""
        control_alpha, control_beta = self.update_beliefs(control_successes, control_trials)
        treatment_alpha, treatment_beta = self.update_beliefs(treatment_successes, treatment_trials)

        samples = 10000
        control_samples = np.random.beta(control_alpha, control_beta, samples)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, samples)
        
        return np.mean(treatment_samples > control_samples)

class PerformanceMetrics:
    """Track and calculate performance metrics"""
    def __init__(self):
        self.metrics = {
            'impressions': 0,
            'clicks': 0,
            'purchases': 0,
            'revenue': 0.0,
            'session_duration': [],
            'recommendation_acceptance': 0,
            'user_retention': set()
        }

    def update_metrics(self, event: Dict[str, Any]):
        """Update metrics based on user event"""
        event_type = event.get('event_type')
        if event_type == 'impression':
            self.metrics['impressions'] += 1
        elif event_type == 'click':
            self.metrics['clicks'] += 1
        elif event_type == 'purchase':
            self.metrics['purchases'] += 1
            self.metrics['revenue'] += event.get('value', 0.0)

        if 'session_duration' in event:
            self.metrics['session_duration'].append(event['session_duration'])

        if event.get('recommendation_accepted'):
            self.metrics['recommendation_acceptance'] += 1

        if 'user_id' in event:
            self.metrics['user_retention'].add(event['user_id'])

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        total_sessions = len(self.metrics['session_duration'])
        return {
            'ctr': self.metrics['clicks'] / max(1, self.metrics['impressions']),
            'conversion_rate': self.metrics['purchases'] / max(1, self.metrics['clicks']),
            'average_order_value': self.metrics['revenue'] / max(1, self.metrics['purchases']),
            'average_session_duration': np.mean(self.metrics['session_duration']) if self.metrics['session_duration'] else 0,
            'recommendation_acceptance_rate': self.metrics['recommendation_acceptance'] / max(1, self.metrics['impressions']),
            'unique_users': len(self.metrics['user_retention'])
        }

class RecommendationExplainer:
    """Generate explanations for recommendations"""
    def __init__(self):
        self.explanation_templates = {
            'collaborative': "Based on preferences of users similar to you",
            'content': "Similar to {item} that you've shown interest in",
            'popularity': "Popular among shoppers in your area",
            'context': "Perfect for {context}",
            'personal': "Matches your {preference} preference"
        }

    def generate_explanation(self, recommendation: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """Generate human-readable explanation for a recommendation"""
        explanation_type = recommendation.get('recommendation_type', 'collaborative')
        template = self.explanation_templates.get(explanation_type)

        if explanation_type == 'content' and 'reference_item' in recommendation:
            return template.format(item=recommendation['reference_item'])
        elif explanation_type == 'context' and 'context_factor' in recommendation:
            return template.format(context=recommendation['context_factor'])
        elif explanation_type == 'personal' and 'matched_preference' in recommendation:
            return template.format(preference=recommendation['matched_preference'])
        
        return template if template else "Recommended based on your shopping patterns"

class PerformanceTestingAgent:
    """Main agent for performance testing and optimization"""
    def __init__(self):
        self.ab_tester = BayesianABTesting()
        self.metrics = PerformanceMetrics()
        self.explainer = RecommendationExplainer()
        self.active_tests: Dict[str, Dict[str, Any]] = {}

    def start_ab_test(self, test_id: str, description: str, variants: List[str]):
        """Start a new A/B test"""
        self.active_tests[test_id] = {
            'description': description,
            'start_time': datetime.utcnow(),
            'variants': variants,
            'data': {variant: {'successes': 0, 'trials': 0} for variant in variants}
        }

    def record_test_event(self, test_id: str, variant: str, success: bool):
        """Record an event for an active test"""
        if test_id in self.active_tests and variant in self.active_tests[test_id]['data']:
            test_data = self.active_tests[test_id]['data'][variant]
            test_data['trials'] += 1
            if success:
                test_data['successes'] += 1

    def get_test_results(self, test_id: str) -> Optional[TestResult]:
        """Get results for an active test"""
        if test_id not in self.active_tests:
            return None

        test = self.active_tests[test_id]
        control_data = test['data']['control']
        treatment_data = test['data']['treatment']

        # Calculate conversion rates
        control_rate = control_data['successes'] / max(1, control_data['trials'])
        treatment_rate = treatment_data['successes'] / max(1, treatment_data['trials'])

        # Calculate confidence intervals
        control_ci = stats.beta.interval(0.95, control_data['successes'] + 1, control_data['trials'] - control_data['successes'] + 1)
        treatment_ci = stats.beta.interval(0.95, treatment_data['successes'] + 1, treatment_data['trials'] - treatment_data['successes'] + 1)

        # Calculate probability of treatment being better
        probability = self.ab_tester.calculate_probability(
            control_data['successes'], control_data['trials'],
            treatment_data['successes'], treatment_data['trials']
        )

        return {
            'control': TestResult(
                variant=TestVariant.CONTROL,
                sample_size=control_data['trials'],
                conversion_rate=control_rate,
                confidence_interval=control_ci,
                p_value=1 - probability,
                is_significant=probability > 0.95
            ),
            'treatment': TestResult(
                variant=TestVariant.TREATMENT,
                sample_size=treatment_data['trials'],
                conversion_rate=treatment_rate,
                confidence_interval=treatment_ci,
                p_value=probability,
                is_significant=probability > 0.95
            )
        }

    def track_event(self, event: Dict[str, Any]):
        """Track a user event and update metrics"""
        self.metrics.update_metrics(event)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of current performance metrics"""
        return self.metrics.get_summary()

    def explain_recommendation(self, recommendation: Dict[str, Any],
                             user_context: Dict[str, Any]) -> str:
        """Generate explanation for a recommendation"""
        return self.explainer.generate_explanation(recommendation, user_context)

    def get_active_tests_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active tests"""
        summaries = []
        for test_id, test in self.active_tests.items():
            results = self.get_test_results(test_id)
            if results:
                summaries.append({
                    'test_id': test_id,
                    'description': test['description'],
                    'duration': (datetime.utcnow() - test['start_time']).days,
                    'total_samples': results['control'].sample_size + results['treatment'].sample_size,
                    'is_significant': results['treatment'].is_significant,
                    'improvement': (results['treatment'].conversion_rate / results['control'].conversion_rate - 1) * 100
                    if results['control'].conversion_rate > 0 else 0
                })
        return summaries