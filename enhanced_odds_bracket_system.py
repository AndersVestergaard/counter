#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED ODDS BRACKET ANALYSIS SYSTEM
Much improved version with better categorization and higher impact

KEY IMPROVEMENTS:
1. More precise bracket categorization
2. Higher default weight (20% instead of 10%)
3. Dynamic weight adjustment based on sample size
4. Better spread and ratio detection
5. Confidence bonuses for strong patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from super_optimized_system import EnhancedSuperOptimizedBettingSystem
from collections import deque
import statistics

class SuperChargedOddsBracketSystem(EnhancedSuperOptimizedBettingSystem):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 🚀 ENHANCED BRACKET PARAMETERS
        self.params.update({
            'odds_bracket_weight': 0.20,        # DOUBLED from 0.10 - higher impact
            'odds_bracket_window': 100,         # Larger window for more data
            'min_bracket_samples': 5,           # Higher minimum for reliability
            'ideal_bracket_samples': 30,        # Higher ideal for confidence
            'bracket_confidence_boost': 1.5,    # Boost strong patterns
            'bracket_dynamic_weight': True,     # Adjust weight based on data quality
        })
    
    def categorize_odds_triplet_enhanced(self, odds_triplet):
        """ENHANCED: More precise odds triplet categorization"""
        if len(odds_triplet) != 3:
            return "invalid"
        
        home_odds, draw_odds, away_odds = odds_triplet
        
        # Handle invalid odds
        if any(odd <= 0 for odd in odds_triplet):
            return "invalid"
        
        # Calculate key metrics
        sorted_odds = sorted(odds_triplet)
        min_odds, mid_odds, max_odds = sorted_odds
        
        spread = max_odds - min_odds
        odds_std = statistics.stdev(odds_triplet)  # Standard deviation for variation
        
        # Identify favorite
        if home_odds == min_odds:
            favorite = "home"
        elif away_odds == min_odds:
            favorite = "away"
        else:
            favorite = "draw"
        
        # Calculate ratios
        min_to_max_ratio = min_odds / max_odds if max_odds > 0 else 0
        draw_position = (draw_odds - min_odds) / (max_odds - min_odds) if spread > 0 else 0.5
        
        # 🎯 ENHANCED CATEGORIZATION RULES
        
        # 1. EXTREME FAVORITES (very low odds)
        if min_odds < 1.3:
            if spread > 8.0:
                return f"extreme_{favorite}_massive_spread"
            elif spread > 4.0:
                return f"extreme_{favorite}_large_spread"
            else:
                return f"extreme_{favorite}_tight"
        
        # 2. VERY EQUAL ODDS (all close)
        elif odds_std < 0.3:  # Very low standard deviation
            return f"very_equal_{favorite}_minimal_diff"
        elif odds_std < 0.6:  # Low standard deviation
            return f"equal_{favorite}_small_diff"
        
        # 3. STRONG FAVORITES (clear but not extreme)
        elif min_odds < 1.8:
            if draw_position < 0.3:  # Draw close to favorite
                return f"strong_{favorite}_draw_close"
            elif draw_position > 0.7:  # Draw close to underdog
                return f"strong_{favorite}_draw_far"
            else:
                return f"strong_{favorite}_draw_middle"
        
        # 4. MODERATE FAVORITES
        elif min_odds < 2.5:
            if min_to_max_ratio > 0.6:  # Relatively close odds
                return f"moderate_{favorite}_competitive"
            else:
                return f"moderate_{favorite}_clear"
        
        # 5. CLOSE CONTESTS (no clear favorite)
        elif min_odds < 3.5 and spread < 2.0:
            if abs(home_odds - away_odds) < 0.3:  # Very close home/away
                return f"tight_contest_even_draw_{int(draw_odds)}"
            else:
                return f"close_contest_slight_{favorite}"
        
        # 6. WIDE SPREAD (chaotic odds)
        elif spread > 4.0:
            return f"wide_spread_{favorite}_chaotic"
        
        # 7. DEFAULT CATEGORIES
        else:
            return f"standard_{favorite}_normal"
    
    def get_enhanced_bracket_confidence(self, odds_triplet):
        """ENHANCED: Better confidence calculation with dynamic weighting"""
        bracket = self.categorize_odds_triplet_enhanced(odds_triplet)
        
        if bracket == "invalid" or bracket not in self.odds_bracket_patterns:
            return {'0': 0.33, '1': 0.33, '2': 0.34}
        
        pattern = self.odds_bracket_patterns[bracket]
        outcomes = list(pattern['outcomes'])
        
        min_samples = int(self.params.get('min_bracket_samples', 5))
        if len(outcomes) < min_samples:
            return {'0': 0.33, '1': 0.33, '2': 0.34}
        
        # Calculate historical probabilities
        total = len(outcomes)
        home_rate = outcomes.count('0') / total
        draw_rate = outcomes.count('1') / total
        away_rate = outcomes.count('2') / total
        
        # 🚀 ENHANCED CONFIDENCE CALCULATION
        
        # 1. Dynamic sample confidence (better scaling)
        ideal_samples = int(self.params.get('ideal_bracket_samples', 30))
        sample_confidence = min(1.0, (total / ideal_samples) ** 0.7)  # Smoother scaling
        
        # 2. Pattern strength bonus
        max_rate = max(home_rate, draw_rate, away_rate)
        pattern_strength = max_rate - 0.33  # How much above random
        
        confidence_boost = 1.0
        if pattern_strength > 0.2:  # Strong pattern (>53%)
            confidence_boost = self.params.get('bracket_confidence_boost', 1.5)
        elif pattern_strength > 0.1:  # Moderate pattern (>43%)
            confidence_boost = 1.2
        
        # 3. Blend with defaults (less blending for strong patterns)
        default_weight = (1.0 - sample_confidence) * (1.0 / confidence_boost)
        historical_weight = 1.0 - default_weight
        
        confidence = {
            '0': (home_rate * historical_weight) + (0.33 * default_weight),
            '1': (draw_rate * historical_weight) + (0.33 * default_weight),
            '2': (away_rate * historical_weight) + (0.34 * default_weight)
        }
        
        # 4. Apply confidence boost to strongest outcome
        max_outcome = max(confidence.keys(), key=lambda k: confidence[k])
        confidence[max_outcome] *= confidence_boost
        
        # 5. Renormalize
        total_conf = sum(confidence.values())
        if total_conf > 0:
            for outcome in confidence:
                confidence[outcome] /= total_conf
        
        return confidence
    
    def get_dynamic_bracket_weight(self, odds_triplet):
        """Calculate dynamic weight based on data quality for this bracket"""
        bracket = self.categorize_odds_triplet_enhanced(odds_triplet)
        
        if not self.params.get('bracket_dynamic_weight', True):
            return self.params.get('odds_bracket_weight', 0.20)
        
        if bracket == "invalid" or bracket not in self.odds_bracket_patterns:
            return 0.05  # Very low weight for unknown patterns
        
        pattern = self.odds_bracket_patterns[bracket]
        outcomes = list(pattern['outcomes'])
        total = len(outcomes)
        
        # Base weight
        base_weight = self.params.get('odds_bracket_weight', 0.20)
        
        # Adjust based on sample size
        min_samples = int(self.params.get('min_bracket_samples', 5))
        ideal_samples = int(self.params.get('ideal_bracket_samples', 30))
        
        if total < min_samples:
            return 0.02  # Almost no weight
        elif total >= ideal_samples:
            return base_weight * 1.5  # Boost weight for rich data
        else:
            # Linear scaling between min and ideal
            scaling = total / ideal_samples
            return base_weight * (0.5 + 0.5 * scaling)
    
    def calculate_match_confidence_enhanced(self, teams, odds, match_index):
        """ENHANCED: Use improved bracket analysis in confidence calculation"""
        # Get base confidence from parent class
        base_confidence = super().calculate_match_confidence(teams, odds, match_index)
        
        # Get odds for this match
        match_odds = odds[match_index * 3:(match_index + 1) * 3]
        if len(match_odds) != 3:
            return base_confidence
        
        # Get enhanced bracket confidence
        bracket_confidence = self.get_enhanced_bracket_confidence(match_odds)
        
        # Get dynamic weight for this specific bracket
        dynamic_bracket_weight = self.get_dynamic_bracket_weight(match_odds)
        
        # Recalculate with enhanced bracket analysis
        odds_diff_weight = self.params.get('odds_diff_weight', 0.15)
        total_new_weight = odds_diff_weight + dynamic_bracket_weight
        
        # Adjust existing weights
        adjusted_odds_weight = self.params['odds_weight'] * (1.0 - total_new_weight)
        adjusted_form_weight = self.params['form_weight'] * (1.0 - total_new_weight)
        adjusted_team_weight = self.params['team_weight'] * (1.0 - total_new_weight)
        
        # Get components from base calculation
        home_odds, draw_odds, away_odds = match_odds
        odds_confidence = {
            '0': 1.0 / home_odds if home_odds > 0 else 0,
            '1': 1.0 / draw_odds if draw_odds > 0 else 0,
            '2': 1.0 / away_odds if away_odds > 0 else 0
        }
        total_odds_conf = sum(odds_confidence.values())
        if total_odds_conf > 0:
            for outcome in odds_confidence:
                odds_confidence[outcome] /= total_odds_conf
        
        # Get team pair info
        team_pair = teams[match_index]
        if isinstance(team_pair, dict) and '1' in team_pair and '2' in team_pair:
            home_team = team_pair['1']
            away_team = team_pair['2']
            
            # Get form and consistency
            home_form = self.get_enhanced_team_form_score(home_team, away_team, is_home=True)
            away_form = self.get_enhanced_team_form_score(away_team, home_team, is_home=False)
            form_difference = home_form - away_form
            
            form_confidence = {
                '0': 0.5 + (form_difference * self.params.get('form_sensitivity', 0.4)),
                '1': 0.5 - abs(form_difference * self.params.get('draw_sensitivity', 0.25)),
                '2': 0.5 - (form_difference * self.params.get('form_sensitivity', 0.4))
            }
            
            for outcome in form_confidence:
                form_confidence[outcome] = max(0.1, min(0.9, form_confidence[outcome]))
            
            team_consistency = self.get_enhanced_team_consistency(home_team, away_team)
        else:
            form_confidence = {'0': 0.33, '1': 0.33, '2': 0.34}
            team_consistency = 0.5
        
        # Enhanced combination with dynamic bracket weight
        enhanced_confidence = {}
        for outcome in ['0', '1', '2']:
            enhanced_confidence[outcome] = (
                adjusted_odds_weight * odds_confidence[outcome] +
                adjusted_form_weight * form_confidence[outcome] +
                adjusted_team_weight * team_consistency +
                odds_diff_weight * base_confidence.get(outcome, 0.33) +  # From odds diff analysis
                dynamic_bracket_weight * bracket_confidence[outcome]     # 🚀 ENHANCED bracket factor
            )
        
        # Apply home bias if applicable
        if isinstance(team_pair, dict) and '1' in team_pair:
            home_odds = match_odds[0]
            if (self.params['home_bias_min_odds'] <= home_odds <= self.params['home_bias_max_odds']):
                enhanced_confidence['0'] *= self.params['home_bias_factor']
                total = sum(enhanced_confidence.values())
                if total > 0:
                    for outcome in enhanced_confidence:
                        enhanced_confidence[outcome] /= total
        
        return enhanced_confidence
    
    # Override the update method to use enhanced categorization
    def update_odds_bracket_patterns(self, odds_triplet, result, match_index):
        """Enhanced bracket pattern updating with better categorization"""
        if not hasattr(self, 'odds_bracket_patterns'):
            self.odds_bracket_patterns = {}
        
        # Use enhanced categorization
        bracket = self.categorize_odds_triplet_enhanced(odds_triplet)
        
        if bracket == "invalid":
            return  # Skip invalid odds
        
        if bracket not in self.odds_bracket_patterns:
            window_size = int(self.params.get('odds_bracket_window', 100))
            self.odds_bracket_patterns[bracket] = {
                'outcomes': deque(maxlen=window_size),
                'total_count': 0,
                'home_wins': 0,
                'draws': 0,
                'away_wins': 0
            }
        
        pattern = self.odds_bracket_patterns[bracket]
        
        # Store the outcome
        pattern['outcomes'].append(result)
        pattern['total_count'] += 1
        
        # Update counters
        if result == '0':
            pattern['home_wins'] += 1
        elif result == '1':
            pattern['draws'] += 1
        else:  # result == '2'
            pattern['away_wins'] += 1


def test_enhanced_system():
    """Test the enhanced odds bracket system"""
    print("🚀 TESTING ENHANCED ODDS BRACKET SYSTEM")
    print("=" * 60)
    
    # Initialize enhanced system
    system = SuperChargedOddsBracketSystem(random_seed=42, verbose=False)
    
    # Test different odds patterns
    test_cases = [
        ([2.33, 2.33, 2.33], "Equal odds - your example"),
        ([1.09, 13, 18], "Extreme favorite"),
        ([1.5, 4.0, 6.0], "Strong home favorite"),
        ([3.1, 3.2, 3.0], "Very tight contest"),
        ([1.2, 8.0, 15.0], "Extreme home vs longshots"),
        ([2.8, 3.1, 2.9], "Close three-way"),
    ]
    
    print("\n📊 ENHANCED CATEGORIZATION & CONFIDENCE:")
    print("-" * 60)
    
    for odds_triplet, description in test_cases:
        bracket = system.categorize_odds_triplet_enhanced(odds_triplet)
        confidence = system.get_enhanced_bracket_confidence(odds_triplet)
        dynamic_weight = system.get_dynamic_bracket_weight(odds_triplet)
        
        print(f"\n{description}:")
        print(f"  Odds: {odds_triplet}")
        print(f"  Enhanced Bracket: {bracket}")
        print(f"  Confidence: H={confidence['0']:.1%}, D={confidence['1']:.1%}, A={confidence['2']:.1%}")
        print(f"  Dynamic Weight: {dynamic_weight:.3f}")
    
    print(f"\n🎯 SYSTEM IMPROVEMENTS:")
    print("-" * 60)
    print(f"  ✅ Enhanced categorization: More precise bracket types")
    print(f"  ✅ Dynamic weighting: {system.params['bracket_dynamic_weight']}")
    print(f"  ✅ Higher base weight: {system.params['odds_bracket_weight']:.1%}")
    print(f"  ✅ Larger data window: {system.params['odds_bracket_window']} samples")
    print(f"  ✅ Confidence boosting: {system.params['bracket_confidence_boost']}x for strong patterns")
    print(f"  ✅ Better sample thresholds: Min={system.params['min_bracket_samples']}, Ideal={system.params['ideal_bracket_samples']}")
    
    return system


if __name__ == "__main__":
    system = test_enhanced_system()
    
    print(f"\n🎉 ENHANCED ODDS BRACKET SYSTEM READY!")
    print("=" * 60)
    print("Key improvements over original:")
    print("✅ 2x higher weight (20% vs 10%)")
    print("✅ Dynamic weight adjustment based on data quality") 
    print("✅ More precise bracket categorization")
    print("✅ Confidence boosting for strong patterns")
    print("✅ Better handling of sample sizes")
    print("")
    print("This should make MUCH more impact on predictions!") 