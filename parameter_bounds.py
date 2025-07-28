#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PARAMETER BOUNDS CONFIGURATION
Shared parameter bounds for both the optimizer and prediction system
"""

def get_parameter_bounds():
    """Define parameter bounds for validation and optimization"""
    return {
        # Core weights
        'odds_weight': (0.05, 4.0),
        'team_weight': (0.01, 2.0),
        'form_weight': (0.001, 2.0),
        
        # Pattern generation
        'default_patterns': (25, 200),  # Number of bets to make
        
        # Bias parameters
        'home_bias_max_odds': (0.5, 10.0),
        'home_bias_factor': (0.3, 2.0),
        
        # Streak parameters
        'winning_streak_boost': (0.5, 3.0),
        'losing_streak_penalty': (0.1, 2.0),
        'streak_length': (1, 5),
        
        # Form boost parameters
        'home_form_boost': (0.3, 3.0),
        'away_form_boost': (0.3, 3.0),
        
        # Enhanced opponent analysis weights (optimizable ranges)
        'base_form_weight': (0.1, 0.8),         # Traditional form weight
        'opponent_pattern_weight': (0.05, 0.5), # Opponent patterns weight
        'head_to_head_weight': (0.05, 0.4),     # Head-to-head history weight
        'tendency_weight': (0.02, 0.3),         # Recent tendencies weight
        'contextual_weight': (0.02, 0.3),       # Contextual form weight
        
        # Enhanced consistency weights (should sum to ~1.0)
        'consistency_base_weight': (0.2, 0.8),  # Base consistency weight
        'consistency_h2h_weight': (0.1, 0.6),   # H2H consistency weight
        'consistency_opponent_weight': (0.1, 0.5), # Opponent pattern consistency weight
        
        # Form confidence sensitivity
        'form_sensitivity': (0.1, 0.8),         # Form difference sensitivity
        'draw_sensitivity': (0.1, 0.5),         # Draw likelihood sensitivity
        
        # Odds difference analysis bounds (optimizable ranges)
        'odds_diff_weight': (0.05, 0.4),        # Weight for odds difference analysis
        'equal_match_boost': (0.8, 2.0),        # Boost for equal match performers  
        'favorite_performance_boost': (0.8, 1.8), # Boost for strong favorites
        'underdog_performance_boost': (0.9, 2.5), # Boost for strong underdogs
        'odds_diff_threshold_tight': (0.1, 0.6), # Tight odds difference threshold
        'odds_diff_threshold_moderate': (0.4, 1.2), # Moderate odds difference threshold
        'odds_diff_sensitivity': (0.2, 1.0),     # Sensitivity to odds patterns
        'odds_diff_window': (4, 15),             # Historical window size
    }

def get_default_parameters():
    """Define default parameter values"""
    return {
        # Core weights
        'odds_weight': 0.393,
        'team_weight': 0.276,
        'form_weight': 0.335,
        
        # Pattern generation
        'default_patterns': 60,
        
        # Confidence thresholds
        'high_confidence_threshold': 0.65,
        'medium_confidence_threshold': 0.55,
        'max_confidence': 0.95,
        
        # Form analysis
        'form_win_weight': 3.0,
        'form_draw_weight': 1.0,
        'form_loss_weight': 0.0,
        'form_window': 5,
        
        # Home bias parameters
        'home_bias_min_odds': 1.3,
        'home_bias_max_odds': 2.5,
        'home_bias_factor': 0.868,
        
        # Streak adjustments
        'winning_streak_boost': 2.898,
        'losing_streak_penalty': 0.85,
        'streak_length': 3,
        
        # Form boosts
        'home_form_boost': 1.212279924526011,
        'away_form_boost': 1.1,
        'strong_form_threshold': 0.7,
        
        # Enhanced opponent analysis weights
        'base_form_weight': 0.4,
        'opponent_pattern_weight': 0.25,
        'head_to_head_weight': 0.15,
        'tendency_weight': 0.1,
        'contextual_weight': 0.1,
        
        # Enhanced consistency weights
        'consistency_base_weight': 0.5,
        'consistency_h2h_weight': 0.3,
        'consistency_opponent_weight': 0.2,
        
        # Form confidence sensitivity
        'form_sensitivity': 0.4,
        'draw_sensitivity': 0.25,
        
        # Odds difference analysis parameters
        'odds_diff_weight': 0.15,
        'equal_match_boost': 1.2,
        'favorite_performance_boost': 1.1,
        'underdog_performance_boost': 1.3,
        'odds_diff_threshold_tight': 0.3,
        'odds_diff_threshold_moderate': 0.7,
        'odds_diff_sensitivity': 0.5,
        'odds_diff_window': 8,
    } 