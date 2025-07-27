#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMART OPTIMIZER: Avoids endless loops with intelligent sampling
Uses gradient-free optimization techniques to maximize total winnings
Now with MULTIPROCESSING for faster optimization!
"""

import json
import os
import sys
import time
import random
import multiprocessing as mp
from multiprocessing import Process, Queue, cpu_count
from comprehensive_winnings_test import EnhancedSuperOptimizedBettingSystem, calculate_winnings, load_test_file, load_all_complete_test_files


class SmartOptimizer:
    def __init__(self):
        # Load optimized parameters from file if available
        optimized_params = self.load_latest_optimized_parameters()
        
        # Define all default parameters first
        default_params = {
                'odds_weight': 0.393,
                'team_weight': 0.276,
                'form_weight': 0.335,
                'default_patterns': 60,
                'high_confidence_threshold': 0.65,
                'medium_confidence_threshold': 0.55,
                'max_confidence': 0.95,
                'form_win_weight': 3.0,
                'form_draw_weight': 1.0,
                'form_loss_weight': 0.0,
                'form_window': 5,
                'home_bias_min_odds': 1.3,
                'home_bias_max_odds': 2.5,
                'home_bias_factor': 0.868,
                'winning_streak_boost': 2.898,
                'losing_streak_penalty': 0.85,
                'streak_length': 3,
                'home_form_boost': 1.212279924526011,
                'away_form_boost': 1.1,
                'strong_form_threshold': 0.7,
                
                # Enhanced opponent analysis weights (optimizable)
                'base_form_weight': 0.4,
                'opponent_pattern_weight': 0.25,
                'head_to_head_weight': 0.15,
                'tendency_weight': 0.1,
                'contextual_weight': 0.1,
                
                # Enhanced consistency weights (optimizable)
                'consistency_base_weight': 0.5,
                'consistency_h2h_weight': 0.3,
                'consistency_opponent_weight': 0.2,
                
                # Form confidence sensitivity (optimizable)
                'form_sensitivity': 0.4,
                'draw_sensitivity': 0.25,
                
                # 🆕 NEW: ODDS DIFFERENCE ANALYSIS PARAMETERS (optimizable)
                'odds_diff_weight': 0.15,        # Weight for odds difference analysis in confidence
                'equal_match_boost': 1.2,        # Boost for teams that perform well in equal matches
                'favorite_performance_boost': 1.1, # Boost for teams that perform well as favorites
                'underdog_performance_boost': 1.3, # Boost for teams that perform well as underdogs
                'odds_diff_threshold_tight': 0.3,  # Threshold for "tight" odds differences
                'odds_diff_threshold_moderate': 0.7, # Threshold for "moderate" odds differences  
                'odds_diff_sensitivity': 0.5,    # Sensitivity to odds difference patterns
                'odds_diff_window': 8,           # Number of historical matches to analyze for odds patterns
            }
        
        # Merge optimized parameters with defaults (ensures all parameters are present)
        if optimized_params:
            # Start with defaults, then update with optimized values
            self.best_params = default_params.copy()
            self.best_params.update(optimized_params)
            print(f"   ✅ Starting with optimized parameters from file (merged with new defaults)")
        else:
            # Use defaults only
            self.best_params = default_params
            print(f"   ⚠️  No optimized parameters found, using defaults")
        
        # Parameter bounds for optimization (MASSIVELY EXTREME search for 100k iterations)
        self.param_bounds = {
            'odds_weight': (0.05, 4.0),
            'team_weight': (0.01, 2.0),
            'form_weight': (0.001, 2.0),
            'default_patterns': (25, 125),  # Number of bets to make
            'home_bias_max_odds': (0.5, 10.0),
            'home_bias_factor': (0.3, 2.0),
            'winning_streak_boost': (0.5, 3.0),
            'losing_streak_penalty': (0.1, 2.0),
            'streak_length': (1, 5),
            'home_form_boost': (0.3, 3.0),
            'away_form_boost': (0.3, 3.0),
            
            # NEW: Enhanced opponent analysis weights (optimizable ranges)
            'base_form_weight': (0.1, 0.8),         # Traditional form weight
            'opponent_pattern_weight': (0.05, 0.5), # Opponent patterns weight
            'head_to_head_weight': (0.05, 0.4),     # Head-to-head history weight
            'tendency_weight': (0.02, 0.3),         # Recent tendencies weight
            'contextual_weight': (0.02, 0.3),       # Contextual form weight
            
            # NEW: Enhanced consistency weights (should sum to ~1.0)
            'consistency_base_weight': (0.2, 0.8),  # Base consistency weight
            'consistency_h2h_weight': (0.1, 0.6),   # H2H consistency weight
            'consistency_opponent_weight': (0.1, 0.5), # Opponent pattern consistency weight
            
            # NEW: Form confidence sensitivity
            'form_sensitivity': (0.1, 0.8),         # Form difference sensitivity
            'draw_sensitivity': (0.1, 0.5),         # Draw likelihood sensitivity
            
            # 🆕 NEW: ODDS DIFFERENCE ANALYSIS BOUNDS (optimizable ranges)
            'odds_diff_weight': (0.05, 0.4),        # Weight for odds difference analysis
            'equal_match_boost': (0.8, 2.0),        # Boost for equal match performers  
            'favorite_performance_boost': (0.8, 1.8), # Boost for strong favorites
            'underdog_performance_boost': (0.9, 2.5), # Boost for strong underdogs
            'odds_diff_threshold_tight': (0.1, 0.6), # Tight odds difference threshold
            'odds_diff_threshold_moderate': (0.4, 1.2), # Moderate odds difference threshold
            'odds_diff_sensitivity': (0.2, 1.0),     # Sensitivity to odds patterns
            'odds_diff_window': (4, 15),             # Historical window size
        }
        
        # Multi-objective optimization weights
        self.winnings_weight = 0.7  # Weight for total winnings
        self.win_rate_weight = 0.3  # Weight for win frequency (weeks with 12+ correct)
        
        self.best_winnings = 0  # Starting winnings - will find the actual best through optimization
        self.test_files = []
        self.load_test_files()
        
    def load_test_files(self):
        """Load ALL test files with complete data for accurate optimization"""
        self.test_files = load_all_complete_test_files()
    
    def validate_weights(self, odds_w, team_w, form_w):
        """Ensure weights sum to approximately 1.0"""
        total = odds_w + team_w + form_w
        return 0.9 <= total <= 1.1
    
    def validate_all_weights(self, params):
        """Comprehensive validation for all weight groups"""
        # Original weight validation
        if not self.validate_weights(params['odds_weight'], params['team_weight'], params['form_weight']):
            return False
        
        # Enhanced form weights should sum to approximately 1.0
        form_total = (params.get('base_form_weight', 0.4) + 
                     params.get('opponent_pattern_weight', 0.25) + 
                     params.get('head_to_head_weight', 0.15) + 
                     params.get('tendency_weight', 0.1) + 
                     params.get('contextual_weight', 0.1))
        if not (0.9 <= form_total <= 1.1):
            return False
        
        # Consistency weights should sum to approximately 1.0
        consistency_total = (params.get('consistency_base_weight', 0.5) + 
                            params.get('consistency_h2h_weight', 0.3) + 
                            params.get('consistency_opponent_weight', 0.2))
        if not (0.9 <= consistency_total <= 1.1):
            return False
        
        return True
    
    def evaluate_parameters(self, params):
        """Evaluate a parameter set with multi-objective scoring: winnings + win rate + weekly weight capping"""
        try:
            # Validate all weights (including new enhanced parameters)
            if not self.validate_all_weights(params):
                return -10000  # Invalid weight combination
            
            system = EnhancedSuperOptimizedBettingSystem(random_seed=42, verbose=False)
            system.params.update(params)
            
            total_winnings = 0
            total_cost = 0
            weeks_with_wins = 0  # Count weeks where we got 12+ correct
            total_weeks = 0
            weekly_contributions = []  # Track each week's contribution to prevent overfitting
            
            for test_data in self.test_files:
                filename = test_data['filename']
                odds = test_data['odds']
                teams = test_data['teams']
                result = test_data['result']
                penge_values = test_data['penge']
                
                if not result or not odds:
                    continue
                
                patterns = system.generate_optimized_patterns(odds, teams)
                total_weeks += 1
                
                # Track best performance for this week
                week_best_correct = 0
                week_winnings = 0
                week_cost = len(patterns)
                
                for pattern in patterns:
                    winnings = calculate_winnings(pattern, result, penge_values)
                    week_winnings += winnings
                    
                    # Count correct predictions for this pattern
                    correct_count = sum(1 for pred, actual in zip(pattern, result) if pred == actual)
                    week_best_correct = max(week_best_correct, correct_count)
                
                # 🎯 WEEKLY WEIGHT CAPPING: Limit influence of weeks with excessive winnings
                week_profit = week_winnings - week_cost
                
                # Cap weekly profit contribution to prevent overfitting on "easy" weeks
                max_weekly_profit = 10000  # Cap at 10k points per week max influence
                capped_week_profit = min(week_profit, max_weekly_profit) if week_profit > 0 else week_profit
                
                # Also cap weekly loss to prevent one bad week from dominating
                min_weekly_loss = -2000  # Don't let one week contribute more than 2k loss
                capped_week_profit = max(capped_week_profit, min_weekly_loss)
                
                # Track for balanced optimization
                weekly_contributions.append({
                    'filename': filename,
                    'original_profit': week_profit,
                    'capped_profit': capped_week_profit,
                    'patterns_count': len(patterns)
                })
                
                total_winnings += week_winnings
                total_cost += week_cost
                
                # Count this week as a win if best pattern got 12+ correct
                if week_best_correct >= 12:
                    weeks_with_wins += 1
            
            if total_cost == 0 or total_weeks == 0:
                return -10000
            
            # 🎯 NEW: Calculate metrics using CAPPED weekly contributions for balanced optimization
            capped_total_profit = sum(week['capped_profit'] for week in weekly_contributions)
            net_profit = total_winnings - total_cost
            win_rate = weeks_with_wins / total_weeks
            
            # Multi-objective composite score using CAPPED profits
            # This prevents weeks with 50+ winning patterns from dominating optimization
            normalized_capped_profit = capped_total_profit / 1000.0
            
            # Composite score: balance between CAPPED winnings and win frequency
            composite_score = (self.winnings_weight * normalized_capped_profit + 
                             self.win_rate_weight * win_rate * 10000)  # Scale win_rate up
            
            # Additional diversity bonus: reward systems that perform well across many weeks
            weeks_with_positive_capped_profit = sum(1 for week in weekly_contributions if week['capped_profit'] > 0)
            diversity_bonus = (weeks_with_positive_capped_profit / total_weeks) * 1000  # Up to 1000 bonus
            
            composite_score += diversity_bonus * 0.1  # 10% weight for diversity
            
            return composite_score
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return -10000
    
    def random_neighbor(self, params, step_size=0.15, iteration=0):
        """Generate a random neighbor of current parameters with AGGRESSIVE variation"""
        new_params = params.copy()
        
        # Use iteration to vary the random state while keeping reproducible results [[memory:4430342]]
        # This way same iteration always produces same result, but different iterations vary
        temp_state = random.getstate()
        random.seed(42 + iteration * 17)  # Different seed per iteration, but reproducible
        
        # Pick 1-5 random parameters to modify (more changes per iteration)
        param_names = list(self.param_bounds.keys())
        num_changes = random.randint(1, 5)
        to_change = random.sample(param_names, num_changes)
        
        # 80% chance for LARGE jumps to aggressively explore
        large_jump = random.random() < 0.8
        
        for param in to_change:
            if param in ['streak_length', 'default_patterns', 'odds_diff_window']:
                # Integer parameters
                min_val, max_val = self.param_bounds[param]
                current = new_params[param]
                if large_jump:
                    # Big jump: random value in range
                    new_val = random.randint(min_val, max_val)
                else:
                    # Small step
                    change = random.choice([-2, -1, 1, 2])
                    new_val = max(min_val, min(max_val, current + change))
                new_params[param] = new_val
            else:
                # Float parameters
                min_val, max_val = self.param_bounds[param]
                current = new_params[param]
                if large_jump:
                    # Big jump: random value in range
                    new_val = random.uniform(min_val, max_val)
                else:
                    # Variable step size (sometimes bigger steps)
                    dynamic_step = step_size * random.uniform(0.5, 2.0)
                    change = random.uniform(-dynamic_step, dynamic_step)
                    new_val = max(min_val, min(max_val, current + change))
                new_params[param] = new_val
        
        # Restore original random state
        random.setstate(temp_state)
        
        return new_params
    
    def get_detailed_metrics(self, params):
        """Get detailed metrics for a parameter set: winnings, win rate, etc."""
        try:
            # Validate all weights (including new enhanced parameters)
            if not self.validate_all_weights(params):
                return None
            
            system = EnhancedSuperOptimizedBettingSystem(random_seed=42, verbose=False)
            system.params.update(params)
            
            total_winnings = 0
            total_cost = 0
            weeks_with_wins = 0
            total_weeks = 0
            
            for test_data in self.test_files:
                filename = test_data['filename']
                odds = test_data['odds']
                teams = test_data['teams']
                result = test_data['result']
                penge_values = test_data['penge']
                
                if not result or not odds:
                    continue
                
                patterns = system.generate_optimized_patterns(odds, teams)
                total_weeks += 1
                
                # Track best performance for this week
                week_best_correct = 0
                
                for pattern in patterns:
                    winnings = calculate_winnings(pattern, result, penge_values)
                    total_winnings += winnings
                    total_cost += 1
                    
                    # Count correct predictions for this pattern
                    correct_count = sum(1 for pred, actual in zip(pattern, result) if pred == actual)
                    week_best_correct = max(week_best_correct, correct_count)
                
                # Count this week as a win if best pattern got 12+ correct
                if week_best_correct >= 12:
                    weeks_with_wins += 1
            
            if total_cost == 0 or total_weeks == 0:
                return None
            
            # Calculate all metrics
            net_profit = total_winnings - total_cost
            win_rate = weeks_with_wins / total_weeks
            roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
            
            return {
                'total_winnings': total_winnings,
                'total_cost': total_cost,
                'net_profit': net_profit,
                'win_rate': win_rate,
                'weeks_with_wins': weeks_with_wins,
                'total_weeks': total_weeks,
                'roi': roi
            }
            
        except Exception as e:
            return None
    
    def hill_climbing_optimization(self, max_iterations=1000, max_time_minutes=15):
        """Hill climbing optimization with time limit"""
        print(f"🚀 SMART OPTIMIZATION")
        print(f"   Max iterations: {max_iterations:,}")
        print(f"   Max time: {max_time_minutes} minutes")
        print(f"   📊 Using {len(self.test_files)} complete data files")
        print("=" * 60)
        
        current_params = self.best_params.copy()
        current_winnings = self.best_winnings
        
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        
        improvements = 0
        
        # First, evaluate current parameters to establish baseline
        print("🔍 Evaluating starting parameters...")
        current_winnings = self.evaluate_parameters(current_params)
        print(f"✅ Starting Winnings: {current_winnings:+.1f}")
        print()
        
        for iteration in range(max_iterations):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_time_seconds:
                print(f"\n⏰ Time limit reached ({elapsed:.1f}s)")
                break
            
            # Generate neighbor
            neighbor_params = self.random_neighbor(current_params, iteration=iteration)
            
            # Show minimal progress - just iteration number
            print(f"\r{iteration+1:,}/{max_iterations}", end="", flush=True)
            
            # Evaluate neighbor
            neighbor_winnings = self.evaluate_parameters(neighbor_params)
            
            # Only print if we found an improvement
            if neighbor_winnings > current_winnings:
                improvement = neighbor_winnings - current_winnings
                print(f"\n🎯 IMPROVEMENT {iteration+1:,}: {neighbor_winnings:+.1f} (+{improvement:.1f})")
                print(f"   📊 Parameters: odds={neighbor_params['odds_weight']:.3f}, team={neighbor_params['team_weight']:.3f}, form={neighbor_params['form_weight']:.3f}")
                print(f"                 bets={neighbor_params['default_patterns']}, streak_boost={neighbor_params['winning_streak_boost']:.3f}, bias_factor={neighbor_params['home_bias_factor']:.3f}")
                
                current_params = neighbor_params
                current_winnings = neighbor_winnings
                improvements += 1
        
        print("\n" + "=" * 60)
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"   Iterations: {iteration + 1}")
        print(f"   Improvements found: {improvements}")
        print(f"   Final Winnings: {current_winnings:+.1f}")
        
        if improvements > 0:
            print(f"   🏆 Best parameters found:")
            print(f"     odds_weight: {current_params['odds_weight']:.3f}")
            print(f"     team_weight: {current_params['team_weight']:.3f}")
            print(f"     form_weight: {current_params['form_weight']:.3f}")
            print(f"     default_patterns: {current_params['default_patterns']} bets")
            print(f"     winning_streak_boost: {current_params['winning_streak_boost']:.3f}")
            print(f"     home_bias_factor: {current_params['home_bias_factor']:.3f}")
            
            # Ask user if they want to save parameters
            save_params = input(f"\n💾 Save optimized parameters (Winnings: {current_winnings:+.1f}) to file? (y/n): ").strip().lower()
            if save_params == 'y':
                self.save_parameters_to_file(current_params, current_winnings, improvements)
        else:
            print(f"   📊 No improvements found - current parameters may already be optimal")
        
        return current_params, current_winnings
    
    def _worker_hill_climbing(self, worker_id, start_params, iterations_per_worker, max_time_seconds, progress_queue, result_queue):
        """Worker function for parallel hill climbing optimization"""
        # Each worker gets its own random seed based on worker_id to ensure reproducibility [[memory:4430342]]
        worker_seed = 42 + worker_id * 1000
        random.seed(worker_seed)
        
        # Load test files in this worker process
        test_files = load_all_complete_test_files()
        
        # Initialize worker's best parameters (slightly randomized from start_params)
        current_params = start_params.copy()
        
        # Slightly randomize starting parameters for each worker to explore different regions
        if worker_id > 0:
            temp_state = random.getstate()
            random.seed(worker_seed)
            for param_name in ['odds_weight', 'team_weight', 'home_bias_factor', 'winning_streak_boost']:
                if param_name in current_params and param_name in self.param_bounds:
                    min_val, max_val = self.param_bounds[param_name]
                    current_val = current_params[param_name]
                    # Small random adjustment (±10% of range)
                    range_size = max_val - min_val
                    adjustment = random.uniform(-0.1 * range_size, 0.1 * range_size)
                    new_val = max(min_val, min(max_val, current_val + adjustment))
                    current_params[param_name] = new_val
            random.setstate(temp_state)
        
        # Evaluate starting parameters
        current_winnings = self._evaluate_parameters_worker(current_params, test_files)
        best_params = current_params.copy()
        best_winnings = current_winnings
        
        improvements = 0
        start_time = time.time()
        
        for iteration in range(iterations_per_worker):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > max_time_seconds:
                break
            
            # Generate neighbor (use global iteration count for reproducibility)
            global_iteration = worker_id * iterations_per_worker + iteration
            neighbor_params = self.random_neighbor(current_params, iteration=global_iteration)
            
            # Evaluate neighbor
            neighbor_winnings = self._evaluate_parameters_worker(neighbor_params, test_files)
            
            # Accept improvement
            if neighbor_winnings > current_winnings:
                current_params = neighbor_params
                current_winnings = neighbor_winnings
                improvements += 1
                
                # Update best if this is the best so far
                if current_winnings > best_winnings:
                    best_params = current_params.copy()
                    best_winnings = current_winnings
            
            # Report progress every 500 iterations
            if (iteration + 1) % 500 == 0:
                progress_queue.put({
                    'worker_id': worker_id,
                    'iteration': iteration + 1,
                    'total_iterations': iterations_per_worker,
                    'best_winnings': best_winnings,
                    'improvements': improvements,
                    'elapsed_time': time.time() - start_time
                })
        
        # Send final result
        result_queue.put({
            'worker_id': worker_id,
            'best_params': best_params,
            'best_winnings': best_winnings,
            'improvements': improvements,
            'total_iterations': iteration + 1,
            'elapsed_time': time.time() - start_time
        })
    
    def _evaluate_parameters_worker(self, params, test_files):
        """Worker-specific parameter evaluation function with multi-objective scoring + weekly weight capping"""
        try:
            # Validate all weights (including new enhanced parameters)
            if not self.validate_all_weights(params):
                return -10000  # Invalid weight combination
            
            system = EnhancedSuperOptimizedBettingSystem(random_seed=42, verbose=False)
            system.params.update(params)
            
            total_winnings = 0
            total_cost = 0
            weeks_with_wins = 0  # Count weeks where we got 12+ correct
            total_weeks = 0
            weekly_contributions = []  # Track each week's contribution to prevent overfitting
            
            for test_data in test_files:
                filename = test_data['filename']
                odds = test_data['odds']
                teams = test_data['teams']
                result = test_data['result']
                penge_values = test_data['penge']
                
                if not result or not odds:
                    continue
                
                patterns = system.generate_optimized_patterns(odds, teams)
                total_weeks += 1
                
                # Track best performance for this week
                week_best_correct = 0
                week_winnings = 0
                week_cost = len(patterns)
                
                for pattern in patterns:
                    winnings = calculate_winnings(pattern, result, penge_values)
                    week_winnings += winnings
                    
                    # Count correct predictions for this pattern
                    correct_count = sum(1 for pred, actual in zip(pattern, result) if pred == actual)
                    week_best_correct = max(week_best_correct, correct_count)
                
                # 🎯 WEEKLY WEIGHT CAPPING: Limit influence of weeks with excessive winnings
                week_profit = week_winnings - week_cost
                
                # Cap weekly profit contribution to prevent overfitting on "easy" weeks
                max_weekly_profit = 10000  # Cap at 10k points per week max influence
                capped_week_profit = min(week_profit, max_weekly_profit) if week_profit > 0 else week_profit
                
                # Also cap weekly loss to prevent one bad week from dominating
                min_weekly_loss = -2000  # Don't let one week contribute more than 2k loss
                capped_week_profit = max(capped_week_profit, min_weekly_loss)
                
                # Track for balanced optimization
                weekly_contributions.append({
                    'filename': filename,
                    'original_profit': week_profit,
                    'capped_profit': capped_week_profit,
                    'patterns_count': len(patterns)
                })
                
                total_winnings += week_winnings
                total_cost += week_cost
                
                # Count this week as a win if best pattern got 12+ correct
                if week_best_correct >= 12:
                    weeks_with_wins += 1
            
            if total_cost == 0 or total_weeks == 0:
                return -10000
            
            # 🎯 NEW: Calculate metrics using CAPPED weekly contributions for balanced optimization
            capped_total_profit = sum(week['capped_profit'] for week in weekly_contributions)
            net_profit = total_winnings - total_cost
            win_rate = weeks_with_wins / total_weeks
            
            # Multi-objective composite score using CAPPED profits
            # This prevents weeks with 50+ winning patterns from dominating optimization
            normalized_capped_profit = capped_total_profit / 1000.0
            
            # Composite score: balance between CAPPED winnings and win frequency
            composite_score = (self.winnings_weight * normalized_capped_profit + 
                             self.win_rate_weight * win_rate * 10000)  # Scale win_rate up
            
            # Additional diversity bonus: reward systems that perform well across many weeks
            weeks_with_positive_capped_profit = sum(1 for week in weekly_contributions if week['capped_profit'] > 0)
            diversity_bonus = (weeks_with_positive_capped_profit / total_weeks) * 1000  # Up to 1000 bonus
            
            composite_score += diversity_bonus * 0.1  # 10% weight for diversity
            
            return composite_score
            
        except Exception as e:
            return -10000
    
    def parallel_hill_climbing_optimization(self, max_iterations=10000, max_time_minutes=60, num_processes=None):
        """Parallel hill climbing optimization using multiple processes"""
        if num_processes is None:
            num_processes = min(cpu_count(), 8)  # Limit to 8 processes max
        
        print(f"🚀 PARALLEL SMART OPTIMIZATION")
        print(f"   Max iterations: {max_iterations:,}")
        print(f"   Max time: {max_time_minutes} minutes")
        print(f"   Processes: {num_processes}")
        print(f"   📊 Using {len(self.test_files)} complete data files")
        print("=" * 60)
        
        # Calculate iterations per worker
        iterations_per_worker = max_iterations // num_processes
        max_time_seconds = max_time_minutes * 60
        
        # Create queues for progress reporting and results
        progress_queue = Queue()
        result_queue = Queue()
        
        # Start worker processes
        processes = []
        start_time = time.time()
        
        for worker_id in range(num_processes):
            p = Process(
                target=self._worker_hill_climbing,
                args=(worker_id, self.best_params, iterations_per_worker, max_time_seconds, progress_queue, result_queue)
            )
            p.start()
            processes.append(p)
        
        print(f"✅ Started {num_processes} worker processes")
        print(f"   Each worker will run {iterations_per_worker:,} iterations")
        print()
        
        # Monitor progress
        completed_workers = 0
        worker_progress = {i: 0 for i in range(num_processes)}
        last_summary_time = time.time()
        
        while completed_workers < num_processes:
            try:
                # Check for progress updates (non-blocking)
                try:
                    progress = progress_queue.get(timeout=1.0)
                    worker_id = progress['worker_id']
                    iteration = progress['iteration']
                    total = progress['total_iterations']
                    winnings = progress['best_winnings']
                    improvements = progress['improvements']
                    elapsed = progress['elapsed_time']
                    
                    worker_progress[worker_id] = iteration
                    
                    # Calculate total progress across all workers
                    total_completed = sum(worker_progress.values())
                    total_target = max_iterations
                    progress_percent = (total_completed / total_target) * 100
                    
                    # Display progress update with total progress
                    print(f"🔄 Worker {worker_id+1}: {iteration:,}/{total:,} iterations, "
                          f"Best Score: {winnings:+.1f}, Improvements: {improvements}, "
                          f"Time: {elapsed:.1f}s")
                    print(f"   📊 TOTAL PROGRESS: {total_completed:,}/{total_target:,} iterations ({progress_percent:.1f}%)")
                    print()
                    
                except:
                    # No progress update available, continue
                    pass
                
                # Show periodic summary every 10 seconds
                current_time = time.time()
                if current_time - last_summary_time >= 10.0:
                    total_completed = sum(worker_progress.values())
                    total_target = max_iterations
                    progress_percent = (total_completed / total_target) * 100
                    elapsed_total = current_time - start_time
                    
                    # Estimate remaining time
                    if total_completed > 0:
                        rate = total_completed / elapsed_total
                        remaining_iterations = total_target - total_completed
                        eta_seconds = remaining_iterations / rate if rate > 0 else 0
                        eta_minutes = eta_seconds / 60
                    else:
                        eta_minutes = 0
                    
                    print(f"⏱️  SUMMARY: {total_completed:,}/{total_target:,} iterations ({progress_percent:.1f}%) - "
                          f"Elapsed: {elapsed_total:.0f}s, ETA: {eta_minutes:.1f}min")
                    
                    # Show individual worker status
                    for i in range(num_processes):
                        worker_percent = (worker_progress[i] / iterations_per_worker) * 100 if iterations_per_worker > 0 else 0
                        status = "✅ Done" if not processes[i].is_alive() else f"{worker_percent:.1f}%"
                        print(f"   Worker {i+1}: {worker_progress[i]:,}/{iterations_per_worker:,} ({status})")
                    print()
                    
                    last_summary_time = current_time
                
                # Check if any processes have finished
                alive_count = sum(1 for p in processes if p.is_alive())
                completed_workers = num_processes - alive_count
                
                # Check overall time limit
                if time.time() - start_time > max_time_seconds:
                    print(f"\n⏰ Overall time limit reached, terminating workers...")
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    break
                    
            except KeyboardInterrupt:
                print(f"\n⚠️ Interrupted by user, terminating workers...")
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                break
        
        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=5.0)  # Give them 5 seconds to finish gracefully
            if p.is_alive():
                p.terminate()
                p.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                results.append(result)
            except:
                break
        
        print("\n" + "=" * 60)
        print(f"✅ PARALLEL OPTIMIZATION COMPLETE")
        print(f"   Total time: {time.time() - start_time:.1f}s")
        print(f"   Workers completed: {len(results)}")
        
        if not results:
            print("❌ No results collected from workers")
            return self.best_params, self.best_winnings
        
        # Find best result across all workers
        best_result = max(results, key=lambda x: x['best_winnings'])
        best_params = best_result['best_params']
        best_winnings = best_result['best_winnings']
        
        # Get detailed metrics for the best parameters
        best_metrics = self.get_detailed_metrics(best_params)
        
        # Display results summary
        total_improvements = sum(r['improvements'] for r in results)
        total_iterations = sum(r['total_iterations'] for r in results)
        
        print(f"   Total iterations: {total_iterations:,}")
        print(f"   Total improvements: {total_improvements}")
        print(f"   Best composite score: {best_winnings:+.1f}")
        print(f"   Best worker: {best_result['worker_id'] + 1}")
        
        if best_metrics:
            print(f"\n📈 DETAILED METRICS FOR BEST PARAMETERS:")
            print(f"   💰 Net Profit: {best_metrics['net_profit']:+,.0f} points")
            print(f"   🎯 Win Rate: {best_metrics['win_rate']:.1%} ({best_metrics['weeks_with_wins']}/{best_metrics['total_weeks']} weeks with 12+ correct)")
            print(f"   📊 ROI: {best_metrics['roi']:+.1f}%")
            print(f"   💸 Total Cost: {best_metrics['total_cost']:,} bets")
            print(f"   🏆 Total Winnings: {best_metrics['total_winnings']:,} points")
        
        # Show all worker results
        print(f"\n📊 Worker Results (Composite Scores):")
        for i, result in enumerate(sorted(results, key=lambda x: x['worker_id'])):
            print(f"   Worker {result['worker_id']+1}: {result['best_winnings']:+8.1f} "
                  f"({result['improvements']} improvements, {result['total_iterations']:,} iterations)")
        
        if total_improvements > 0:
            print(f"\n🏆 Best parameters found:")
            print(f"     odds_weight: {best_params['odds_weight']:.3f}")
            print(f"     team_weight: {best_params['team_weight']:.3f}")
            print(f"     form_weight: {best_params['form_weight']:.3f}")
            print(f"     default_patterns: {best_params['default_patterns']} bets")
            print(f"     winning_streak_boost: {best_params['winning_streak_boost']:.3f}")
            print(f"     home_bias_factor: {best_params['home_bias_factor']:.3f}")
            
            # Ask user if they want to save parameters
            if best_metrics:
                save_prompt = f"\n💾 Save optimized parameters (Win Rate: {best_metrics['win_rate']:.1%}, Net Profit: {best_metrics['net_profit']:+,.0f}) to file? (y/n): "
            else:
                save_prompt = f"\n💾 Save optimized parameters (Score: {best_winnings:+.1f}) to file? (y/n): "
            
            save_params = input(save_prompt).strip().lower()
            if save_params == 'y':
                # Save with detailed metrics if available
                save_score = best_metrics['net_profit'] if best_metrics else best_winnings
                self.save_parameters_to_file(best_params, save_score, total_improvements, best_metrics)
        else:
            print(f"   📊 No improvements found - current parameters may already be optimal")
        
        return best_params, best_winnings
    
    def save_parameters_to_file(self, params, winnings, improvements, metrics=None):
        """Save optimized parameters to a JSON file with detailed metrics"""
        import datetime
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_parameters_{timestamp}.json"
        
        # Prepare data to save
        data = {
            "optimized_parameters": params,
            "winnings_amount": winnings,
            "improvements_found": improvements,
            "optimization_timestamp": datetime.datetime.now().isoformat(),
            "dataset_size": len(self.test_files),
            "optimization_weights": {
                "winnings_weight": self.winnings_weight,
                "win_rate_weight": self.win_rate_weight
            }
        }
        
        # Add detailed metrics if available
        if metrics:
            data["detailed_metrics"] = metrics
            data["description"] = f"Multi-objective optimizer: {metrics['win_rate']:.1%} win rate, {metrics['net_profit']:+,.0f} net profit from {improvements} improvements"
        else:
            data["description"] = f"Smart optimizer results with {winnings:+.1f} score from {improvements} improvements"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Parameters saved to: {filename}")
            if metrics:
                print(f"   🎯 Win Rate: {metrics['win_rate']:.1%}")
                print(f"   💰 Net Profit: {metrics['net_profit']:+,.0f}")
                print(f"   📊 ROI: {metrics['roi']:+.1f}%")
            else:
                print(f"   📊 Score: {winnings:+.1f}")
            print(f"   🔧 Load in code with: json.load(open('{filename}'))")
            
        except Exception as e:
            print(f"   ❌ Error saving parameters: {e}")
    
    def quick_test_current(self):
        """Quick test of current best parameters with detailed metrics"""
        print("🔍 Testing current parameters...")
        composite_score = self.evaluate_parameters(self.best_params)
        
        # Get detailed metrics
        metrics = self.get_detailed_metrics(self.best_params)
        if metrics:
            print(f"   💰 Net Profit: {metrics['net_profit']:+,.0f} points")
            print(f"   🎯 Win Rate: {metrics['win_rate']:.1%} ({metrics['weeks_with_wins']}/{metrics['total_weeks']} weeks with 12+ correct)")
            print(f"   📊 ROI: {metrics['roi']:+.1f}%")
            print(f"   🔢 Composite Score: {composite_score:+.1f} (Winnings: {self.winnings_weight:.1%}, Win Rate: {self.win_rate_weight:.1%})")
        
        return composite_score

    def load_latest_optimized_parameters(self):
        """Load the latest optimized parameters from file"""
        import glob
        
        # Find the latest optimized parameters file
        param_files = glob.glob("optimized_parameters_*.json")
        if not param_files:
            return None
        
        # Get the most recent file
        latest_file = max(param_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            params = data.get('optimized_parameters', {})
            # Handle backward compatibility - check for both new and old formats
            winnings = data.get('winnings_amount', data.get('roi_percentage', 0))
            
            print(f"   📂 Loading from: {latest_file}")
            print(f"   📊 Previous Winnings: {winnings:+.1f}")
            return params
            
        except Exception as e:
            print(f"   ❌ Error loading optimized parameters: {e}")
            return None


def main():
    optimizer = SmartOptimizer()
    
    print("SMART OPTIMIZER - MULTI-OBJECTIVE")
    print("Optimizes for both TOTAL WINNINGS and WIN FREQUENCY (weeks with 12+ correct)")
    print("=" * 75)
    
    # Quick test current system
    current_score = optimizer.quick_test_current()
    print(f"Current parameters Composite Score: {current_score:+.1f}")
    
    # Ask user what type of optimization they want
    print("\nOptimization Options:")
    print("1. Single-threaded optimization (original)")
    print("2. Multi-process optimization (FASTER!) 🚀")
    print("3. Custom multi-process optimization (specify workers & iterations)")
    print("4. Cancel")
    
    choice = input("\nChoose optimization type (1/2/3/4): ").strip()
    
    if choice == '1':
        print("\n🔧 Running single-threaded optimization...")
        best_params, best_winnings = optimizer.hill_climbing_optimization(max_iterations=10000, max_time_minutes=60)
    elif choice == '2':
        print("\n🚀 Running multi-process optimization...")
        best_params, best_winnings = optimizer.parallel_hill_climbing_optimization(max_iterations=10000, max_time_minutes=60)
    elif choice == '3':
        print("\n⚙️  Custom multi-process optimization...")
        try:
            max_iterations = int(input("Number of iterations (e.g., 20000): ").strip())
            num_workers = int(input("Number of worker processes (e.g., 12): ").strip())
            max_time = int(input("Max time in minutes (e.g., 120): ").strip())
            
            # Ask user about optimization balance
            print(f"\n🎯 Current optimization balance:")
            print(f"   Winnings weight: {optimizer.winnings_weight:.1%}")
            print(f"   Win rate weight: {optimizer.win_rate_weight:.1%}")
            
            change_balance = input("\nChange optimization balance? (y/n): ").strip().lower()
            if change_balance == 'y':
                print("\nChoose optimization focus:")
                print("1. Prioritize total winnings (60% winnings, 40% win rate)")
                print("2. Balanced approach (50% winnings, 50% win rate)")
                print("3. Prioritize win frequency (40% winnings, 60% win rate)")
                print("4. Custom weights")
                
                balance_choice = input("Choice (1/2/3/4): ").strip()
                
                if balance_choice == '1':
                    optimizer.winnings_weight = 0.6
                    optimizer.win_rate_weight = 0.4
                elif balance_choice == '2':
                    optimizer.winnings_weight = 0.5
                    optimizer.win_rate_weight = 0.5
                elif balance_choice == '3':
                    optimizer.winnings_weight = 0.4
                    optimizer.win_rate_weight = 0.6
                elif balance_choice == '4':
                    win_weight = float(input("Winnings weight (0.0-1.0): ").strip())
                    rate_weight = 1.0 - win_weight
                    optimizer.winnings_weight = win_weight
                    optimizer.win_rate_weight = rate_weight
                
                print(f"✅ Updated weights: Winnings {optimizer.winnings_weight:.1%}, Win Rate {optimizer.win_rate_weight:.1%}")
            
            print(f"\n🚀 Running optimization with {num_workers} workers, {max_iterations:,} iterations, {max_time} min timeout...")
            print(f"   Optimization focus: {optimizer.winnings_weight:.1%} winnings, {optimizer.win_rate_weight:.1%} win rate")
            best_params, best_winnings = optimizer.parallel_hill_climbing_optimization(
                max_iterations=max_iterations, 
                max_time_minutes=max_time, 
                num_processes=num_workers
            )
        except ValueError:
            print("❌ Invalid input. Please enter numbers only.")
        except KeyboardInterrupt:
            print("\n⚠️ Optimization cancelled by user.")
    else:
        print("Optimization cancelled.")


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    try:
        if hasattr(mp, 'set_start_method') and mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Start method already set
    
    main() 