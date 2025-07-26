#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMART OPTIMIZER: Avoids endless loops with intelligent sampling
Uses gradient-free optimization techniques to improve on +1724.3% ROI
"""

import json
import os
import sys
import time
import random
from comprehensive_winnings_test import SuperOptimizedBettingSystem, calculate_winnings, load_test_file, load_all_complete_test_files


class SmartOptimizer:
    def __init__(self):
        # Load optimized parameters from file if available
        optimized_params = self.load_latest_optimized_parameters()
        
        if optimized_params:
            self.best_params = optimized_params
            print(f"   ✅ Starting with optimized parameters from file")
        else:
            # Fallback to default parameters
            self.best_params = {
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
                'strong_form_threshold': 0.7
            }
            print(f"   ⚠️  No optimized parameters found, using defaults")
        
        # Parameter bounds for optimization (MASSIVELY EXTREME search for 100k iterations)
        self.param_bounds = {
            'odds_weight': (0.05, 4.0),
            'team_weight': (0.01, 2.0),
            'form_weight': (0.001, 2.0),
            'home_bias_max_odds': (0.8, 10.0),
            'home_bias_factor': (0.3, 2.0),
            'winning_streak_boost': (0.5, 3.0),
            'losing_streak_penalty': (0.1, 2.0),
            'streak_length': (1, 5),
            'home_form_boost': (0.3, 3.0),
            'away_form_boost': (0.3, 3.0)
        }
        
        self.best_roi = 0  # Starting ROI - will find the actual best through optimization
        self.test_files = []
        self.load_test_files()
        
    def load_test_files(self):
        """Load ALL test files with complete data for accurate optimization"""
        self.test_files = load_all_complete_test_files()
    
    def validate_weights(self, odds_w, team_w, form_w):
        """Ensure weights sum to approximately 1.0"""
        total = odds_w + team_w + form_w
        return 0.9 <= total <= 1.1
    
    def evaluate_parameters(self, params):
        """Evaluate a parameter set and return ROI"""
        try:
            # Validate weights
            if not self.validate_weights(params['odds_weight'], params['team_weight'], params['form_weight']):
                return -1000  # Invalid weight combination
            
            system = SuperOptimizedBettingSystem(random_seed=42, verbose=False)
            system.params.update(params)
            
            total_winnings = 0
            total_cost = 0
            
            for test_data in self.test_files:
                filename = test_data['filename']
                odds = test_data['odds']
                teams = test_data['teams']
                result = test_data['result']
                penge_values = test_data['penge']
                
                if not result or not odds:
                    continue
                
                patterns = system.generate_optimized_patterns(odds, teams)
                
                for pattern in patterns:
                    winnings = calculate_winnings(pattern, result, penge_values)
                    total_winnings += winnings
                    total_cost += 1
            
            if total_cost == 0:
                return -1000
            
            net_profit = total_winnings - total_cost
            roi = (net_profit / total_cost) * 100
            
            return roi
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return -1000
    
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
            if param in ['streak_length']:
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
    
    def hill_climbing_optimization(self, max_iterations=1000, max_time_minutes=15):
        """Hill climbing optimization with time limit"""
        print(f"🚀 SMART OPTIMIZATION")
        print(f"   Max iterations: {max_iterations:,}")
        print(f"   Max time: {max_time_minutes} minutes")
        print(f"   📊 Using {len(self.test_files)} complete data files")
        print("=" * 60)
        
        current_params = self.best_params.copy()
        current_roi = self.best_roi
        
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        
        improvements = 0
        
        # First, evaluate current parameters to establish baseline
        print("🔍 Evaluating starting parameters...")
        current_roi = self.evaluate_parameters(current_params)
        print(f"✅ Starting ROI: {current_roi:+.1f}%")
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
            neighbor_roi = self.evaluate_parameters(neighbor_params)
            
            # Only print if we found an improvement
            if neighbor_roi > current_roi:
                improvement = neighbor_roi - current_roi
                print(f"\n🎯 IMPROVEMENT {iteration+1:,}: {neighbor_roi:+.1f}% (+{improvement:.1f})")
                print(f"   📊 Parameters: odds={neighbor_params['odds_weight']:.3f}, team={neighbor_params['team_weight']:.3f}, form={neighbor_params['form_weight']:.3f}")
                print(f"                 streak_boost={neighbor_params['winning_streak_boost']:.3f}, bias_factor={neighbor_params['home_bias_factor']:.3f}")
                
                current_params = neighbor_params
                current_roi = neighbor_roi
                improvements += 1
        
        print("\n" + "=" * 60)
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"   Iterations: {iteration + 1}")
        print(f"   Improvements found: {improvements}")
        print(f"   Final ROI: {current_roi:+.1f}%")
        
        if improvements > 0:
            print(f"   🏆 Best parameters found:")
            print(f"     odds_weight: {current_params['odds_weight']:.3f}")
            print(f"     team_weight: {current_params['team_weight']:.3f}")
            print(f"     form_weight: {current_params['form_weight']:.3f}")
            print(f"     winning_streak_boost: {current_params['winning_streak_boost']:.3f}")
            print(f"     home_bias_factor: {current_params['home_bias_factor']:.3f}")
            
            # Ask user if they want to save parameters
            save_params = input(f"\n💾 Save optimized parameters (ROI: {current_roi:+.1f}%) to file? (y/n): ").strip().lower()
            if save_params == 'y':
                self.save_parameters_to_file(current_params, current_roi, improvements)
        else:
            print(f"   📊 No improvements found - current parameters may already be optimal")
        
        return current_params, current_roi
    
    def save_parameters_to_file(self, params, roi, improvements):
        """Save optimized parameters to a JSON file"""
        import datetime
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_parameters_{timestamp}.json"
        
        # Prepare data to save
        data = {
            "optimized_parameters": params,
            "roi_percentage": roi,
            "improvements_found": improvements,
            "optimization_timestamp": datetime.datetime.now().isoformat(),
            "dataset_size": len(self.test_files),
            "description": f"Smart optimizer results with {roi:+.1f}% ROI from {improvements} improvements"
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Parameters saved to: {filename}")
            print(f"   📊 ROI: {roi:+.1f}%")
            print(f"   🔧 Load in code with: json.load(open('{filename}'))")
            
        except Exception as e:
            print(f"   ❌ Error saving parameters: {e}")
    
    def quick_test_current(self):
        """Quick test of current best parameters"""
        print("🔍 Testing current parameters...")
        roi = self.evaluate_parameters(self.best_params)
        return roi

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
            roi = data.get('roi_percentage', 0)
            
            print(f"   📂 Loading from: {latest_file}")
            print(f"   📊 Previous ROI: {roi:+.1f}%")
            return params
            
        except Exception as e:
            print(f"   ❌ Error loading optimized parameters: {e}")
            return None


def main():
    optimizer = SmartOptimizer()
    
    print("SMART OPTIMIZER")
    print("=" * 60)
    
    # Quick test current system
    current_roi = optimizer.quick_test_current()
    print(f"Current parameters ROI: {current_roi:+.1f}%")
    
    # Ask user if they want to proceed
    proceed = input("\nStart optimization? (y/n): ").strip().lower()
    
    if proceed == 'y':
        # Run optimization with 10,000 iterations
        best_params, best_roi = optimizer.hill_climbing_optimization(max_iterations=10000, max_time_minutes=60)
        
    else:
        print("Optimization cancelled.")


if __name__ == "__main__":
    main() 