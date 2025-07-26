#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PARAMETER TESTER: Systematic Real Performance Optimization
Tests different parameter combinations against real historical data
ONLY optimizes for actual ROI performance, not theoretical numbers
"""

import json
import os
import sys
import itertools
import time
from comprehensive_winnings_test import SuperOptimizedBettingSystem, calculate_winnings, load_test_file


class ParameterTester:
    def __init__(self):
        self.base_params = {
            # Fixed parameters (don't optimize)
            'default_patterns': 60,
            'high_confidence_threshold': 0.65,
            'medium_confidence_threshold': 0.55,
            'max_confidence': 0.95,
            'form_win_weight': 3.0,
            'form_draw_weight': 1.0,
            'form_loss_weight': 0.0,
            'form_window': 5,
            'home_bias_min_odds': 1.3,
            'strong_form_threshold': 0.7,
        }
        
        # Parameters to test (10 well-distributed values each for efficient optimization)
        self.test_ranges = {
            'odds_weight': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            'team_weight': [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],  
            'form_weight': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
            'home_bias_max_odds': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            'home_bias_factor': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05],
            'winning_streak_boost': [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50],
            'losing_streak_penalty': [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10],
            'streak_length': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'home_form_boost': [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40],
            'away_form_boost': [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40]
        }
        
        self.results = []
        
    def validate_weights(self, odds_weight, team_weight, form_weight):
        """Ensure weights sum to approximately 1.0"""
        total = odds_weight + team_weight + form_weight
        return 0.95 <= total <= 1.05
        
    def test_parameter_combination(self, params):
        """Test a specific parameter combination"""
        print(f"Testing: odds={params['odds_weight']:.2f}, team={params['team_weight']:.2f}, "
              f"form={params['form_weight']:.2f}, home_bias_factor={params['home_bias_factor']:.2f}")
        
        # Create a modified system with these parameters
        system = SuperOptimizedBettingSystem(random_seed=42)
        system.params.update(params)
        
        # Test against all files
        test_files = []
        data_files = [f for f in os.listdir("data") if f.endswith('.json')]
        
        for filename in data_files:
            test_data = load_test_file(filename)
            if test_data:
                test_files.append(test_data)
        
        if not test_files:
            return None
            
        total_winnings = 0
        total_cost = 0
        winning_files = 0
        
        for test_data in test_files:
            filename = test_data['filename']
            actual_result = test_data['result']
            penge_values = test_data['penge']
            
            # Generate patterns for this file
            analysis = system.analyze_week(filename)
            
            if 'error' in analysis:
                continue
                
            patterns = analysis['patterns']
            
            # Calculate winnings
            file_winnings = 0
            file_cost = len(patterns)
            
            for pattern in patterns:
                winnings = calculate_winnings(pattern, actual_result, penge_values)
                file_winnings += winnings
            
            file_profit = file_winnings - file_cost
            total_winnings += file_winnings
            total_cost += file_cost
            
            if file_profit > 0:
                winning_files += 1
        
        if total_cost == 0:
            return None
            
        net_profit = total_winnings - total_cost
        roi = (net_profit / total_cost) * 100
        win_rate = (winning_files / len(test_files)) * 100
        
        result = {
            'params': params.copy(),
            'files_tested': len(test_files),
            'winning_files': winning_files,
            'win_rate': win_rate,
            'total_winnings': total_winnings,
            'total_cost': total_cost,
            'net_profit': net_profit,
            'roi': roi
        }
        
        print(f"  → ROI: {roi:+.1f}%, Win Rate: {win_rate:.1f}%, Net Profit: {net_profit:,.0f}")
        
        return result
    
    def run_optimization(self, max_combinations=50):
        """Run parameter optimization with limited combinations"""
        print("🚀 PARAMETER OPTIMIZATION - REAL PERFORMANCE TESTING")
        print("=" * 70)
        print(f"Testing up to {max_combinations} parameter combinations...")
        print()
        
        # Generate all possible combinations
        keys = list(self.test_ranges.keys())
        values = list(self.test_ranges.values())
        all_combinations = list(itertools.product(*values))
        
        print(f"Total possible combinations: {len(all_combinations)}")
        
        # Limit combinations for reasonable runtime
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)
            combinations = random.sample(all_combinations, max_combinations)
            print(f"Testing random sample of {max_combinations} combinations")
        else:
            combinations = all_combinations
            print(f"Testing all {len(combinations)} combinations")
        
        print()
        
        valid_results = []
        start_time = time.time()
        
        for i, combination in enumerate(combinations, 1):
            params = dict(zip(keys, combination))
            params.update(self.base_params)
            
            # Validate weights sum to ~1.0
            if not self.validate_weights(params['odds_weight'], params['team_weight'], params['form_weight']):
                continue
            
            # Progress bar and timing info
            elapsed = time.time() - start_time
            avg_time_per_test = elapsed / i if i > 0 else 0
            remaining_tests = len(combinations) - i
            estimated_remaining = avg_time_per_test * remaining_tests
            
            progress = i / len(combinations)
            bar_length = 30
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"\n[{i:2d}/{len(combinations)}] {bar} {progress*100:.1f}%")
            print(f"Elapsed: {elapsed:.1f}s | Est. remaining: {estimated_remaining:.1f}s")
            sys.stdout.flush()  # Force output to display immediately
            
            result = self.test_parameter_combination(params)
            
            if result:
                valid_results.append(result)
                # Show best ROI so far
                best_roi = max(r['roi'] for r in valid_results)
                print(f"  → Best ROI so far: {best_roi:+.1f}%")
                
        print()
        print("=" * 70)
        print(f"✅ Completed testing {len(valid_results)} valid combinations")
        
        if not valid_results:
            print("❌ No valid results found!")
            return None
            
        # Sort by ROI
        valid_results.sort(key=lambda x: x['roi'], reverse=True)
        
        print()
        print("🏆 TOP 10 PERFORMING PARAMETER COMBINATIONS:")
        print("-" * 70)
        
        for i, result in enumerate(valid_results[:10], 1):
            params = result['params']
            print(f"{i:2d}. ROI: {result['roi']:+7.1f}% | "
                  f"Profit: {result['net_profit']:8,.0f} | "
                  f"Win Rate: {result['win_rate']:4.1f}%")
            print(f"    odds={params['odds_weight']:.2f}, team={params['team_weight']:.2f}, "
                  f"form={params['form_weight']:.2f}")
            print(f"    home_bias: max_odds={params['home_bias_max_odds']:.1f}, "
                  f"factor={params['home_bias_factor']:.2f}")
            print(f"    streaks: boost={params['winning_streak_boost']:.2f}, "
                  f"penalty={params['losing_streak_penalty']:.2f}, length={params['streak_length']}")
            print()
        
        # Show current baseline for comparison
        print("📊 COMPARISON WITH CURRENT SYSTEM:")
        current_result = self.test_current_system()
        if current_result:
            print(f"Current: ROI: {current_result['roi']:+7.1f}% | "
                  f"Profit: {current_result['net_profit']:8,.0f} | "
                  f"Win Rate: {current_result['win_rate']:4.1f}%")
            
            best = valid_results[0]
            improvement = best['roi'] - current_result['roi']
            print(f"Best:    ROI: {best['roi']:+7.1f}% | "
                  f"Profit: {best['net_profit']:8,.0f} | "
                  f"Win Rate: {best['win_rate']:4.1f}%")
            print(f"Improvement: {improvement:+.1f} percentage points")
        
        return valid_results[0] if valid_results else None
    
    def test_current_system(self):
        """Test the current system for comparison"""
        system = SuperOptimizedBettingSystem(random_seed=42)
        
        # Test against all files
        test_files = []
        data_files = [f for f in os.listdir("data") if f.endswith('.json')]
        
        for filename in data_files:
            test_data = load_test_file(filename)
            if test_data:
                test_files.append(test_data)
        
        if not test_files:
            return None
            
        total_winnings = 0
        total_cost = 0
        winning_files = 0
        
        for test_data in test_files:
            filename = test_data['filename']
            actual_result = test_data['result']
            penge_values = test_data['penge']
            
            # Generate patterns for this file
            analysis = system.analyze_week(filename)
            
            if 'error' in analysis:
                continue
                
            patterns = analysis['patterns']
            
            # Calculate winnings
            file_winnings = 0
            file_cost = len(patterns)
            
            for pattern in patterns:
                winnings = calculate_winnings(pattern, actual_result, penge_values)
                file_winnings += winnings
            
            file_profit = file_winnings - file_cost
            total_winnings += file_winnings
            total_cost += file_cost
            
            if file_profit > 0:
                winning_files += 1
        
        if total_cost == 0:
            return None
            
        net_profit = total_winnings - total_cost
        roi = (net_profit / total_cost) * 100
        win_rate = (winning_files / len(test_files)) * 100
        
        return {
            'files_tested': len(test_files),
            'winning_files': winning_files,
            'win_rate': win_rate,
            'total_winnings': total_winnings,
            'total_cost': total_cost,
            'net_profit': net_profit,
            'roi': roi
        }
    
    def apply_best_parameters(self, best_result):
        """Apply the best parameters to the main system"""
        if not best_result:
            print("❌ No best result to apply!")
            return
            
        best_params = best_result['params']
        
        print()
        print("🔧 APPLYING BEST PARAMETERS TO SUPER_OPTIMIZED_SYSTEM.PY:")
        print("-" * 50)
        
        # Read the current file
        with open('super_optimized_system.py', 'r') as f:
            content = f.read()
        
        # Replace parameter values
        replacements = {
            "'odds_weight': 0.45": f"'odds_weight': {best_params['odds_weight']}",
            "'team_weight': 0.35": f"'team_weight': {best_params['team_weight']}",
            "'form_weight': 0.20": f"'form_weight': {best_params['form_weight']}",
            "'home_bias_max_odds': 2.5": f"'home_bias_max_odds': {best_params['home_bias_max_odds']}",
            "'home_bias_factor': 0.85": f"'home_bias_factor': {best_params['home_bias_factor']}",
            "'winning_streak_boost': 1.25": f"'winning_streak_boost': {best_params['winning_streak_boost']}",
            "'losing_streak_penalty': 0.75": f"'losing_streak_penalty': {best_params['losing_streak_penalty']}",
            "'streak_length': 3": f"'streak_length': {best_params['streak_length']}",
            "'home_form_boost': 1.1": f"'home_form_boost': {best_params['home_form_boost']}",
            "'away_form_boost': 1.1": f"'away_form_boost': {best_params['away_form_boost']}"
        }
        
        for old, new in replacements.items():
            if old in content:
                content = content.replace(old, new)
                print(f"  {old} → {new}")
        
        # Write back
        with open('super_optimized_system.py', 'w') as f:
            f.write(content)
        
        print()
        print(f"✅ Applied parameters with {best_result['roi']:+.1f}% ROI improvement!")
        print("Run 'python3 comprehensive_winnings_test.py' to verify.")


def main():
    """Run parameter optimization"""
    if len(sys.argv) > 1:
        max_combinations = int(sys.argv[1])
    else:
        max_combinations = 50  # Balanced default for 10-value ranges
    
    tester = ParameterTester()
    best_result = tester.run_optimization(max_combinations)
    
    if best_result:
        apply = input("\nApply best parameters to super_optimized_system.py? (y/n): ").lower()
        if apply == 'y':
            tester.apply_best_parameters(best_result)
    else:
        print("❌ No improvements found!")


if __name__ == "__main__":
    main() 