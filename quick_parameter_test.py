#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK PARAMETER TEST: Debug version to see where the process hangs
"""

import json
import os
import sys
import time
from comprehensive_winnings_test import SuperOptimizedBettingSystem, calculate_winnings, load_test_file


def simple_test():
    print("🔍 DEBUGGING PARAMETER OPTIMIZATION")
    print("=" * 50)
    
    print("Step 1: Loading test files...")
    sys.stdout.flush()
    
    # Test file loading
    test_files = []
    data_files = [f for f in os.listdir("data") if f.endswith('.json')]
    
    for filename in data_files[:5]:  # Only test first 5 files
        test_data = load_test_file(filename)
        if test_data:
            test_files.append(test_data)
    
    print(f"✅ Loaded {len(test_files)} test files")
    sys.stdout.flush()
    
    print("Step 2: Creating betting system...")
    sys.stdout.flush()
    
    # Test system creation
    system = SuperOptimizedBettingSystem(random_seed=42)
    print("✅ System created successfully")
    sys.stdout.flush()
    
    print("Step 3: Testing parameter combination...")
    sys.stdout.flush()
    
    # Test one parameter set
    test_params = {
        'odds_weight': 0.45,
        'team_weight': 0.35,
        'form_weight': 0.20,
        'home_bias_max_odds': 2.5,
        'home_bias_factor': 0.85,
        'winning_streak_boost': 1.15,
        'losing_streak_penalty': 0.85,
        'streak_length': 3,
        'home_form_boost': 1.1,
        'away_form_boost': 1.1
    }
    
    system.params.update(test_params)
    print("✅ Parameters updated")
    sys.stdout.flush()
    
    total_winnings = 0
    total_cost = 0
    
    print("Step 4: Testing against files...")
    sys.stdout.flush()
    
    for i, test_data in enumerate(test_files, 1):
        print(f"  Testing file {i}/{len(test_files)}: {test_data['filename']}")
        sys.stdout.flush()
        
        filename = test_data['filename']
        actual_result = test_data['result']
        penge_values = test_data['penge']
        
        # Generate patterns for this file
        analysis = system.analyze_week(filename)
        
        if 'error' in analysis:
            print(f"    ❌ Error: {analysis['error']}")
            continue
            
        patterns = analysis['patterns']
        print(f"    ✅ Generated {len(patterns)} patterns")
        sys.stdout.flush()
        
        # Calculate winnings
        file_winnings = 0
        file_cost = len(patterns)
        
        for pattern in patterns:
            winnings = calculate_winnings(pattern, actual_result, penge_values)
            file_winnings += winnings
        
        file_profit = file_winnings - file_cost
        total_winnings += file_winnings
        total_cost += file_cost
        
        print(f"    💰 Winnings: {file_winnings}, Cost: {file_cost}, Profit: {file_profit}")
        sys.stdout.flush()
    
    if total_cost > 0:
        net_profit = total_winnings - total_cost
        roi = (net_profit / total_cost) * 100
        
        print()
        print("🏆 FINAL RESULTS:")
        print(f"Total winnings: {total_winnings:,}")
        print(f"Total cost: {total_cost:,}")
        print(f"Net profit: {net_profit:,}")
        print(f"ROI: {roi:+.1f}%")
    else:
        print("❌ No valid results")


if __name__ == "__main__":
    simple_test() 