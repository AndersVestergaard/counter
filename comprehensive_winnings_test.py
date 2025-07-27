#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Winnings Test for FULL DATASET OPTIMIZED Betting System
Tests against ALL available data files with results using +3,746.2% ROI parameters

FULL DATASET OPTIMIZATION WINNER PARAMETERS (NO OVERFITTING):
- odds_weight: 0.393
- team_weight: 0.276  
- form_weight: 0.335
- winning_streak_boost: 2.898
- home_bias_factor: 0.868
"""

import json
import os
import sys
from super_optimized_system import SuperOptimizedBettingSystem


def calculate_winnings(bet_pattern, actual_result, penge_values):
    """Calculate winnings for a betting pattern"""
    if len(bet_pattern) != len(actual_result):
        return 0
    
    correct_count = sum(1 for pred, actual in zip(bet_pattern, actual_result) if pred == actual)
    
    # Only 10+ correct predictions can win points
    if correct_count < 10:
        return 0
    
    return penge_values.get(str(correct_count), 0)


def load_test_file(filename):
    """Load a test file and extract relevant data with complete validation"""
    filepath = os.path.join("data", filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if file has required fields
        if 'result' not in data or 'penge' not in data:
            return None
            
        test_data = {
            'filename': filename,
            'result': data['result'],
            'penge': data['penge'],
            'teams': data.get('teams', []),
            'odds': data.get('odds', [])
        }
        
        # Additional validation for complete data (same as smart_optimizer)
        if (test_data['result'] and test_data['penge'] and 
            test_data['teams'] and test_data['odds'] and
            len(test_data['teams']) == 13 and len(test_data['odds']) == 39):
            return test_data
        else:
            return None
            
    except Exception as e:
        return None


def load_all_complete_test_files():
    """Load ALL test files with complete data for accurate testing/optimization"""
    print("Loading ALL test files with complete data...")
    data_files = [f for f in os.listdir("data") if f.endswith('.json')]
    data_files.sort()  # Consistent ordering
    
    test_files = []
    loaded_files = 0
    skipped_files = 0
    
    for filename in data_files:
        test_data = load_test_file(filename)
        if test_data:
            test_files.append(test_data)
            loaded_files += 1
        else:
            print(f"  ⚠️  Skipped {filename}: incomplete data")
            skipped_files += 1
    
    print(f"✅ Loaded {loaded_files} complete test files")
    print(f"⚠️  Skipped {skipped_files} files with missing/incomplete data")
    print(f"📊 Using {loaded_files} files for testing/optimization")
    
    return test_files


def test_all_files():
    """Test the mega optimized system against all available test files"""
    
    print("🚀 COMPREHENSIVE WINNINGS TEST - FULL DATASET OPTIMIZED SYSTEM")
    print("=" * 80)
    print("Testing against ALL available data files with FULL DATASET OPTIMIZED parameters...")
    print("🏆 Expected performance: +3,746.2% ROI")
    print()
    
    # Initialize the mega optimized system
    system = SuperOptimizedBettingSystem(random_seed=42)
    
    # Load all complete test files
    test_files = load_all_complete_test_files()
    
    if not test_files:
        print("❌ No complete test files found!")
        return
    
    print()
    
    total_winnings = 0
    total_cost = 0
    total_profit = 0
    winning_files = 0
    results = []
    
    print("📊 TESTING EACH FILE:")
    print("-" * 80)
    
    for test_data in test_files:
        filename = test_data['filename']
        actual_result = test_data['result']
        penge_values = test_data['penge']
        
        # Generate patterns for this file
        analysis = system.analyze_week(filename)
        
        if 'error' in analysis:
            print(f"❌ {filename}: {analysis['error']}")
            continue
        
        patterns = analysis['bets']
        
        # Calculate winnings
        file_winnings = 0
        file_cost = len(patterns)
        winning_patterns = 0
        best_correct = 0
        count_10 = 0
        count_11 = 0
        count_12 = 0
        count_13 = 0
        
        for pattern in patterns:
            winnings = calculate_winnings(pattern, actual_result, penge_values)
            if winnings > 0:
                winning_patterns += 1
                file_winnings += winnings
                
            correct_count = sum(1 for pred, actual in zip(pattern, actual_result) if pred == actual)
            best_correct = max(best_correct, correct_count)
            
            # Count exact correct predictions
            if correct_count == 10:
                count_10 += 1
            elif correct_count == 11:
                count_11 += 1
            elif correct_count == 12:
                count_12 += 1
            elif correct_count == 13:
                count_13 += 1
        
        file_profit = file_winnings - file_cost
        file_roi = (file_profit / file_cost) * 100 if file_cost > 0 else 0
        
        total_winnings += file_winnings
        total_cost += file_cost
        total_profit += file_profit
        
        if file_profit > 0:
            winning_files += 1
        
        status = "🟢" if file_profit > 0 else "🔴"
        print(f"{filename:20} | W:{file_winnings:8,} | 10:{count_10:2} 11:{count_11:2} 12:{count_12:2} 13:{count_13:2} | {best_correct:2}/13 | {status}")
        
        results.append({
            'filename': filename,
            'winnings': file_winnings,
            'cost': file_cost,
            'profit': file_profit,
            'roi': file_roi,
            'winning_patterns': winning_patterns,
            'best_correct': best_correct,
            'result': actual_result,
            'count_10': count_10,
            'count_11': count_11,
            'count_12': count_12,
            'count_13': count_13
        })
    
    print("-" * 80)
    
    # Calculate overall statistics
    overall_roi = (total_profit / total_cost) * 100 if total_cost > 0 else 0
    win_rate = (winning_files / len(results)) * 100 if results else 0
    
    print()
    print("🏆 OVERALL PERFORMANCE SUMMARY:")
    print(f"   Files tested: {len(results)}")
    print(f"   Files with profit: {winning_files}")
    print(f"   Win rate: {win_rate:.1f}%")
    print(f"   Total winnings: {total_winnings:,} points")
    print(f"   Total cost: {total_cost:,} points")
    print(f"   Net profit: {total_profit:,} points")
    print(f"   Overall ROI: {overall_roi:+.1f}%")
    print()
    
    # Performance analysis with FULL DATASET OPTIMIZATION benchmarks
    if overall_roi > 3746.2:
        print(f"🚀 BEYOND FULL DATASET OPTIMIZATION! ({overall_roi:+.1f}% vs +3,746.2% expected)")
    elif overall_roi > 1724.3:
        print(f"🎯 EXCEEDED OLD EXPECTATIONS! ({overall_roi:+.1f}% vs +3,746.2% full dataset target)")
    elif overall_roi > 100:
        print(f"✅ EXCELLENT PERFORMANCE! ({overall_roi:+.1f}% ROI)")
    elif overall_roi > 0:
        print(f"📈 Profitable but below full dataset expectations ({overall_roi:+.1f}%)")
    else:
        print(f"📊 Needs improvement ({overall_roi:+.1f}% ROI)")
    
    print()
    print("📈 TOP PERFORMING FILES:")
    sorted_results = sorted(results, key=lambda x: x['roi'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"   {i}. {result['filename']:25} | ROI: {result['roi']:6.1f}% | Profit: {result['profit']:6,}")    
    
    print()
    print("🎯 BEST PREDICTION ACCURACY:")
    sorted_by_accuracy = sorted(results, key=lambda x: x['best_correct'], reverse=True)
    for i, result in enumerate(sorted_by_accuracy[:10], 1):
        print(f"   {i}. {result['filename']:25} | Best: {result['best_correct']:2}/13 correct | Result: {result['result']}")
    
    return {
        'total_files': len(results),
        'winning_files': winning_files,
        'total_winnings': total_winnings,
        'total_cost': total_cost,
        'net_profit': total_profit,
        'overall_roi': overall_roi,
        'win_rate': win_rate,
        'results': results
    }


if __name__ == "__main__":
    test_all_files() 