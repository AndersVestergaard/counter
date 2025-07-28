#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYZE SUPER OPTIMIZED SUGGESTIONS
Compares betting patterns against actual results to show performance
"""

import json
import os
import sys
from collections import defaultdict


def calculate_winnings(bet_pattern, actual_result, penge_values):
    """Calculate winnings for a betting pattern"""
    if len(bet_pattern) != len(actual_result):
        return 0
    
    correct_count = sum(1 for pred, actual in zip(bet_pattern, actual_result) if pred == actual)
    
    # Only 10+ correct predictions can win points
    if correct_count < 10:
        return 0
    
    return penge_values.get(str(correct_count), 0)


def load_suggestions_file(date):
    """Load super optimized suggestions for a specific date"""
    # Try different possible filenames
    possible_files = [
        f"super_optimized_suggestions_{date}.json",
        f"super_optimized_suggestions_data/{date}.json",
        f"super_optimized_suggestions_data/test-{date}.json",
        f"super_optimized_suggestions_{date.replace('-', '_')}.json"
    ]
    
    for filename in possible_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    return None


def load_test_file(date):
    """Load test file with actual results for a specific date"""
    # Try different possible filenames
    possible_files = [
        f"data/test-{date}.json",
        f"data/{date}.json",
        f"test-{date}.json"
    ]
    
    for filename in possible_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Validate required fields
                    if 'result' in data and 'penge' in data:
                        return data
                    else:
                        print(f"⚠️  {filename} missing required fields (result/penge)")
                        continue
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    return None


def generate_suggestions_if_needed(date):
    """Generate super optimized suggestions if they don't exist"""
    print(f"🔄 Generating super optimized suggestions for {date}...")
    
    # Check if test file exists
    test_file = None
    possible_test_files = [f"data/test-{date}.json", f"data/{date}.json"]
    
    for filename in possible_test_files:
        if os.path.exists(filename):
            test_file = filename
            break
    
    if not test_file:
        print(f"❌ No test file found for {date}")
        return False
    
    # Generate suggestions using super_optimized_system.py
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "super_optimized_system.py", test_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully generated suggestions for {date}")
            return True
        else:
            print(f"❌ Error generating suggestions: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout generating suggestions for {date}")
        return False
    except Exception as e:
        print(f"❌ Error running super_optimized_system.py: {e}")
        return False


def analyze_suggestions(date):
    """Analyze super optimized suggestions against actual results"""
    print(f"🔍 ANALYZING SUPER OPTIMIZED SUGGESTIONS FOR {date}")
    print("=" * 70)
    
    # Load suggestions
    suggestions_data = load_suggestions_file(date)
    if not suggestions_data:
        print(f"📁 No suggestions file found for {date}")
        if input("Generate suggestions now? (y/n): ").strip().lower() == 'y':
            if generate_suggestions_if_needed(date):
                suggestions_data = load_suggestions_file(date)
            else:
                print("❌ Failed to generate suggestions")
                return
        else:
            print("❌ Cannot analyze without suggestions")
            return
    
    # Load actual results
    test_data = load_test_file(date)
    if not test_data:
        print(f"❌ No test file found for {date}")
        return
    
    actual_result = test_data['result']
    penge_values = test_data['penge']
    betting_patterns = suggestions_data.get('bets', [])
    
    if not betting_patterns:
        print("❌ No betting patterns found in suggestions file")
        return
    
    print(f"📊 Analyzing {len(betting_patterns)} betting patterns")
    print(f"🎯 Actual result: {actual_result}")
    print(f"💰 Penge structure: {penge_values}")
    print()
    
    # Analyze each pattern
    results = []
    total_winnings = 0
    correct_counts = defaultdict(int)
    
    for i, pattern in enumerate(betting_patterns, 1):
        if len(pattern) != len(actual_result):
            print(f"⚠️  Pattern {i} has wrong length: {len(pattern)} vs {len(actual_result)}")
            continue
        
        # Count correct predictions
        correct_count = sum(1 for pred, actual in zip(pattern, actual_result) if pred == actual)
        correct_counts[correct_count] += 1
        
        # Calculate winnings
        winnings = calculate_winnings(pattern, actual_result, penge_values)
        total_winnings += winnings
        
        # Store result
        results.append({
            'pattern_num': i,
            'pattern': pattern,
            'correct_count': correct_count,
            'winnings': winnings
        })
    
    # Sort by performance (correct count, then winnings)
    results.sort(key=lambda x: (x['correct_count'], x['winnings']), reverse=True)
    
    # Display top performing patterns
    print("🏆 TOP PERFORMING PATTERNS:")
    print("-" * 70)
    print(f"{'#':<3} {'Pattern':<15} {'Correct':<8} {'Winnings':<10} {'Match Details'}")
    print("-" * 70)
    
    for i, result in enumerate(results[:20], 1):  # Show top 20
        pattern = result['pattern']
        correct = result['correct_count']
        winnings = result['winnings']
        pattern_num = result['pattern_num']
        
        # Show match-by-match comparison for top patterns
        if i <= 5:
            match_details = ""
            for p, a in zip(pattern, actual_result):
                if p == a:
                    match_details += "✓"
                else:
                    match_details += "✗"
        else:
            match_details = f"{correct}/13 correct"
        
        status = "🟢" if winnings > 0 else "🔴"
        print(f"{i:<3} {pattern:<15} {correct:<8} {winnings:<10,} {match_details}")
        
        if i == 5 and len(results) > 5:
            print("-" * 70)
    
    # Summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print("-" * 40)
    
    total_cost = len(betting_patterns)  # Each pattern costs 1 point
    net_profit = total_winnings - total_cost
    roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
    
    winning_patterns = sum(1 for r in results if r['winnings'] > 0)
    win_rate = (winning_patterns / len(results) * 100) if results else 0
    
    print(f"Total patterns analyzed:    {len(betting_patterns):,}")
    print(f"Winning patterns:          {winning_patterns:,} ({win_rate:.1f}%)")
    print(f"Total winnings:            {total_winnings:,} points")
    print(f"Total cost:                {total_cost:,} points")
    print(f"Net profit:                {net_profit:+,} points")
    print(f"ROI:                       {roi:+.1f}%")
    
    # Accuracy distribution
    print(f"\n🎯 ACCURACY DISTRIBUTION:")
    print("-" * 30)
    for correct in sorted(correct_counts.keys(), reverse=True):
        count = correct_counts[correct]
        percentage = (count / len(results) * 100) if results else 0
        points = penge_values.get(str(correct), 0) if correct >= 10 else 0
        status = "💰" if points > 0 else "🔴"
        print(f"{correct:2d}/13 correct: {count:3d} patterns ({percentage:5.1f}%) {status} {points:,} pts each")
    
    # Best single pattern details
    if results:
        best = results[0]
        print(f"\n🏆 BEST PATTERN:")
        print(f"   Pattern #{best['pattern_num']}: {best['pattern']}")
        print(f"   Correct predictions: {best['correct_count']}/13")
        print(f"   Winnings: {best['winnings']:,} points")
        
        print(f"\n🔍 MATCH-BY-MATCH BREAKDOWN (Best Pattern):")
        print("   Match | Predicted | Actual | Result")
        print("   ------|-----------|--------|--------")
        for i, (pred, actual) in enumerate(zip(best['pattern'], actual_result), 1):
            result_symbol = "✓" if pred == actual else "✗"
            pred_name = {"0": "Home", "1": "Draw", "2": "Away"}[pred]
            actual_name = {"0": "Home", "1": "Draw", "2": "Away"}[actual]
            print(f"   {i:5d} | {pred_name:9s} | {actual_name:6s} | {result_symbol}")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        date = sys.argv[1]
        # Remove .json extension if provided
        if date.endswith('.json'):
            date = date[:-5]
        # Remove test- prefix if provided
        if date.startswith('test-'):
            date = date[5:]
    else:
        # Show available dates
        print("📅 Available dates:")
        test_files = [f for f in os.listdir("data") if f.startswith("test-") and f.endswith(".json")]
        test_files.sort()
        
        recent_files = test_files[-10:]  # Show last 10
        for i, filename in enumerate(recent_files, 1):
            date_part = filename[5:-5]  # Remove "test-" and ".json"
            print(f"  {i:2d}. {date_part}")
        
        choice = input("\nEnter date (YYYY-MM-DD) or number: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(recent_files):
            filename = recent_files[int(choice) - 1]
            date = filename[5:-5]  # Remove "test-" and ".json"
        else:
            date = choice
    
    analyze_suggestions(date)


if __name__ == "__main__":
    main() 