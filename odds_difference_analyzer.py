#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ODDS DIFFERENCE ANALYZER: Demonstrate new odds difference analysis features
Shows how teams perform when equally matched vs as favorites/underdogs

Usage: python3 odds_difference_analyzer.py
"""

import json
import os
from super_optimized_system import EnhancedSuperOptimizedBettingSystem
from collections import defaultdict


def analyze_odds_difference_patterns():
    """Analyze and report odds difference patterns from historical data"""
    print("🔍 ODDS DIFFERENCE PATTERN ANALYSIS")
    print("=" * 60)
    print("Analyzing team performance based on historical odds differences...")
    print("🎯 Key Question: Do teams perform better when equally matched vs favorites/underdogs?")
    print()
    
    # Initialize the enhanced system to load historical data
    system = EnhancedSuperOptimizedBettingSystem(random_seed=42, verbose=False)
    
    print(f"📊 Loaded data for {len(system.odds_difference_patterns)} teams")
    print(f"📈 Historical matches analyzed: {len(system.historical_matches)}")
    print()
    
    # Collect statistics across all teams
    global_stats = {
        'tight_matches': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'moderate_matches': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'heavy_favorite': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'heavy_underdog': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0},
        'equal_strength': {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0}
    }
    
    team_examples = defaultdict(list)
    
    # Analyze each team's odds difference patterns
    for team_name, patterns in system.odds_difference_patterns.items():
        team_summary = system.get_team_odds_performance_summary(team_name)
        
        if team_summary:
            for scenario, stats in team_summary.items():
                if stats['matches'] >= 3:  # Only include teams with sufficient data
                    global_stats[scenario]['wins'] += stats['wins']
                    global_stats[scenario]['draws'] += stats['draws']
                    global_stats[scenario]['losses'] += stats['losses']
                    global_stats[scenario]['total'] += stats['matches']
                    
                    # Collect examples of teams that perform well in each scenario
                    if stats['success_rate'] > 0.7 and stats['matches'] >= 4:
                        team_examples[scenario].append({
                            'team': team_name,
                            'success_rate': stats['success_rate'],
                            'matches': stats['matches'],
                            'wins': stats['wins'],
                            'draws': stats['draws']
                        })
    
    # Display global statistics
    print("🌍 GLOBAL PERFORMANCE BY ODDS DIFFERENCE SCENARIO:")
    print("-" * 60)
    
    for scenario, stats in global_stats.items():
        if stats['total'] > 0:
            win_rate = stats['wins'] / stats['total']
            draw_rate = stats['draws'] / stats['total']
            success_rate = (stats['wins'] + stats['draws'] * 0.3) / stats['total']
            
            scenario_name = scenario.replace('_', ' ').title()
            print(f"📈 {scenario_name:20s}: {stats['total']:4d} matches")
            print(f"   Win Rate:     {win_rate:6.1%} ({stats['wins']:3d} wins)")
            print(f"   Draw Rate:    {draw_rate:6.1%} ({stats['draws']:3d} draws)")
            print(f"   Success Rate: {success_rate:6.1%} (wins + 0.3×draws)")
            print()
    
    # Display team examples
    print("🏆 TEAMS THAT EXCEL IN SPECIFIC SCENARIOS:")
    print("-" * 60)
    
    for scenario in ['equal_strength', 'tight_matches', 'heavy_favorite', 'heavy_underdog']:
        if scenario in team_examples and team_examples[scenario]:
            scenario_name = scenario.replace('_', ' ').title()
            print(f"🔥 Best at {scenario_name}:")
            
            # Sort by success rate and show top 5
            top_teams = sorted(team_examples[scenario], key=lambda x: x['success_rate'], reverse=True)[:5]
            
            for team_data in top_teams:
                print(f"   {team_data['team']:20s}: {team_data['success_rate']:5.1%} "
                      f"({team_data['wins']}W-{team_data['draws']}D in {team_data['matches']} matches)")
            print()
    
    # Analyze the key insight: Equal matches vs favorites/underdogs
    print("🎯 KEY INSIGHT ANALYSIS:")
    print("-" * 60)
    
    equal_performance = global_stats['equal_strength']
    favorite_performance = global_stats['heavy_favorite']
    underdog_performance = global_stats['heavy_underdog']
    
    if all(stats['total'] > 0 for stats in [equal_performance, favorite_performance, underdog_performance]):
        equal_success = (equal_performance['wins'] + equal_performance['draws'] * 0.3) / equal_performance['total']
        favorite_success = (favorite_performance['wins'] + favorite_performance['draws'] * 0.3) / favorite_performance['total']
        underdog_success = (underdog_performance['wins'] + underdog_performance['draws'] * 0.3) / underdog_performance['total']
        
        print(f"📊 Equal Strength Matches:  {equal_success:5.1%} success rate ({equal_performance['total']:3d} matches)")
        print(f"📊 Heavy Favorite Matches:  {favorite_success:5.1%} success rate ({favorite_performance['total']:3d} matches)")
        print(f"📊 Heavy Underdog Matches:  {underdog_success:5.1%} success rate ({underdog_performance['total']:3d} matches)")
        print()
        
        if equal_success > favorite_success and equal_success > underdog_success:
            print("✅ INSIGHT: Teams tend to perform MORE RELIABLY in EQUAL STRENGTH matches!")
            print("   This suggests the new odds difference analysis should boost equal match confidence.")
        elif favorite_success > equal_success and favorite_success > underdog_success:
            print("✅ INSIGHT: Teams perform BEST when they are HEAVY FAVORITES!")
            print("   This suggests boosting confidence when teams have strong favorite history.")
        elif underdog_success > equal_success and underdog_success > favorite_success:
            print("✅ INSIGHT: Teams perform SURPRISINGLY WELL as HEAVY UNDERDOGS!")
            print("   This suggests some teams thrive in underdog scenarios.")
        else:
            print("📊 INSIGHT: Performance varies significantly across odds difference scenarios.")
            print("   The new analysis should help identify team-specific patterns.")
    
    print()
    print("🔧 OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 60)
    print("1. 🎯 Focus on 'equal_match_boost' parameter - equal matches show consistent patterns")
    print("2. 🚀 Optimize 'odds_diff_threshold_tight' to better identify equal strength matches")
    print("3. ⚡ Test 'odds_diff_sensitivity' to find optimal pattern recognition level")
    print("4. 📈 Use 'odds_diff_window' to balance recent vs historical pattern data")
    print()
    print("Run 'python3 smart_optimizer.py' to optimize these new parameters!")


def demonstrate_team_specific_analysis():
    """Show detailed analysis for specific teams"""
    print("\n" + "=" * 60)
    print("🔍 TEAM-SPECIFIC ODDS DIFFERENCE ANALYSIS")
    print("=" * 60)
    
    system = EnhancedSuperOptimizedBettingSystem(random_seed=42, verbose=False)
    
    # Find teams with substantial data
    teams_with_data = []
    for team_name, patterns in system.odds_difference_patterns.items():
        total_matches = sum(len(results) for results in patterns.values())
        if total_matches >= 10:  # Teams with at least 10 historical matches
            teams_with_data.append((team_name, total_matches))
    
    # Sort by total matches and show top teams
    teams_with_data.sort(key=lambda x: x[1], reverse=True)
    top_teams = teams_with_data[:5]
    
    print(f"Detailed analysis for top 5 teams with most historical data:")
    print()
    
    for i, (team_name, total_matches) in enumerate(top_teams, 1):
        print(f"{i}. 🏟️  {team_name} ({total_matches} total matches)")
        
        summary = system.get_team_odds_performance_summary(team_name)
        if summary:
            for scenario, stats in summary.items():
                if stats['matches'] > 0:
                    scenario_name = scenario.replace('_', ' ').title()
                    print(f"   {scenario_name:18s}: {stats['success_rate']:5.1%} "
                          f"({stats['wins']}W-{stats['draws']}D-{stats['losses']}L)")
        print()


def simulate_odds_difference_impact():
    """Simulate how the new odds difference analysis affects predictions"""
    print("\n" + "=" * 60)
    print("🎮 SIMULATION: ODDS DIFFERENCE IMPACT ON PREDICTIONS")
    print("=" * 60)
    
    system = EnhancedSuperOptimizedBettingSystem(random_seed=42, verbose=False)
    
    # Load a recent test file for simulation
    test_files = [f for f in os.listdir("data") if f.startswith("test-2024") and f.endswith(".json")]
    if test_files:
        test_file = test_files[-1]  # Use most recent
        
        try:
            with open(f"data/{test_file}", 'r') as f:
                data = json.load(f)
            
            teams = data.get('teams', [])
            odds = data.get('odds', [])
            
            if len(teams) >= 3 and len(odds) >= 9:  # At least 3 matches
                print(f"📋 Simulating predictions for: {test_file}")
                print(f"🎯 Showing how odds difference analysis affects confidence...")
                print()
                
                for i in range(min(3, len(teams))):  # Show first 3 matches
                    team_pair = teams[i]
                    if isinstance(team_pair, dict) and '1' in team_pair and '2' in team_pair:
                        home_team = team_pair['1']
                        away_team = team_pair['2']
                        match_odds = odds[i*3:(i+1)*3] if len(odds) >= (i+1)*3 else [2.0, 3.0, 2.0]
                        
                        print(f"🥊 Match {i+1}: {home_team} vs {away_team}")
                        print(f"   Odds: Home {match_odds[0]:.2f}, Draw {match_odds[1]:.2f}, Away {match_odds[2]:.2f}")
                        
                        # Calculate odds difference factors
                        home_factor = system.get_odds_difference_factor(home_team, away_team, match_odds, True)
                        away_factor = system.get_odds_difference_factor(away_team, home_team, match_odds, False)
                        
                        # Determine match type
                        home_odds, away_odds = match_odds[0], match_odds[2]
                        odds_ratio = home_odds / away_odds
                        
                        if abs(odds_ratio - 1.0) <= 0.3:
                            match_type = "🟰 Equal Strength"
                        elif home_odds < away_odds:
                            match_type = "🏠 Home Favored"
                        else:
                            match_type = "✈️  Away Favored"
                        
                        print(f"   Type: {match_type}")
                        print(f"   Odds Diff Factor - Home: {home_factor:.3f}, Away: {away_factor:.3f}")
                        
                        if home_factor > 0.6:
                            print(f"   💡 {home_team} has strong history in this odds scenario!")
                        if away_factor > 0.6:
                            print(f"   💡 {away_team} has strong history in this odds scenario!")
                        print()
        
        except Exception as e:
            print(f"❌ Error loading test file: {e}")


if __name__ == "__main__":
    analyze_odds_difference_patterns()
    demonstrate_team_specific_analysis()
    simulate_odds_difference_impact()
    
    print("\n" + "🎉" * 20)
    print("🚀 ODDS DIFFERENCE ANALYSIS COMPLETE!")
    print("🎯 The algorithm now considers how teams perform in equal vs unequal matches")
    print("⚡ Run the smart optimizer to find optimal parameters for these new features")
    print("🏆 This should improve prediction accuracy by understanding team-specific patterns")
    print("🎉" * 20) 