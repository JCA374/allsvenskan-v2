#!/usr/bin/env python3
"""
Allsvenskan Monte Carlo Forecast - Command Line Interface

This CLI provides command-line access to all the main features of the Allsvenskan forecasting system.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Import custom modules
from core.data.scraper import AllsvenskanScraper
from core.data.cleaner import DataCleaner
from core.data.strength import TeamStrengthCalculator
from core.models.poisson_model import PoissonModel
from core.simulation.simulator import MonteCarloSimulator
from core.analysis.aggregator import ResultsAggregator

def setup_directories():
    """Ensure required directories exist"""
    dirs = [
        'data/raw',
        'data/clean',
        'data/processed',
        'data/db',
        'reports/simulations',
        'models'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def scrape_data(seasons=None):
    """Scrape Allsvenskan match data from football-data.co.uk"""
    print("⚽ Starting data scraping...")

    if seasons is None:
        current_year = datetime.now().year
        # 5 seasons gives enough data for stable home-advantage and league-avg
        # estimation; time-decay in the model ensures old matches don't dominate.
        seasons = list(range(current_year - 4, current_year + 1))

    scraper = AllsvenskanScraper()
    raw_data = scraper.scrape_matches(seasons=seasons)

    if raw_data is not None and len(raw_data) > 0:
        output_file = f'data/raw/allsvenskan_matches_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        raw_data.to_csv(output_file, index=False)
        print(f"✅ Scraped {len(raw_data)} matches")
        print(f"📁 Saved to: {output_file}")
        return raw_data
    else:
        print("❌ No data scraped")
        return None


def clean_data(input_file=None):
    """Clean and separate results from fixtures"""
    print("🧹 Cleaning data...")

    if input_file is None:
        # Find most recent raw data file
        raw_files = list(Path('data/raw').glob('allsvenskan_matches_*.csv'))
        if not raw_files:
            print("❌ No raw data files found. Run 'scrape' first.")
            return None, None
        input_file = str(max(raw_files, key=os.path.getctime))

    raw_data = pd.read_csv(input_file)
    print(f"📁 Loading: {input_file}")

    cleaner = DataCleaner()
    results, fixtures = cleaner.clean_data(raw_data)

    results.to_csv('data/clean/results.csv', index=False)
    fixtures.to_csv('data/clean/fixtures.csv', index=False)

    print(f"✅ Cleaned {len(results)} completed matches")
    print(f"✅ Found {len(fixtures)} upcoming fixtures")
    print(f"📁 Saved to: data/clean/results.csv and data/clean/fixtures.csv")

    return results, fixtures


def train_model(advanced=False):
    """Train Poisson model"""
    print("🎯 Training Poisson model...")

    # Load results
    results_file = 'data/clean/results.csv'
    if not Path(results_file).exists():
        print("❌ Results file not found. Run 'clean' first.")
        return None

    results = pd.read_csv(results_file)

    # Calculate team strengths
    print("📊 Calculating team strengths...")
    strength_calc = TeamStrengthCalculator()
    team_stats = strength_calc.calculate_strengths(results)
    team_stats.to_csv('data/processed/team_stats.csv')
    print(f"✅ Team statistics calculated for {len(team_stats)} teams")

    # Train model
    print(f"🤖 Training model (advanced={advanced})...")
    # Use MLE and Dixon-Coles if advanced training requested
    model = PoissonModel(use_mle=advanced, use_dixon_coles=advanced)
    model.fit(results, team_stats)

    # Save model
    model_file = 'models/poisson_params.pkl'
    model.save(model_file)

    print(f"✅ Model trained successfully")
    print(f"📁 Saved to: {model_file}")

    return model, team_stats


def run_simulation(n_simulations=10000):
    """Run Monte Carlo simulation"""
    print(f"🎲 Running Monte Carlo simulation ({n_simulations:,} iterations)...")

    # Load model
    model_file = 'models/poisson_params.pkl'
    if not Path(model_file).exists():
        print("❌ Model file not found. Run 'train' first.")
        return None

    model = PoissonModel()
    model.load(model_file)

    # Load fixtures
    fixtures_file = 'data/clean/fixtures.csv'
    if not Path(fixtures_file).exists():
        print("❌ Fixtures file not found. Run 'clean' first.")
        return None

    fixtures = pd.read_csv(fixtures_file)

    # Run simulation
    simulator = MonteCarloSimulator(fixtures, model)
    results = simulator.run(
        n_simulations=n_simulations,
        progress_callback=lambda progress: print(f"Progress: {progress:.1f}%", end='\r')
    )

    print()  # New line after progress

    # Save results
    output_file = f'reports/simulations/sim_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results.to_csv(output_file, index=False)

    print(f"✅ Simulation complete")
    print(f"📁 Saved to: {output_file}")

    return results


def analyze_results(input_file=None):
    """Analyze simulation results"""
    print("📈 Analyzing results...")

    if input_file is None:
        # Find most recent simulation file
        sim_files = list(Path('reports/simulations').glob('sim_results_*.csv'))
        if not sim_files:
            print("❌ No simulation files found. Run 'simulate' first.")
            return None
        input_file = str(max(sim_files, key=os.path.getctime))

    sim_results = pd.read_csv(input_file)
    print(f"📁 Loading: {input_file}")

    aggregator = ResultsAggregator()

    # Get various analyses
    basic_analysis = aggregator.analyze_results(sim_results)
    championship_odds = aggregator.calculate_championship_odds(sim_results)
    relegation_odds = aggregator.calculate_relegation_odds(sim_results)
    expected_points = aggregator.calculate_expected_points(sim_results)

    # Combine into single analysis dict
    analysis = {
        'basic_analysis': basic_analysis,
        'championship_odds': championship_odds,
        'relegation_odds': relegation_odds,
        'expected_points': expected_points
    }

    # Display key insights
    print("\n" + "="*60)
    print("🏆 CHAMPIONSHIP PROBABILITIES")
    print("="*60)
    if championship_odds:
        sorted_champ = sorted(championship_odds.items(), key=lambda x: x[1], reverse=True)
        for team, prob in sorted_champ[:10]:
            print(f"{team:20s}: {prob:6.2%}")

    print("\n" + "="*60)
    print("⬇️  RELEGATION PROBABILITIES")
    print("="*60)
    if relegation_odds:
        sorted_releg = sorted(relegation_odds.items(), key=lambda x: x[1], reverse=True)
        for team, prob in sorted_releg[:10]:
            print(f"{team:20s}: {prob:6.2%}")

    print("\n" + "="*60)
    print("📊 EXPECTED FINAL STANDINGS (by points)")
    print("="*60)
    if expected_points:
        sorted_points = sorted(expected_points.items(), key=lambda x: x[1], reverse=True)
        for rank, (team, points) in enumerate(sorted_points, 1):
            print(f"{rank:2d}. {team:20s}: {points:5.1f} pts")

    # Save analysis
    output_file = f'reports/simulations/analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        json_analysis = {k: v.to_dict() if hasattr(v, 'to_dict') else v
                        for k, v in analysis.items()}
        json.dump(json_analysis, f, indent=2)

    print(f"\n📁 Full analysis saved to: {output_file}")

    return analysis


def predict_fixtures():
    """Generate predictions for upcoming fixtures"""
    print("🔮 Generating fixture predictions...")

    # Load model
    model_file = 'models/poisson_params.pkl'
    if not Path(model_file).exists():
        print("❌ Model file not found. Run 'train' first.")
        return None

    model = PoissonModel()
    model.load(model_file)

    # Load fixtures
    fixtures_file = 'data/clean/fixtures.csv'
    if not Path(fixtures_file).exists():
        print("❌ Fixtures file not found. Run 'clean' first.")
        return None

    fixtures = pd.read_csv(fixtures_file)

    # Generate predictions
    predictions = []
    for _, match in fixtures.iterrows():
        pred = model.predict_outcome_probabilities(match['HomeTeam'], match['AwayTeam'])
        predictions.append({
            'Date': match.get('Date', 'TBD'),
            'HomeTeam': match['HomeTeam'],
            'AwayTeam': match['AwayTeam'],
            'Home_Win_Prob': pred['home_win'],
            'Draw_Prob': pred['draw'],
            'Away_Win_Prob': pred['away_win'],
            'Expected_Home_Goals': pred['mu_home'],
            'Expected_Away_Goals': pred['mu_away']
        })

    predictions_df = pd.DataFrame(predictions)

    # Display predictions
    print("\n" + "="*80)
    print("🔮 UPCOMING FIXTURE PREDICTIONS")
    print("="*80)
    for _, pred in predictions_df.iterrows():
        print(f"\n{pred['Date']}")
        print(f"{pred['HomeTeam']} vs {pred['AwayTeam']}")
        print(f"  Expected Score: {pred['Expected_Home_Goals']:.1f} - {pred['Expected_Away_Goals']:.1f}")
        print(f"  Win Probabilities: {pred['Home_Win_Prob']:.1%} / {pred['Draw_Prob']:.1%} / {pred['Away_Win_Prob']:.1%}")

    # Save predictions
    output_file = 'reports/simulations/fixture_predictions.csv'
    predictions_df.to_csv(output_file, index=False)
    print(f"\n📁 Predictions saved to: {output_file}")

    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description='Allsvenskan Monte Carlo Forecast - CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scrape                    # Scrape latest match data
  %(prog)s clean                     # Clean scraped data
  %(prog)s train                     # Train basic model
  %(prog)s train --advanced          # Train advanced model (slower, more accurate)
  %(prog)s simulate --iterations 50000  # Run 50,000 Monte Carlo simulations
  %(prog)s analyze                   # Analyze simulation results
  %(prog)s predict                   # Generate fixture predictions
  %(prog)s pipeline                  # Run full pipeline (scrape -> clean -> train -> simulate -> analyze)
        """
    )

    parser.add_argument('command',
                       choices=['scrape', 'clean', 'train', 'simulate', 'analyze', 'predict', 'pipeline'],
                       help='Command to execute')
    parser.add_argument('--seasons', nargs='+', type=int,
                       help='Seasons to scrape (default: current and previous)')
    parser.add_argument('--advanced', action='store_true',
                       help='Use advanced model training (MLE + Dixon-Coles)')
    parser.add_argument('--iterations', '-n', type=int, default=10000,
                       help='Number of Monte Carlo simulations (default: 10000)')
    parser.add_argument('--input', '-i', type=str,
                       help='Input file path')

    args = parser.parse_args()

    # Setup directories
    setup_directories()

    # Execute command
    try:
        if args.command == 'scrape':
            scrape_data(seasons=args.seasons)

        elif args.command == 'clean':
            clean_data(input_file=args.input)

        elif args.command == 'train':
            train_model(advanced=args.advanced)

        elif args.command == 'simulate':
            run_simulation(n_simulations=args.iterations)

        elif args.command == 'analyze':
            analyze_results(input_file=args.input)

        elif args.command == 'predict':
            predict_fixtures()

        elif args.command == 'pipeline':
            print("🚀 Running full pipeline...\n")

            # 1. Scrape
            raw_data = scrape_data(seasons=args.seasons)
            if raw_data is None:
                print("❌ Pipeline failed at scraping step")
                return 1

            print("\n" + "-"*60 + "\n")

            # 2. Clean
            results, fixtures = clean_data()
            if results is None:
                print("❌ Pipeline failed at cleaning step")
                return 1

            print("\n" + "-"*60 + "\n")

            # 3. Train
            model, team_stats = train_model(advanced=args.advanced)
            if model is None:
                print("❌ Pipeline failed at training step")
                return 1

            print("\n" + "-"*60 + "\n")

            # 4. Simulate
            sim_results = run_simulation(n_simulations=args.iterations)
            if sim_results is None:
                print("❌ Pipeline failed at simulation step")
                return 1

            print("\n" + "-"*60 + "\n")

            # 5. Analyze
            analysis = analyze_results()
            if analysis is None:
                print("❌ Pipeline failed at analysis step")
                return 1

            print("\n" + "-"*60 + "\n")

            # 6. Predict
            predictions = predict_fixtures()

            print("\n" + "="*60)
            print("✅ PIPELINE COMPLETE!")
            print("="*60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
