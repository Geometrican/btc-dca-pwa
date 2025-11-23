#!/usr/bin/env python3
"""
Percentile Calculator for DCA Dashboard

Calculates historical percentiles for market metrics to identify extreme conditions.
Uses 14+ years of BTC historical data to determine when current values are in
bottom 10% (buy signal) or top 10% (caution signal).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json


class PercentileCalculator:
    """Calculate and cache historical percentiles for market indicators"""

    def __init__(self, data_path: str = None):
        """
        Initialize percentile calculator

        Args:
            data_path: Path to BTCUSDT_1d.csv file. If None, uses default location.
        """
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "BTCUSDT_1d.csv"

        self.data_path = Path(data_path)
        self.df = None
        self.percentiles = {}

        # Load and calculate percentiles on init
        self._load_data()
        if self.df is not None:
            self._calculate_all_percentiles()

    def _load_data(self):
        """Load historical BTC price data"""
        try:
            if not self.data_path.exists():
                print(f"Warning: Data file not found at {self.data_path}")
                return

            df = pd.read_csv(self.data_path)
            df['date'] = pd.to_datetime(df['time_period_end']).dt.tz_localize(None)
            df = df.sort_values('date').reset_index(drop=True)

            self.df = df
            print(f"Loaded {len(df)} days of historical data ({df['date'].min()} to {df['date'].max()})")

        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = None

    def _calculate_all_percentiles(self):
        """Pre-calculate percentiles for all metrics"""
        if self.df is None:
            return

        print("Calculating historical percentiles...")

        # 1. Volatility percentiles (30-day rolling)
        self.percentiles['volatility'] = self._calc_volatility_percentiles()

        # 2. Volume percentiles
        self.percentiles['volume'] = self._calc_volume_percentiles()

        # 3. RSI percentiles
        self.percentiles['rsi'] = self._calc_rsi_percentiles()

        # 4. Price % from 200-MA percentiles
        self.percentiles['ma_distance'] = self._calc_ma_distance_percentiles()

        # 5. Drawdown from ATH percentiles
        self.percentiles['drawdown'] = self._calc_drawdown_percentiles()

        # Note: MVRV percentiles removed - use real-time MVRV data instead of estimation
        # Real MVRV should be fetched from Glassnode/CoinGlass APIs

        print("Percentile calculation complete!")
        self._print_percentile_summary()

    def _calc_volatility_percentiles(self) -> Dict:
        """Calculate 30-day volatility percentiles"""
        try:
            returns = self.df['close'].pct_change()
            volatilities = returns.rolling(window=30).std() * np.sqrt(365) * 100  # Annualized %
            volatilities = volatilities.dropna()

            return {
                'p10': float(np.percentile(volatilities, 10)),
                'p90': float(np.percentile(volatilities, 90)),
                'p50': float(np.percentile(volatilities, 50)),
                'count': len(volatilities),
                'unit': '%'
            }
        except Exception as e:
            print(f"Error calculating volatility percentiles: {e}")
            return {}

    def _calc_volume_percentiles(self) -> Dict:
        """Calculate volume percentiles"""
        try:
            volumes = self.df['volume'].dropna()

            return {
                'p10': float(np.percentile(volumes, 10)),
                'p90': float(np.percentile(volumes, 90)),
                'p50': float(np.percentile(volumes, 50)),
                'count': len(volumes),
                'unit': 'BTC'
            }
        except Exception as e:
            print(f"Error calculating volume percentiles: {e}")
            return {}

    def _calc_rsi_percentiles(self) -> Dict:
        """Calculate RSI percentiles"""
        try:
            prices = self.df['close']
            period = 14

            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.dropna()

            return {
                'p10': float(np.percentile(rsi, 10)),
                'p90': float(np.percentile(rsi, 90)),
                'p50': float(np.percentile(rsi, 50)),
                'count': len(rsi),
                'unit': ''
            }
        except Exception as e:
            print(f"Error calculating RSI percentiles: {e}")
            return {}

    def _calc_ma_distance_percentiles(self) -> Dict:
        """Calculate % distance from 200-MA percentiles"""
        try:
            prices = self.df['close']
            ma_200 = prices.rolling(window=200).mean()

            # Calculate % above/below MA
            pct_distance = ((prices / ma_200) - 1) * 100
            pct_distance = pct_distance.dropna()

            return {
                'p10': float(np.percentile(pct_distance, 10)),
                'p90': float(np.percentile(pct_distance, 90)),
                'p50': float(np.percentile(pct_distance, 50)),
                'count': len(pct_distance),
                'unit': '%'
            }
        except Exception as e:
            print(f"Error calculating MA distance percentiles: {e}")
            return {}

    def _calc_drawdown_percentiles(self) -> Dict:
        """Calculate drawdown from ATH percentiles"""
        try:
            prices = self.df['close']

            # Calculate running max (ATH at each point)
            running_max = prices.expanding().max()

            # Calculate drawdown %
            drawdown = ((prices / running_max) - 1) * 100  # Negative values
            drawdown = drawdown.dropna()

            return {
                'p10': float(np.percentile(drawdown, 10)),  # Most negative (deepest crashes)
                'p90': float(np.percentile(drawdown, 90)),  # Least negative (near ATH)
                'p50': float(np.percentile(drawdown, 50)),
                'count': len(drawdown),
                'unit': '%'
            }
        except Exception as e:
            print(f"Error calculating drawdown percentiles: {e}")
            return {}


    def _print_percentile_summary(self):
        """Print summary of calculated percentiles"""
        print("\n" + "="*60)
        print("HISTORICAL PERCENTILE THRESHOLDS")
        print("="*60)

        metrics = [
            ('volatility', 'Volatility (30d ann.)'),
            ('volume', 'Volume (24h)'),
            ('rsi', 'RSI (14-day)'),
            ('ma_distance', '% from 200-MA'),
            ('drawdown', 'Drawdown from ATH')
        ]

        for key, name in metrics:
            if key in self.percentiles and self.percentiles[key]:
                p = self.percentiles[key]
                unit = p.get('unit', '')
                print(f"\n{name}:")
                print(f"  10th percentile: {p['p10']:.2f}{unit}")
                print(f"  50th percentile: {p['p50']:.2f}{unit}")
                print(f"  90th percentile: {p['p90']:.2f}{unit}")
                if 'note' in p:
                    print(f"  Note: {p['note']}")

        print("="*60 + "\n")

    def rank_value(self, metric: str, current_value: float) -> Dict:
        """
        Rank a current value against historical percentiles

        Args:
            metric: One of 'volatility', 'volume', 'rsi', 'ma_distance', 'drawdown', 'mvrv'
            current_value: Current value to rank

        Returns:
            dict with percentile rank, signal, and interpretation
        """
        if metric not in self.percentiles or not self.percentiles[metric]:
            return {
                'success': False,
                'error': f'Percentiles not available for {metric}'
            }

        p = self.percentiles[metric]

        # Calculate approximate percentile rank
        if current_value <= p['p10']:
            percentile_rank = 10
            in_bottom_10 = True
            in_top_10 = False
        elif current_value >= p['p90']:
            percentile_rank = 90
            in_bottom_10 = False
            in_top_10 = True
        else:
            # Linear interpolation
            if current_value <= p['p50']:
                percentile_rank = 10 + (current_value - p['p10']) / (p['p50'] - p['p10']) * 40
            else:
                percentile_rank = 50 + (current_value - p['p50']) / (p['p90'] - p['p50']) * 40
            in_bottom_10 = False
            in_top_10 = False

        # Determine signal based on metric type
        signal = self._get_signal(metric, in_bottom_10, in_top_10)

        return {
            'success': True,
            'current_value': current_value,
            'percentile_rank': round(percentile_rank, 1),
            'in_bottom_10': in_bottom_10,
            'in_top_10': in_top_10,
            'p10_threshold': p['p10'],
            'p90_threshold': p['p90'],
            'signal': signal['signal'],
            'badge_color': signal['color'],
            'interpretation': signal['interpretation'],
            'unit': p.get('unit', '')
        }

    def _get_signal(self, metric: str, in_bottom_10: bool, in_top_10: bool) -> Dict:
        """
        Determine signal based on metric type and percentile position

        Different metrics have different interpretations:
        - Low volatility = complacency (decrease multiplier)
        - High volatility = opportunity (increase multiplier)
        - Low RSI = oversold (increase multiplier)
        - High RSI = overbought (decrease multiplier)
        - etc.
        """

        # Define signal logic for each metric
        signal_rules = {
            'volatility': {
                'bottom_10': ('Multiplier Decrease', 'warning', 'Dead calm - complacency'),
                'top_10': ('Multiplier Increase', 'success', 'High chaos - opportunity'),
                'normal': ('Normal Range', 'neutral', 'Normal volatility')
            },
            'volume': {
                'bottom_10': ('Multiplier Increase', 'success', 'Low volume - apathy/bottom'),
                'top_10': ('Multiplier Decrease', 'warning', 'High volume - euphoria'),
                'normal': ('Normal Range', 'neutral', 'Normal volume')
            },
            'rsi': {
                'bottom_10': ('Multiplier Increase', 'success', 'Extreme oversold'),
                'top_10': ('Multiplier Decrease', 'danger', 'Extreme overbought'),
                'normal': ('Normal Range', 'neutral', 'Normal RSI range')
            },
            'ma_distance': {
                'bottom_10': ('Multiplier Increase', 'success', 'Extreme discount to MA'),
                'top_10': ('Multiplier Decrease', 'danger', 'Extreme premium to MA'),
                'normal': ('Normal Range', 'neutral', 'Normal distance from MA')
            },
            'drawdown': {
                'bottom_10': ('Normal Range', 'neutral', 'Near ATH - no special signal'),
                'top_10': ('Multiplier Increase', 'success', 'Deep crash - max opportunity'),
                'normal': ('Normal Range', 'neutral', 'Normal drawdown')
            },
            'mvrv': {
                'bottom_10': ('Multiplier Increase', 'success', 'Extreme undervaluation'),
                'top_10': ('Multiplier Decrease', 'danger', 'Extreme overvaluation'),
                'normal': ('Normal Range', 'neutral', 'Fair value range')
            }
        }

        rules = signal_rules.get(metric, {
            'bottom_10': ('Normal Range', 'neutral', 'Unknown metric'),
            'top_10': ('Normal Range', 'neutral', 'Unknown metric'),
            'normal': ('Normal Range', 'neutral', 'Unknown metric')
        })

        if in_bottom_10:
            signal, color, interpretation = rules['bottom_10']
        elif in_top_10:
            signal, color, interpretation = rules['top_10']
        else:
            signal, color, interpretation = rules['normal']

        return {
            'signal': signal,
            'color': color,
            'interpretation': interpretation
        }

    def get_all_signals(self, current_values: Dict[str, float]) -> Dict:
        """
        Get percentile rankings and signals for all metrics at once

        Args:
            current_values: dict with metric names and their current values
                Example: {
                    'volatility': 65.2,
                    'rsi': 28.5,
                    'volume': 125000,
                    ...
                }

        Returns:
            dict with signal data for each metric
        """
        signals = {}

        for metric, value in current_values.items():
            if value is not None:
                signals[metric] = self.rank_value(metric, value)

        # Count special signals
        increase_signals = sum(1 for s in signals.values()
                              if s.get('success') and s['signal'] == 'Multiplier Increase')
        decrease_signals = sum(1 for s in signals.values()
                              if s.get('success') and s['signal'] == 'Multiplier Decrease')

        return {
            'signals': signals,
            'summary': {
                'increase_count': increase_signals,
                'decrease_count': decrease_signals,
                'total_metrics': len(signals)
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }

    def save_percentiles(self, filepath: str = 'percentile_thresholds.json'):
        """Save calculated percentiles to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.percentiles, f, indent=2)
            print(f"Percentiles saved to {filepath}")
        except Exception as e:
            print(f"Error saving percentiles: {e}")

    def load_percentiles(self, filepath: str = 'percentile_thresholds.json'):
        """Load percentiles from JSON file"""
        try:
            with open(filepath, 'r') as f:
                self.percentiles = json.load(f)
            print(f"Percentiles loaded from {filepath}")
        except Exception as e:
            print(f"Error loading percentiles: {e}")


# Module-level function for easy access
def create_percentile_calculator(data_path: str = None) -> PercentileCalculator:
    """Create and return a PercentileCalculator instance"""
    return PercentileCalculator(data_path=data_path)


if __name__ == '__main__':
    # Test the percentile calculator
    print("Testing Percentile Calculator\n")

    calc = PercentileCalculator()

    if calc.df is not None:
        # Test individual metric ranking
        print("\n" + "="*60)
        print("TESTING CURRENT VALUE RANKINGS")
        print("="*60)

        test_values = {
            'volatility': 75.0,  # High volatility
            'rsi': 25.0,         # Low RSI (oversold)
            'volume': 50000,     # Some volume
            'ma_distance': -15.0,  # Below MA
            'drawdown': -30.0,   # 30% from ATH
            'mvrv': 1.2          # Undervalued
        }

        for metric, value in test_values.items():
            result = calc.rank_value(metric, value)
            if result['success']:
                print(f"\n{metric.upper()}: {value}{result['unit']}")
                print(f"  Percentile: {result['percentile_rank']}th")
                print(f"  Signal: {result['signal']}")
                print(f"  {result['interpretation']}")

        # Test batch signal generation
        print("\n" + "="*60)
        print("BATCH SIGNAL GENERATION")
        print("="*60)

        all_signals = calc.get_all_signals(test_values)
        print(f"\nIncrease signals: {all_signals['summary']['increase_count']}")
        print(f"Decrease signals: {all_signals['summary']['decrease_count']}")
        print(f"Total metrics: {all_signals['summary']['total_metrics']}")
