#!/usr/bin/env python3
"""
Bitcoin DCA Decision Dashboard

Real-time web UI showing all indicators needed for sentiment-driven DCA purchases:
- Fear & Greed Index
- Position size multiplier recommendation
- BTC price and technical indicators
- Final recommended purchase amount
"""

from flask import Flask, render_template, jsonify, request, abort
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
from functools import wraps
from percentile_calculator import PercentileCalculator
from mvrv_estimator import MVRVEstimator, load_historical_prices

app = Flask(__name__)

# IP Whitelist Configuration
# Set ALLOWED_IPS environment variable on Render with comma-separated IPs
# Example: "123.456.789.0,98.765.432.1" or set to "all" to disable
ALLOWED_IPS = os.environ.get('ALLOWED_IPS', 'all')

def check_ip_whitelist():
    """Middleware to check if request IP is whitelisted"""
    if ALLOWED_IPS == 'all':
        return  # IP filtering disabled

    # Get client IP (works with Render's proxy setup)
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ',' in client_ip:
        client_ip = client_ip.split(',')[0].strip()

    allowed_ips = [ip.strip() for ip in ALLOWED_IPS.split(',')]

    if client_ip not in allowed_ips:
        abort(403)  # Forbidden

@app.before_request
def before_request():
    """Run before each request"""
    check_ip_whitelist()

class DCACalculator:
    """Calculate DCA recommendations based on current market conditions"""

    def __init__(self, base_amount: float = 100.0):
        self.base_amount = base_amount
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes

    def get_fear_greed_index(self) -> dict:
        """Fetch current Fear & Greed Index from API"""
        cache_key = 'fear_greed'

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_timeout:
                return cached_data

        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=10)
            data = response.json()

            if 'data' in data and len(data['data']) > 0:
                fg_data = data['data'][0]
                result = {
                    'value': int(fg_data['value']),
                    'classification': fg_data['value_classification'],
                    'timestamp': datetime.fromtimestamp(int(fg_data['timestamp'])),
                    'success': True
                }
                self.cache[cache_key] = (datetime.now(), result)
                return result
        except Exception as e:
            print(f"Error fetching Fear & Greed: {e}")

        return {'success': False, 'error': 'Failed to fetch Fear & Greed Index'}

    def get_btc_price(self) -> dict:
        """Fetch current BTC price from Binance"""
        cache_key = 'btc_price'

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 60:  # 1 minute cache
                return cached_data

        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            response = requests.get(url, timeout=10)
            data = response.json()

            result = {
                'price': float(data['price']),
                'timestamp': datetime.now(),
                'success': True
            }
            self.cache[cache_key] = (datetime.now(), result)
            return result
        except Exception as e:
            print(f"Error fetching BTC price: {e}")

        return {'success': False, 'error': 'Failed to fetch BTC price'}

    def get_historical_prices(self, days: int = 250) -> pd.Series:
        """Fetch historical daily prices for technical indicators"""
        cache_key = f'historical_{days}'

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 3600:  # 1 hour cache
                return cached_data

        try:
            # Try to use local data first
            data_path = Path(__file__).parent / "data" / "BTCUSDT_1d.csv"
            if data_path.exists():
                df = pd.read_csv(data_path)
                df['date'] = pd.to_datetime(df['time_period_end']).dt.tz_localize(None)
                df = df.sort_values('date')
                # Set date as index for the Series
                df_tail = df.tail(days)
                prices = pd.Series(df_tail['close'].values, index=df_tail['date'].values)
                self.cache[cache_key] = (datetime.now(), prices)
                return prices

            # Fallback to API
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit={days}"
            response = requests.get(url, timeout=10)
            data = response.json()

            prices = pd.Series([float(candle[4]) for candle in data])  # Close prices
            self.cache[cache_key] = (datetime.now(), prices)
            return prices
        except Exception as e:
            print(f"Error fetching historical prices: {e}")
            return pd.Series()

    def calculate_ma_200(self, current_price: float) -> dict:
        """Calculate 200-day moving average"""
        prices = self.get_historical_prices(250)

        if len(prices) < 200:
            return {'success': False, 'error': 'Insufficient data'}

        ma_200 = float(prices.tail(200).mean())
        pct_above = float(((current_price / ma_200) - 1) * 100)

        return {
            'ma_200': ma_200,
            'pct_above': pct_above,
            'above_threshold': bool(pct_above > 50),
            'success': True
        }

    def calculate_drawdown_from_peak(self, current_price: float) -> dict:
        """Calculate drawdown from recent peak (1-year lookback)"""
        prices = self.get_historical_prices(365)

        if len(prices) < 30:
            return {'success': False, 'error': 'Insufficient data'}

        # Find peak price and its date
        # idxmax() returns the label (date) of the maximum value
        peak_date = prices.idxmax()  # This is the timestamp/date of the peak
        peak_price = float(prices.max())

        # Calculate months from peak
        current_date = datetime.now()
        if isinstance(peak_date, pd.Timestamp):
            months_from_peak = (current_date.year - peak_date.year) * 12 + (current_date.month - peak_date.month)
            days_from_peak = (current_date - peak_date.to_pydatetime()).days
            peak_date_str = peak_date.strftime('%Y-%m-%d')
        else:
            months_from_peak = 0
            days_from_peak = 0
            peak_date_str = str(peak_date)

        drawdown = float(((current_price / peak_price) - 1) * 100)

        # Determine bear phase based on drawdown
        if drawdown > -40:
            phase = "SHALLOW"  # Early bear
            risk_level = "VERY_HIGH"
        elif drawdown > -60:
            phase = "MODERATE"  # Mid bear
            risk_level = "HIGH"
        else:
            phase = "DEEP"  # Late bear
            risk_level = "MODERATE"

        return {
            'peak_price': peak_price,
            'peak_date': peak_date_str,
            'days_from_peak': days_from_peak,
            'months_from_peak': months_from_peak,
            'drawdown_pct': drawdown,
            'phase': phase,
            'risk_level': risk_level,
            'success': True
        }

    def get_bear_market_allocation_phase(self, drawdown_pct: float, months_from_peak: int,
                                           days_from_peak: int, fear_greed: int) -> dict:
        """
        Determine bear market allocation phase based on historical patterns

        Historical markers (average across bear markets):
        - 27 days to -20%: First warning
        - 39 days to -30%: Confirmation of bear
        - 99 days to -40%: Panic begins (START HEAVY BUYING)
        - 196 days to -50%: Deep fear (MAXIMUM ALLOCATION)
        - 280 days to -60%: Capitulation (GENERATIONAL BOTTOM)

        Allocation strategy for $10,000 capital:
        - Month 1-3 (0-90 days, -20% to -30%): 5-10% allocation per month
        - Month 4-5 (90-150 days, -30% to -40%): 15% per month
        - Month 6-7 (150-210 days, -40% to -50%): 20% per month (AGGRESSIVE)
        - Month 8-9 (210-270 days, >-50%): 10-5% per month (near bottom)
        """

        # Determine allocation multiplier based on phase
        if months_from_peak <= 3 or drawdown_pct > -30:
            # Early bear - be patient
            phase_name = "EARLY BEAR"
            allocation_pct = 5.0
            phase_multiplier = 0.5
            description = "Slow bleed phase - be very patient"
            color = "warning"
            confidence = "LOW"

        elif months_from_peak <= 5 or (drawdown_pct > -40 and drawdown_pct <= -30):
            # Approaching panic
            phase_name = "PRE-PANIC"
            allocation_pct = 15.0
            phase_multiplier = 1.5
            description = "Building position as fear increases"
            color = "info"
            confidence = "MEDIUM"

        elif months_from_peak <= 7 or (drawdown_pct > -50 and drawdown_pct <= -40):
            # Panic zone - maximum aggression
            phase_name = "PANIC ZONE"
            allocation_pct = 20.0
            phase_multiplier = 2.5
            description = "MAXIMUM ALLOCATION - Peak fear zone"
            color = "success"
            confidence = "HIGH"

        elif drawdown_pct <= -50:
            # Deep capitulation - likely near bottom
            phase_name = "CAPITULATION"
            allocation_pct = 10.0
            phase_multiplier = 2.0
            description = "Deep drawdown - likely near bottom"
            color = "success"
            confidence = "VERY_HIGH"

        else:
            # Uncertain phase
            phase_name = "UNCERTAIN"
            allocation_pct = 5.0
            phase_multiplier = 0.5
            description = "Unclear phase - be conservative"
            color = "secondary"
            confidence = "VERY_LOW"

        # Adjust by fear & greed (enhance signal)
        fg_adjustment = 1.0
        fg_note = ""
        if fear_greed <= 18:
            fg_adjustment = 1.2
            fg_note = "Extreme fear adds conviction (+20%)"
        elif fear_greed <= 25:
            fg_adjustment = 1.1
            fg_note = "High fear adds conviction (+10%)"
        elif fear_greed >= 40:
            fg_adjustment = 0.7
            fg_note = "Sentiment not fearful enough (-30%)"

        final_multiplier = phase_multiplier * fg_adjustment

        # Historical context
        if days_from_peak < 99:
            historical_note = f"Historically, -40% reached at ~99 days. Currently at day {days_from_peak}"
        elif days_from_peak < 196:
            historical_note = f"Historically, -50% reached at ~196 days. Currently at day {days_from_peak}"
        elif days_from_peak < 280:
            historical_note = f"Historically, -60% reached at ~280 days. Currently at day {days_from_peak}"
        else:
            historical_note = f"Beyond typical bear duration ({days_from_peak} days). Likely near bottom or recovering"

        return {
            'phase_name': phase_name,
            'allocation_pct': allocation_pct,
            'base_multiplier': phase_multiplier,
            'fg_adjustment': fg_adjustment,
            'final_multiplier': final_multiplier,
            'description': description,
            'fg_note': fg_note,
            'historical_note': historical_note,
            'color': color,
            'confidence': confidence,
            'success': True
        }

    def calculate_rsi(self, period: int = 14) -> dict:
        """Calculate RSI indicator"""
        prices = self.get_historical_prices(50)

        if len(prices) < period + 1:
            return {'success': False, 'error': 'Insufficient data'}

        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = float(rsi.iloc[-1])

        return {
            'rsi': current_rsi,
            'overheated': bool(current_rsi > 80),
            'success': True
        }

    def get_funding_rate(self) -> dict:
        """Get current BTC perpetual funding rate from Binance"""
        cache_key = 'funding_rate'

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:  # 5 minute cache
                return cached_data

        try:
            # Binance funding rate endpoint
            url = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1"
            response = requests.get(url, timeout=10)
            data = response.json()

            if data and len(data) > 0:
                funding_rate = float(data[0]['fundingRate'])
                funding_pct = funding_rate * 100  # Convert to percentage

                # Classify funding rate
                if funding_rate > 0.01:
                    classification = "Extremely Bullish"
                    signal = "Over-leveraged longs - correction risk"
                    color = "danger"
                elif funding_rate > 0.0001:
                    classification = "Bullish"
                    signal = "Healthy long bias"
                    color = "success"
                elif funding_rate >= -0.0001:
                    classification = "Neutral"
                    signal = "Balanced market"
                    color = "info"
                elif funding_rate > -0.01:
                    classification = "Bearish"
                    signal = "Short bias"
                    color = "warning"
                else:
                    classification = "Extremely Bearish"
                    signal = "Over-leveraged shorts - bounce risk"
                    color = "success"

                result = {
                    'funding_rate': funding_rate,
                    'funding_pct': funding_pct,
                    'classification': classification,
                    'signal': signal,
                    'color': color,
                    'timestamp': datetime.fromtimestamp(int(data[0]['fundingTime']) / 1000),
                    'success': True
                }
                self.cache[cache_key] = (datetime.now(), result)
                return result

        except Exception as e:
            print(f"Error fetching funding rate: {e}")

        return {'success': False, 'error': 'Failed to fetch funding rate'}

    def calculate_volatility(self, period: int = 30) -> dict:
        """Calculate 30-day annualized volatility"""
        prices = self.get_historical_prices(50)

        if len(prices) < period + 1:
            return {'success': False, 'error': 'Insufficient data'}

        returns = prices.pct_change().dropna()
        recent_returns = returns.tail(period)
        volatility = float(recent_returns.std() * np.sqrt(365))  # Annualized

        # Classify volatility level
        if volatility > 0.80:
            level = "Very High"
            adjustment = "+0.5×"
        elif volatility > 0.60:
            level = "High"
            adjustment = "+0.25×"
        elif volatility < 0.40:
            level = "Low"
            adjustment = "-0.25×"
        else:
            level = "Normal"
            adjustment = "No change"

        return {
            'volatility': volatility * 100,  # Convert to percentage
            'level': level,
            'adjustment': adjustment,
            'success': True
        }

    def get_mvrv_ratio(self) -> dict:
        """Get MVRV ratio using smart estimation"""
        cache_key = 'mvrv'

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 3600:  # 1 hour cache
                return cached_data

        try:
            # Get current price
            btc_price_data = self.get_btc_price()
            if not btc_price_data['success']:
                return {'success': False, 'error': 'Could not fetch BTC price'}

            # Use MVRV estimator
            estimator = get_mvrv_estimator()
            estimate = estimator.estimate_mvrv(btc_price_data['price'])

            # Override cycle phase with actual market regime
            # The estimator uses time-based phases, but we need reality-based phases
            ma_200_data = self.calculate_ma_200(btc_price_data['price'])
            if ma_200_data['success']:
                actual_regime = self.detect_regime(btc_price_data['price'], ma_200_data['ma_200'])

                # Map regime to realistic cycle phase
                if actual_regime == "BEAR":
                    # In bear market - override estimator's time-based phase
                    drawdown_data = self.calculate_drawdown_from_peak(btc_price_data['price'])
                    if drawdown_data.get('success'):
                        dd = drawdown_data['drawdown_pct']
                        if dd <= -60:
                            cycle_phase = "deep bear (capitulation)"
                        elif dd <= -40:
                            cycle_phase = "bear market (panic)"
                        else:
                            cycle_phase = "bear market (early)"
                    else:
                        cycle_phase = "bear market"

                elif actual_regime == "BULL":
                    # In bull market - keep estimator's phase if it's bull-related
                    if estimate['cycle_phase'] in ['accumulation', 'bull_run', 'euphoria']:
                        cycle_phase = estimate['cycle_phase']
                    else:
                        cycle_phase = "bull market"

                else:  # NEUTRAL
                    cycle_phase = "consolidation"
            else:
                # Fallback to estimator's phase if we can't detect regime
                cycle_phase = estimate['cycle_phase']

            # Classify the estimated MVRV
            result = self._classify_mvrv(
                estimate['mvrv'],
                estimated=True,
                confidence=estimate['confidence'],
                cycle_phase=cycle_phase
            )

            self.cache[cache_key] = (datetime.now(), result)
            return result

        except Exception as e:
            print(f"MVRV estimation failed: {e}")
            return {'success': False, 'error': f'MVRV estimation failed: {e}'}

    def _classify_mvrv(self, mvrv: float, estimated: bool = False, confidence: float = 0.0, cycle_phase: str = "") -> dict:
        """Classify MVRV ratio and provide recommendations"""
        # MVRV zones based on research
        if mvrv > 3.5:
            zone = "Extreme Overvaluation"
            signal = "SKIP purchases"
            multiplier_suggestion = "0×"
            color = "danger"
        elif mvrv > 2.5:
            zone = "Overvalued"
            signal = "Reduce allocation"
            multiplier_suggestion = "0.5×"
            color = "warning"
        elif mvrv > 1.5:
            zone = "Fair Value"
            signal = "Normal DCA"
            multiplier_suggestion = "1×"
            color = "info"
        elif mvrv > 1.0:
            zone = "Undervalued"
            signal = "Increase allocation"
            multiplier_suggestion = "1.5×"
            color = "success"
        else:
            zone = "Extreme Undervaluation"
            signal = "MAX allocation"
            multiplier_suggestion = "2×"
            color = "success"

        return {
            'mvrv': mvrv,
            'zone': zone,
            'signal': signal,
            'multiplier_suggestion': multiplier_suggestion,
            'color': color,
            'estimated': estimated,
            'confidence': confidence,
            'cycle_phase': cycle_phase,
            'success': True
        }

    def detect_regime(self, price: float, ma_200: float) -> str:
        """
        Simple but effective regime detection
        Returns: "BULL", "BEAR", or "NEUTRAL"
        """
        if price > ma_200 * 1.1:  # 10% above 200-MA
            return "BULL"
        elif price < ma_200 * 0.9:  # 10% below 200-MA
            return "BEAR"
        else:
            return "NEUTRAL"

    def get_halving_cycle_position(self) -> dict:
        """Calculate position in current halving cycle"""
        HALVINGS = [
            datetime(2012, 11, 28),
            datetime(2016, 7, 9),
            datetime(2020, 5, 11),
            datetime(2024, 4, 20),
            datetime(2028, 4, 1),  # Estimated
        ]

        current_date = datetime.now()
        months_from_halving = None
        last_halving = None

        for halving in reversed(HALVINGS):
            if halving < current_date:
                months_from_halving = (current_date - halving).days / 30
                last_halving = halving
                break

        return {
            'months_from_halving': months_from_halving,
            'last_halving': last_halving,
            'success': months_from_halving is not None
        }

    def calculate_bull_run_size(self, current_price: float) -> dict:
        """Calculate size of bull run before current ATH"""
        try:
            prices = self.get_historical_prices(1500)  # Get enough history

            if len(prices) < 365:
                return {'success': False, 'error': 'Insufficient data'}

            # Find ATH
            ath_price = float(prices.max())
            ath_idx = prices.idxmax()
            ath_position = prices.index.get_loc(ath_idx)

            # Find previous bear bottom within REASONABLE timeframe (2 years before ATH)
            # This avoids going back to ancient $2 Bitcoin prices
            lookback_days = min(730, ath_position)  # 2 years or less

            # Get prices in the lookback window (before ATH)
            start_idx = max(0, ath_position - lookback_days)
            window = prices.iloc[start_idx:ath_position]

            if len(window) == 0:
                return {'success': False, 'error': 'No data before ATH'}

            # Find the LOWEST price in this window (most recent bear bottom)
            prev_bottom_price = float(window.min())

            # Calculate bull run gain
            bull_run_gain = ((ath_price / prev_bottom_price) - 1) * 100

            return {
                'ath_price': ath_price,
                'prev_bottom_price': prev_bottom_price,
                'bull_run_gain_pct': bull_run_gain,
                'success': True
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def classify_bear_market_risk(self) -> dict:
        """
        Classify whether current situation resembles:
        - Quick bounce (91.3% historical cases)
        - Brutal bear (8.7% historical cases)

        Returns probabilities and automatic multiplier adjustments
        """
        cache_key = 'bear_classification'

        # Check cache (1 hour)
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 3600:
                return cached_data

        try:
            # Get current data
            btc_price_data = self.get_btc_price()
            if not btc_price_data['success']:
                return {'success': False, 'error': 'Could not fetch BTC price'}

            current_price = btc_price_data['price']

            # Get all required metrics
            prices = self.get_historical_prices(1500)
            if len(prices) < 365:
                return {'success': False, 'error': 'Insufficient data'}

            ath_price = float(prices.max())
            ath_idx = prices.idxmax()
            current_dd_from_ath = ((current_price / ath_price) - 1) * 100

            # Calculate months from ATH
            if isinstance(ath_idx, pd.Timestamp):
                months_from_ath = (datetime.now() - ath_idx.to_pydatetime()).days / 30
            else:
                months_from_ath = 0

            # Get MA 200 data
            ma_200_data = self.calculate_ma_200(current_price)
            if not ma_200_data['success']:
                return {'success': False, 'error': 'Could not calculate 200-MA'}

            # Get bull run size
            bull_run_data = self.calculate_bull_run_size(current_price)
            bull_run_gain = bull_run_data.get('bull_run_gain_pct', 0) if bull_run_data['success'] else 0

            # Get halving position
            halving_data = self.get_halving_cycle_position()
            months_from_halving = halving_data.get('months_from_halving', 0)

            # Get drawdown data
            drawdown_data = self.calculate_drawdown_from_peak(current_price)
            cross_dd = current_dd_from_ath  # Approximation

            # BRUTAL BEARS HISTORICAL AVERAGES
            BRUTAL_AVG_DD_AT_CROSS = -48.0
            BRUTAL_AVG_MONTHS_FROM_ATH = 4.0
            BRUTAL_AVG_BULL_RUN = 700.0

            # SCORING SYSTEM (max 10 points each direction)
            score_bounce = 0
            score_brutal = 0

            # Factor 1: Drawdown from ATH when crossing below 200-MA
            if current_dd_from_ath > -40:  # Shallow
                score_bounce += 2
            elif current_dd_from_ath > (BRUTAL_AVG_DD_AT_CROSS + 10):
                score_bounce += 1
            else:
                score_brutal += 2

            # Factor 2: Months from ATH
            if months_from_ath < 2:
                score_brutal += 1  # Very close - risky
            elif months_from_ath > 6:
                score_bounce += 2  # Long time - absorbed pain
            else:
                score_bounce += 1

            # Factor 3: Bull run size
            if bull_run_gain > 1000:
                score_brutal += 2  # Mega bubble
            elif bull_run_gain > 500:
                score_brutal += 1  # Large gain
            else:
                score_bounce += 2  # Moderate gain

            # Factor 4: Halving cycle position
            if months_from_halving and months_from_halving < 24:
                score_bounce += 2  # Early/mid cycle
            elif months_from_halving and months_from_halving > 36:
                score_brutal += 1  # Late cycle

            # Calculate probabilities
            total_score = score_bounce + score_brutal
            if total_score > 0:
                bounce_probability = (score_bounce / total_score * 100)
                brutal_probability = (score_brutal / total_score * 100)
            else:
                bounce_probability = 50
                brutal_probability = 50

            # Determine verdict
            if bounce_probability > 70:
                verdict = "LIKELY_QUICK_BOUNCE"
                verdict_text = "✅ LIKELY QUICK BOUNCE"
                verdict_color = "success"
            elif bounce_probability > 50:
                verdict = "LEANING_BOUNCE"
                verdict_text = "⚠️ LEANING BOUNCE"
                verdict_color = "warning"
            elif brutal_probability > 70:
                verdict = "LIKELY_BRUTAL_BEAR"
                verdict_text = "❌ LIKELY BRUTAL BEAR"
                verdict_color = "danger"
            else:
                verdict = "UNCERTAIN"
                verdict_text = "⚠️ UNCERTAIN"
                verdict_color = "warning"

            # Calculate multiplier adjustment (works BOTH ways)
            multiplier_adjustment = 1.0
            adjustment_reason = None

            # BRUTAL BEAR RISK: Reduce buying
            if brutal_probability > 70:
                multiplier_adjustment = 0.6  # 40% reduction
                adjustment_reason = "High brutal bear risk (>70%) - being extra conservative"
            elif brutal_probability > 50:
                multiplier_adjustment = 0.8  # 20% reduction
                adjustment_reason = "Moderate brutal bear risk (>50%) - slightly more conservative"

            # BOUNCE CONFIDENCE: Increase buying (buy the dip!)
            elif bounce_probability > 70:
                multiplier_adjustment = 1.5  # 50% increase
                adjustment_reason = "High bounce confidence (>70%) - buying the dip aggressively!"
            elif bounce_probability > 60:
                multiplier_adjustment = 1.25  # 25% increase
                adjustment_reason = "Good bounce confidence (>60%) - slightly more aggressive"

            result = {
                'bounce_probability': round(bounce_probability, 1),
                'brutal_probability': round(brutal_probability, 1),
                'verdict': verdict,
                'verdict_text': verdict_text,
                'verdict_color': verdict_color,
                'score_bounce': score_bounce,
                'score_brutal': score_brutal,
                'multiplier_adjustment': multiplier_adjustment,
                'adjustment_reason': adjustment_reason,
                'factors': {
                    'current_dd_from_ath': round(current_dd_from_ath, 1),
                    'months_from_ath': round(months_from_ath, 1),
                    'bull_run_gain_pct': round(bull_run_gain, 0),
                    'months_from_halving': round(months_from_halving, 1) if months_from_halving else None,
                    'brutal_avg_dd': BRUTAL_AVG_DD_AT_CROSS,
                    'brutal_avg_months': BRUTAL_AVG_MONTHS_FROM_ATH,
                    'brutal_avg_bull_run': BRUTAL_AVG_BULL_RUN
                },
                'success': True
            }

            self.cache[cache_key] = (datetime.now(), result)
            return result

        except Exception as e:
            print(f"Bear classification failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def get_multiplier_by_regime(self, fear_greed: int, regime: str, drawdown_pct: float = 0) -> dict:
        """
        Get DCA multiplier based on Fear & Greed, market regime, AND drawdown from peak
        Returns dict with multiplier, expected_winrate, and reasoning

        CRITICAL: Early bear protection - don't buy aggressively until >-40% down
        AUTOMATIC ADJUSTMENT: Reduces multipliers if brutal bear risk is high
        """
        # Get bear market classification for automatic adjustment
        bear_classification = self.classify_bear_market_risk()
        bounce_probability = bear_classification.get('bounce_probability', 0) if bear_classification.get('success') else 0
        brutal_probability = bear_classification.get('brutal_probability', 0) if bear_classification.get('success') else 0
        multiplier_adjustment = 1.0
        adjustment_reason = None

        if bear_classification.get('success'):
            multiplier_adjustment = bear_classification.get('multiplier_adjustment', 1.0)
            adjustment_reason = bear_classification.get('adjustment_reason')

        if regime == "BEAR":
            # BOUNCE CONFIDENCE OVERRIDE: If high bounce probability, ignore conservative early bear logic
            if bounce_probability > 70 and drawdown_pct > -40:
                # Use aggressive bounce-buying multipliers instead of early bear protection
                if fear_greed <= 18:
                    base_mult = 2.5  # Extreme fear + high bounce confidence
                elif fear_greed <= 25:
                    base_mult = 2.0  # High fear + high bounce confidence
                elif fear_greed <= 35:
                    base_mult = 1.5  # Moderate fear + high bounce confidence
                else:
                    base_mult = 1.0  # Normal conditions

                final_mult = base_mult
                return {
                    'multiplier': final_mult,
                    'expected_winrate': 65.0,
                    'reason': f'BOUNCE CONFIDENT ({drawdown_pct:.1f}%) + F&G {fear_greed} - BUYING THE DIP',
                    'confidence': 'HIGH',
                    'bounce_override': True,
                    'warning': f'{bounce_probability:.0f}% bounce confidence - ignoring early bear protection'
                }

            # PHASE-TIMING STRATEGY (Backtest-validated 2400% return)
            # Only used if brutal bear is possible OR we're deep in drawdown
            # Prioritize DRAWDOWN over time - deepest drawdowns get maximum aggression

            # Calculate base multiplier and F&G adjustment
            if fear_greed <= 18:
                fg_mult = 1.2  # Extreme fear adds conviction
            elif fear_greed <= 25:
                fg_mult = 1.1  # High fear adds conviction
            elif fear_greed >= 40:
                fg_mult = 0.7  # Not fearful enough, reduce
            else:
                fg_mult = 1.0

            # DEEP CAPITULATION (>-60%) - MAXIMUM AGGRESSION
            if drawdown_pct <= -60:
                base_mult = 3.5
                final_mult = base_mult * fg_mult * multiplier_adjustment
                result = {
                    'multiplier': final_mult,
                    'expected_winrate': 70.0,
                    'reason': f'DEEP CAPITULATION ({drawdown_pct:.1f}%) + F&G {fear_greed} - MAXIMUM AGGRESSION',
                    'confidence': 'VERY_HIGH',
                    'warning': 'Best buying opportunity - 2869% historical ROI'
                }
                if adjustment_reason:
                    result['brutal_bear_adjustment'] = f"{multiplier_adjustment:.0%} - {adjustment_reason}"
                return result

            # PANIC ZONE (-40% to -60%)
            elif drawdown_pct <= -40:
                base_mult = 2.5
                final_mult = base_mult * fg_mult * multiplier_adjustment
                result = {
                    'multiplier': final_mult,
                    'expected_winrate': 60.0,
                    'reason': f'PANIC ZONE ({drawdown_pct:.1f}%) + F&G {fear_greed} - HIGH AGGRESSION',
                    'confidence': 'HIGH',
                    'warning': '2525% historical ROI in this zone'
                }
                if adjustment_reason:
                    result['brutal_bear_adjustment'] = f"{multiplier_adjustment:.0%} - {adjustment_reason}"
                return result

            # PRE-PANIC (-30% to -40%)
            elif drawdown_pct <= -30:
                base_mult = 1.5
                final_mult = base_mult * fg_mult * multiplier_adjustment
                result = {
                    'multiplier': final_mult,
                    'expected_winrate': 50.0,
                    'reason': f'PRE-PANIC ({drawdown_pct:.1f}%) + F&G {fear_greed} - BUILDING POSITION',
                    'confidence': 'MEDIUM'
                }
                if adjustment_reason:
                    result['brutal_bear_adjustment'] = f"{multiplier_adjustment:.0%} - {adjustment_reason}"
                return result

            # EARLY BEAR (>-30%)
            else:
                base_mult = 0.5
                final_mult = base_mult * fg_mult * multiplier_adjustment
                result = {
                    'multiplier': final_mult,
                    'expected_winrate': 30.0,
                    'reason': f'EARLY BEAR ({drawdown_pct:.1f}%) + F&G {fear_greed} - VERY CONSERVATIVE',
                    'confidence': 'LOW',
                    'warning': 'Early bear - price may drop another -30% to -50%'
                }
                if adjustment_reason:
                    result['brutal_bear_adjustment'] = f"{multiplier_adjustment:.0%} - {adjustment_reason}"
                return result

        elif regime == "BULL":
            if fear_greed <= 31:
                return {
                    'multiplier': 3.0,
                    'expected_winrate': 75.2,
                    'reason': f'BULL market + Extreme Fear (F&G {fear_greed})',
                    'confidence': 'HIGH'
                }
            elif fear_greed <= 45:
                return {
                    'multiplier': 2.5,
                    'expected_winrate': 71.0,
                    'reason': f'BULL market + Fear (F&G {fear_greed})',
                    'confidence': 'HIGH'
                }
            elif fear_greed <= 60:
                return {
                    'multiplier': 1.5,
                    'expected_winrate': 62.0,
                    'reason': f'BULL market + Neutral (F&G {fear_greed})',
                    'confidence': 'MEDIUM'
                }
            elif fear_greed <= 72:
                return {
                    'multiplier': 0.7,
                    'expected_winrate': 55.0,
                    'reason': f'BULL market + Mild Greed (F&G {fear_greed})',
                    'confidence': 'LOW'
                }
            else:
                return {
                    'multiplier': 0.0,
                    'expected_winrate': 44.0,
                    'reason': f'BULL market + High Greed (F&G {fear_greed}) - SKIP',
                    'confidence': 'VERY_LOW'
                }

        else:  # NEUTRAL
            # Average of both bull and bear strategies
            if fear_greed <= 25:
                return {
                    'multiplier': 2.5,
                    'expected_winrate': 65.0,
                    'reason': f'NEUTRAL market + Extreme Fear (F&G {fear_greed})',
                    'confidence': 'HIGH'
                }
            elif fear_greed <= 45:
                return {
                    'multiplier': 1.5,
                    'expected_winrate': 60.0,
                    'reason': f'NEUTRAL market + Fear (F&G {fear_greed})',
                    'confidence': 'MEDIUM'
                }
            elif fear_greed <= 60:
                return {
                    'multiplier': 0.8,
                    'expected_winrate': 52.0,
                    'reason': f'NEUTRAL market + Neutral Sentiment (F&G {fear_greed})',
                    'confidence': 'LOW'
                }
            else:
                return {
                    'multiplier': 0.0,
                    'expected_winrate': 45.0,
                    'reason': f'NEUTRAL market + Greed (F&G {fear_greed}) - SKIP',
                    'confidence': 'VERY_LOW'
                }

    def get_fg_multiplier(self, fg_value: int) -> float:
        """
        DEPRECATED: Use get_multiplier_by_regime instead
        Calculate multiplier based on Fear & Greed Index (Historical Percentile Enhanced)
        """
        if fg_value <= 15:
            return 3.0  # Extreme Fear
        elif fg_value <= 30:
            return 2.0  # Fear
        elif fg_value <= 70:
            return 1.0  # Neutral
        else:
            return 0.0  # Greed - SKIP

    def calculate_recommendation(self) -> dict:
        """Calculate complete DCA recommendation"""
        result = {
            'timestamp': datetime.now(),
            'base_amount': self.base_amount
        }

        # Get Fear & Greed
        fg_data = self.get_fear_greed_index()
        if not fg_data['success']:
            return {'success': False, 'error': fg_data.get('error')}

        result['fear_greed'] = fg_data

        # Get BTC price
        price_data = self.get_btc_price()
        if not price_data['success']:
            return {'success': False, 'error': price_data.get('error')}

        result['btc_price'] = price_data['price']

        # Get 200-day MA first (needed for regime detection)
        ma_data = self.calculate_ma_200(price_data['price'])
        result['ma_200'] = ma_data

        # Calculate drawdown from peak (CRITICAL for early bear detection)
        drawdown_data = self.calculate_drawdown_from_peak(price_data['price'])
        result['drawdown'] = drawdown_data

        # Detect market regime
        if ma_data['success']:
            regime = self.detect_regime(price_data['price'], ma_data['ma_200'])

            # Add drawdown info to regime description
            drawdown_pct = drawdown_data.get('drawdown_pct', 0) if drawdown_data['success'] else 0
            phase = drawdown_data.get('phase', 'UNKNOWN') if drawdown_data['success'] else 'UNKNOWN'

            result['regime'] = {
                'type': regime,
                'price': price_data['price'],
                'ma_200': ma_data['ma_200'],
                'pct_above_ma': ma_data['pct_above'],
                'drawdown_pct': drawdown_pct,
                'phase': phase,
                'description': f"Price is {abs(ma_data['pct_above']):.1f}% {'above' if ma_data['pct_above'] > 0 else 'below'} 200-MA"
            }

            # Calculate regime-aware multiplier WITH drawdown protection
            multiplier_data = self.get_multiplier_by_regime(fg_data['value'], regime, drawdown_pct)
            base_multiplier = multiplier_data['multiplier']
            result['base_multiplier'] = base_multiplier
            result['base_multiplier_reason'] = multiplier_data['reason']
            result['expected_winrate'] = multiplier_data['expected_winrate']
            result['signal_confidence'] = multiplier_data['confidence']
            if 'warning' in multiplier_data:
                result['early_bear_warning'] = multiplier_data['warning']

            # Add bear market allocation phase (if in bear market)
            if regime == "BEAR" and drawdown_data['success']:
                bear_phase = self.get_bear_market_allocation_phase(
                    drawdown_pct=drawdown_pct,
                    months_from_peak=drawdown_data.get('months_from_peak', 0),
                    days_from_peak=drawdown_data.get('days_from_peak', 0),
                    fear_greed=fg_data['value']
                )
                result['bear_allocation_phase'] = bear_phase
        else:
            # Fallback to old method if MA calculation fails
            regime = "UNKNOWN"
            result['regime'] = {'type': regime, 'description': 'Unable to detect regime (MA calculation failed)'}
            base_multiplier = self.get_fg_multiplier(fg_data['value'])
            result['base_multiplier'] = base_multiplier
            result['base_multiplier_reason'] = f"Fear & Greed = {fg_data['value']} ({fg_data['classification']})"
            result['expected_winrate'] = None
            result['signal_confidence'] = 'UNKNOWN'

        # Apply MA protection
        final_multiplier = base_multiplier
        protections_applied = []

        if ma_data['success'] and ma_data['above_threshold']:
            final_multiplier = min(final_multiplier, 0.5)
            protections_applied.append("200-MA Protection (>50% above)")

        # Get RSI
        rsi_data = self.calculate_rsi()
        result['rsi'] = rsi_data

        # Apply RSI protection
        if rsi_data['success'] and rsi_data['overheated']:
            final_multiplier = min(final_multiplier, 0.25)
            protections_applied.append("RSI Protection (>80)")

        # Get Volatility (informational - small adjustment in backtest)
        volatility_data = self.calculate_volatility()
        result['volatility'] = volatility_data

        # Get MVRV ratio (informational - provides additional context)
        mvrv_data = self.get_mvrv_ratio()
        result['mvrv'] = mvrv_data

        # Get Funding Rate (informational - shows leverage sentiment)
        funding_data = self.get_funding_rate()
        result['funding_rate'] = funding_data

        # Get Bear Market Classification (risk assessment)
        bear_classification = self.classify_bear_market_risk()
        result['bear_classification'] = bear_classification

        # Track opportunity signals (increases) and adjustments made
        opportunities_applied = []

        # === OPPORTUNITY SIGNALS (Increase allocation in favorable conditions) ===

        # 1. RSI Oversold Opportunity
        if rsi_data['success'] and rsi_data['rsi'] < 30:
            boost = 1.25
            final_multiplier = min(final_multiplier * boost, base_multiplier * 1.5)
            opportunities_applied.append(f"RSI Oversold Boost (RSI={rsi_data['rsi']:.1f})")

        # 2. Deep Discount to MA Opportunity
        if ma_data['success'] and ma_data['pct_above'] < -20:
            boost = 1.25
            final_multiplier = min(final_multiplier * boost, base_multiplier * 1.5)
            opportunities_applied.append(f"Deep Discount Boost ({ma_data['pct_above']:.1f}% below MA)")

        # 3. MVRV-Based Adjustments
        if mvrv_data['success']:
            mvrv_value = mvrv_data['mvrv']
            if mvrv_value < 1.0:
                # Extreme undervaluation - boost allocation
                mvrv_multiplier = 1.5
                final_multiplier = (final_multiplier + base_multiplier * mvrv_multiplier) / 2
                opportunities_applied.append(f"MVRV Extreme Undervalue (MVRV={mvrv_value:.2f})")
            elif mvrv_value < 1.5:
                # Undervalued - modest boost
                mvrv_multiplier = 1.25
                final_multiplier = (final_multiplier + base_multiplier * mvrv_multiplier) / 2
                opportunities_applied.append(f"MVRV Undervalued (MVRV={mvrv_value:.2f})")
            elif mvrv_value > 3.5:
                # Extreme overvaluation - reduce heavily
                final_multiplier = min(final_multiplier, 0.25)
                protections_applied.append(f"MVRV Extreme Overvalue (MVRV={mvrv_value:.2f})")
            elif mvrv_value > 2.5:
                # Overvalued - reduce allocation
                final_multiplier = min(final_multiplier, 0.5)
                protections_applied.append(f"MVRV Overvalued (MVRV={mvrv_value:.2f})")

        # 4. Extreme Volatility Opportunity (chaos creates bargains)
        if volatility_data['success'] and volatility_data['volatility'] > 80:
            boost = 1.15
            final_multiplier = min(final_multiplier * boost, base_multiplier * 1.5)
            opportunities_applied.append(f"High Volatility Opportunity ({volatility_data['volatility']:.1f}%)")

        # 5. Percentile Signal Integration
        try:
            current_values = self.get_current_metric_values()
            percentile_signals = percentile_calc.get_all_signals(current_values)

            increase_count = percentile_signals['summary']['increase_count']
            decrease_count = percentile_signals['summary']['decrease_count']

            # Strong buy signals (3+ increase signals)
            if increase_count >= 3:
                boost = 1.20
                final_multiplier = min(final_multiplier * boost, base_multiplier * 1.5)
                opportunities_applied.append(f"Percentile Buy Signals ({increase_count} metrics extreme)")

            # Strong sell signals (3+ decrease signals)
            elif decrease_count >= 3:
                reduction = 0.80
                final_multiplier = final_multiplier * reduction
                protections_applied.append(f"Percentile Caution Signals ({decrease_count} metrics extreme)")
        except Exception as e:
            # Percentile calculation failed - continue without it
            pass

        result['final_multiplier'] = final_multiplier
        result['protections_applied'] = protections_applied
        result['opportunities_applied'] = opportunities_applied
        result['recommended_amount'] = self.base_amount * final_multiplier
        result['success'] = True

        return result

    def get_current_metric_values(self) -> dict:
        """Get current values for all percentile-tracked metrics"""
        metric_values = {}

        try:
            # Get BTC price
            price_data = self.get_btc_price()
            if not price_data['success']:
                return {}

            current_price = price_data['price']

            # 1. Volatility
            vol_data = self.calculate_volatility()
            if vol_data['success']:
                metric_values['volatility'] = vol_data['volatility']

            # 2. Volume (most recent day's volume)
            prices_df = self.get_historical_prices(2)
            if len(prices_df) > 0:
                # Need to get volume from the data file directly
                data_path = Path(__file__).parent / "data" / "BTCUSDT_1d.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    df['date'] = pd.to_datetime(df['time_period_end']).dt.tz_localize(None)
                    df = df.sort_values('date')
                    if len(df) > 0:
                        metric_values['volume'] = float(df['volume'].iloc[-1])

            # 3. RSI
            rsi_data = self.calculate_rsi()
            if rsi_data['success']:
                metric_values['rsi'] = rsi_data['rsi']

            # 4. MA Distance
            ma_data = self.calculate_ma_200(current_price)
            if ma_data['success']:
                metric_values['ma_distance'] = ma_data['pct_above']

            # 5. Drawdown from ATH
            prices = self.get_historical_prices(5000)  # Get all available data
            if len(prices) > 0:
                ath = float(prices.max())
                drawdown_pct = ((current_price / ath) - 1) * 100
                metric_values['drawdown'] = drawdown_pct

            # Note: MVRV removed from percentile tracking
            # Dashboard displays real-time MVRV from API (not percentile-based)

        except Exception as e:
            print(f"Error getting metric values: {e}")

        return metric_values


# Global calculator instances
calculator = DCACalculator(base_amount=100.0)
percentile_calc = None  # Initialized on first request to avoid startup delay
mvrv_estimator = None  # Lazy-loaded MVRV estimator

def get_percentile_calculator():
    """Lazy-load percentile calculator (expensive initialization)"""
    global percentile_calc
    if percentile_calc is None:
        print("Initializing percentile calculator...")
        percentile_calc = PercentileCalculator()
    return percentile_calc

def get_mvrv_estimator():
    """Lazy-load MVRV estimator (requires historical data)"""
    global mvrv_estimator
    if mvrv_estimator is None:
        print("Initializing MVRV estimator...")
        prices = load_historical_prices()
        mvrv_estimator = MVRVEstimator(prices)
    return mvrv_estimator


@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return render_template('dca_dashboard.html')


@app.route('/api/recommendation')
def get_recommendation():
    """API endpoint for DCA recommendation"""
    try:
        base_amount = float(request.args.get('base_amount', 100.0))
        calculator.base_amount = base_amount
        recommendation = calculator.calculate_recommendation()

        # Convert datetime objects to strings for JSON serialization
        if 'timestamp' in recommendation and isinstance(recommendation['timestamp'], datetime):
            recommendation['timestamp'] = recommendation['timestamp'].isoformat()
        if 'fear_greed' in recommendation and 'timestamp' in recommendation['fear_greed']:
            if isinstance(recommendation['fear_greed']['timestamp'], datetime):
                recommendation['fear_greed']['timestamp'] = recommendation['fear_greed']['timestamp'].isoformat()

        return jsonify(recommendation)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/refresh')
def refresh_data():
    """Force refresh all cached data"""
    calculator.cache.clear()
    return jsonify({'success': True, 'message': 'Cache cleared'})


@app.route('/api/percentile_signals')
def get_percentile_signals():
    """API endpoint for percentile-based signals"""
    try:
        # Get percentile calculator
        perc_calc = get_percentile_calculator()

        if perc_calc.df is None:
            return jsonify({
                'success': False,
                'error': 'Historical data not available for percentile calculations'
            })

        # Get current metric values
        current_values = calculator.get_current_metric_values()

        if not current_values:
            return jsonify({
                'success': False,
                'error': 'Could not fetch current metric values'
            })

        # Get percentile signals for all metrics
        signals = perc_calc.get_all_signals(current_values)
        signals['success'] = True

        return jsonify(signals)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/myip')
def get_my_ip():
    """Helper endpoint to show your current IP address"""
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ',' in client_ip:
        client_ip = client_ip.split(',')[0].strip()

    return jsonify({
        'your_ip': client_ip,
        'message': f'Add this IP to ALLOWED_IPS environment variable on Render: {client_ip}'
    })


if __name__ == '__main__':
    # Get port from environment variable (for cloud deployment) or use 5000 for local
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'True') == 'True'

    print("\n" + "="*60)
    print("Bitcoin DCA Decision Dashboard")
    print("="*60)
    print(f"\nStarting web server on port {port}...")
    if port == 5000:
        print("Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug_mode)
