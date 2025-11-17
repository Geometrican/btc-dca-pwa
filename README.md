# Bitcoin DCA Assistant PWA

A Progressive Web App that provides Bitcoin Dollar Cost Averaging recommendations based on real-time market sentiment and the Fear & Greed Index.

![PWA](https://img.shields.io/badge/PWA-Enabled-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![Flask](https://img.shields.io/badge/Flask-3.1-red)

## Features

- **Real-time Market Data**: Fear & Greed Index, BTC price, MVRV Z-Score
- **Smart DCA Recommendations**: Position size multipliers based on market sentiment
- **Bear Market Detection**: Identifies accumulation phases and risk levels
- **PWA Ready**: Install on Android/iOS as a native-like app
- **Offline Support**: Service worker caching for faster loads
- **Responsive Design**: Works on desktop, tablet, and mobile

## Live Demo

Visit the deployed app: [Coming Soon]

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/btc-dca-pwa.git
cd btc-dca-pwa

# Install dependencies
pip install -r requirements.txt

# Run the app
python dca_dashboard.py

# Open browser to http://localhost:5000
```

### Install as PWA

1. Visit the app URL on your mobile device
2. **Android**: Tap menu (⋮) → "Add to Home Screen"
3. **iOS**: Tap Share → "Add to Home Screen"
4. App opens fullscreen like a native app!

## Deploy to Render.com (Free)

1. Fork this repository
2. Sign up at [Render.com](https://render.com)
3. Create new Web Service
4. Connect your GitHub repo
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python dca_dashboard.py`
   - **Plan**: Free
6. Deploy! You'll get a URL like `https://btc-dca-pwa.onrender.com`

## How It Works

The dashboard calculates DCA recommendations using:

1. **Fear & Greed Index** (0-100)
   - Extreme Fear (0-25): Maximum aggression
   - Fear (25-45): Increased buying
   - Neutral (45-55): Standard DCA
   - Greed (55+): Reduce position sizes

2. **Bear Market Phases**
   - Tracks months from peak
   - Analyzes drawdown percentage
   - Identifies optimal accumulation zones

3. **MVRV Z-Score**
   - Historical value analysis
   - Percentile rankings
   - Market cycle positioning

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **APIs**:
  - Fear & Greed Index API
  - Yahoo Finance (BTC price data)
- **PWA**: Service Worker, Web App Manifest
- **Hosting**: Render.com (free tier)

## Project Structure

```
btc-dca-pwa/
├── dca_dashboard.py          # Main Flask application
├── percentile_calculator.py  # Historical percentile calculations
├── mvrv_estimator.py         # MVRV Z-Score estimation
├── requirements.txt          # Python dependencies
├── templates/
│   └── dca_dashboard.html    # Main dashboard UI
├── static/
│   ├── manifest.json         # PWA manifest
│   ├── service-worker.js     # Service worker for offline support
│   ├── icon-192.png          # App icon (192x192)
│   └── icon-512.png          # App icon (512x512)
└── README.md
```

## API Endpoints

- `GET /` - Main dashboard UI
- `GET /api/dca` - Current DCA recommendations (JSON)
- `GET /api/percentile-signals` - Historical percentile signals (JSON)

## Environment Variables

- `PORT` - Port number (default: 5000)
- `DEBUG` - Debug mode (default: True)

## Screenshots

[Add screenshots of your PWA here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Disclaimer

This tool is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.

## Author

Built with Python, Flask, and dedication to smart DCA strategies.

---

**Star this repo** if you find it useful! ⭐
