# Belgian Grand Prix 2025 - F1 Race Prediction Model

## Overview

This repository contains a fully automated data pipeline and predictive modeling system built to forecast the final standings of the **2025 Belgian Grand Prix** at Spa-Francorchamps. The pipeline integrates real-time weather forecasts, historical circuit performance, and 2025 driver season form to simulate and output predictions for both dry and wet race conditions.

---

## Project Goals

- Integrate historical performance data from FastF1 API.
- Model dry vs wet conditions based on forecast.
- Build probabilistic position estimates using recent driver form.
- Output robust rankings under both weather scenarios.
- Use zero-login weather APIs for deployability and automation.

---

## Architecture & Flow

### Modules:

- **`BelgianGPPredictor`**: Core engine orchestrating the prediction pipeline.
- **`load_weather()`**: Fetches race-day weather from Open Meteo.
- **`load_driver_data()`**: Compiles 2025 season performance.
- **`load_spa_history()`**: Builds historical averages per driver at Spa.
- **`predict_positions()`**: Generates scenario-based prediction output.
- **`save_predictions()`**: Saves model results into CSVs by weather condition.

### Output

- Dry race predictions: `./models/belgian_gp_dry_predictions.csv`
- Wet race predictions: `./models/belgian_gp_wet_predictions.csv`

---

## Weather Integration

- **Provider**: [Open Meteo](https://open-meteo.com/)
- **Access**: No sign-up or authentication required
- **Forecast Variables**: Max/Min Temperature, Rain Probability
- **Logic**:
  - If rain probability > 30%, use **wet** scenario
  - Otherwise, default to **dry** conditions

---

## Prediction Methodology

### Data Sources:

- **FastF1**: Season telemetry and event-level results
- **Custom Driver Form Model**: Averaged finishing positions over the past 3 races
- **Circuit History**: Driver-specific performance at Spa (2018–2023)

### Scoring:

- Drivers are scored on a composite average:
  - 60% recent form
  - 30% historical Spa result
  - 10% weather-adjusted modifier (for wet specialists)
- Final output includes predicted position and internal probability ranking

---

## Prediction Output

```
Date: July 27, 2025
Circuit: Spa-Francorchamps
Forecast: 13.0°C–20.4°C
Rain Probability: 39%
```

### DRY CONDITIONS (Top 10 excerpt)

```
 1. Oscar Piastri         (McLaren)       [Avg: 2.6]
 2. Lando Norris          (McLaren)       [Avg: 3.2]
 3. George Russell        (Mercedes)      [Avg: 4.9]
 4. Max Verstappen        (Red Bull)      [Avg: 4.9]
 5. Charles Leclerc       (Ferrari)       [Avg: 5.3]
 6. Lewis Hamilton        (Ferrari)       [Avg: 6.0]
```

### WET CONDITIONS (Top 10 excerpt)

```
 1. Oscar Piastri         (McLaren)       [Avg: 2.6]
 2. Lando Norris          (McLaren)       [Avg: 3.2]
 3. George Russell        (Mercedes)      [Avg: 4.9]
 4. Max Verstappen        (Red Bull)      [Avg: 4.9]
 5. Charles Leclerc       (Ferrari)       [Avg: 5.3]
 6. Lewis Hamilton        (Ferrari)       [Avg: 6.0]
```

---

## Developer Notes

- Codebase is modular and easily extensible
- Designed for quick iteration with new circuits or seasons
- Easily replaceable weather API (wrapper pattern used)
- Potential for containerization for full automation (e.g. CRON + GitHub Actions)
- Future upgrade: use telemetry data (braking, throttle, etc.) as ML features

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python belgian_gp_predict.py

# Results will be saved in ./models/
```

---

## License

MIT License

---
