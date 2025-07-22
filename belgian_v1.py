import pandas as pd
import numpy as np
import requests
import fastf1
from datetime import datetime, date
from sklearn.ensemble import RandomForestRegressor
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("cache", exist_ok=True)

fastf1.Cache.enable_cache("cache/")


class BelgianGPPredictor:
    def __init__(self):
        self.current_season_data = None
        self.driver_performance = None
        self.weather_data = None
        self.predictions = None
        self.spa_team_performance = {}
        self.wet_weather_specialists = [
            "Max Verstappen",
            "Lewis Hamilton",
            "George Russell",
            "Fernando Alonso",
        ]

    def load_local_data(self):
        """Load 2025 season data"""
        print("📂 Loading local race data...")

        self.current_season_data = pd.read_csv(
            "./data/compiled_f1_race_results_2025.csv"
        )
        self.current_season_data.columns = self.current_season_data.columns.str.strip()

        print(f"✅ Loaded {len(self.current_season_data)} race entries")
        self._clean_race_data()

    def _clean_race_data(self):
        """Clean race data"""

        def time_to_seconds(time_str):
            if pd.isna(time_str) or time_str == "DNF":
                return np.nan
            if "+" in str(time_str):
                if "Lap" in str(time_str):
                    return 3600
                else:
                    return float(str(time_str).replace("+", "").replace("s", ""))
            return 0

        self.current_season_data["time_gap_seconds"] = self.current_season_data[
            "time"
        ].apply(time_to_seconds)

    def get_weather_forecast(self):
        """Get weather forecast using Open Meteo API"""
        print("🌤️ Fetching weather forecast from Open Meteo API...")

        # Spa coordinates
        lat, lon = 50.4372, 5.9714
        race_date = "2025-07-27"

        try:
            # Open Meteo API call
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
                "start_date": race_date,
                "end_date": race_date,
                "timezone": "Europe/Brussels",
            }

            response = requests.get(url, params=params)
            data = response.json()

            if "daily" in data:
                daily = data["daily"]
                self.weather_data = {
                    "temperature_max": daily["temperature_2m_max"][0],
                    "temperature_min": daily["temperature_2m_min"][0],
                    "precipitation_sum": daily["precipitation_sum"][0],
                    "precipitation_probability": daily["precipitation_probability_max"][
                        0
                    ],
                    "wind_speed": daily["wind_speed_10m_max"][0],
                    "conditions": (
                        "wet"
                        if daily["precipitation_probability_max"][0] > 50
                        else "dry"
                    ),
                }

                print(
                    f"  🌡️ Temperature: {self.weather_data['temperature_min']:.1f}°C - {self.weather_data['temperature_max']:.1f}°C"
                )
                print(
                    f"  🌧️ Rain probability: {self.weather_data['precipitation_probability']:.0f}%"
                )
                print(f"  💨 Wind speed: {self.weather_data['wind_speed']:.1f} km/h")
            else:
                raise Exception("API response format unexpected")

        except Exception as e:
            print(f"  ❌ Weather fetch failed: {e}")
            # Fallback weather data
            self.weather_data = {
                "temperature_max": 22,
                "temperature_min": 15,
                "precipitation_probability": 35,
                "wind_speed": 10,
                "conditions": "mixed",
            }

    def analyze_current_season_performance(self):
        """Analyze 2025 season performance"""
        print("📈 Analyzing 2025 season performance...")

        driver_stats = []
        for driver in self.current_season_data["driver"].unique():
            driver_races = self.current_season_data[
                self.current_season_data["driver"] == driver
            ]

            avg_position = driver_races["position"].mean()
            total_points = driver_races["points"].sum()
            dnf_rate = (driver_races["time"] == "DNF").sum() / len(driver_races)
            podium_rate = (driver_races["position"] <= 3).sum() / len(driver_races)

            team = driver_races["team"].iloc[0]

            driver_stats.append(
                {
                    "driver": driver,
                    "team": team,
                    "avg_position": avg_position,
                    "total_points": total_points,
                    "dnf_rate": dnf_rate,
                    "podium_rate": podium_rate,
                }
            )

        self.driver_performance = pd.DataFrame(driver_stats)
        print(f"✅ Computed performance for {len(self.driver_performance)} drivers")

    def fetch_spa_historical_data(self):
        """Fetch real Spa data using FastF1"""
        print("🏁 Fetching historical Spa data...")

        spa_performance = {}
        for year in [2022, 2023, 2024]:
            try:
                session = fastf1.get_session(year, "Belgium", "R")
                session.load()
                results = session.results

                for _, row in results.iterrows():
                    team = row.get("TeamName", "Unknown")
                    position = row.get("Position", 20)

                    if team not in spa_performance:
                        spa_performance[team] = []
                    spa_performance[team].append(position)

                print(f"  ✅ {year} Spa data loaded")
            except Exception as e:
                print(f"  ❌ {year} failed: {e}")

        # Calculate average positions
        for team, positions in spa_performance.items():
            self.spa_team_performance[team] = np.mean(positions)

        print(f"✅ Historical data for {len(self.spa_team_performance)} teams")

    def create_prediction_features(self, weather_scenario="actual"):
        """Create features for prediction model"""
        print(f"🔧 Creating features for {weather_scenario} scenario...")

        features = []
        drivers = []

        # Determine weather condition
        if weather_scenario == "wet":
            is_wet = True
        elif weather_scenario == "dry":
            is_wet = False
        else:  # actual
            is_wet = self.weather_data["precipitation_probability"] > 50

        for _, driver_perf in self.driver_performance.iterrows():
            driver_name = driver_perf["driver"]
            team = driver_perf["team"]

            # Base features
            feature_vector = [
                driver_perf["avg_position"],
                driver_perf["total_points"] / 100,  # Normalize
                driver_perf["dnf_rate"],
                driver_perf["podium_rate"],
            ]

            # Spa team performance
            spa_performance = self.spa_team_performance.get(team, 10.0)
            feature_vector.append(spa_performance)

            # Weather impact
            weather_impact = 1.0
            if is_wet and driver_name in self.wet_weather_specialists:
                weather_impact = 0.85  # Better in wet
            elif is_wet:
                weather_impact = 1.15  # Worse in wet

            feature_vector.append(weather_impact)

            features.append(feature_vector)
            drivers.append(driver_name)

        return np.array(features), drivers

    def make_predictions(self, scenario="both"):
        """Generate predictions for different weather scenarios"""
        print("🔮 Generating race predictions...")

        scenarios = ["dry", "wet"] if scenario == "both" else [scenario]
        all_predictions = {}

        for weather_scenario in scenarios:
            print(f"  📊 Predicting for {weather_scenario} conditions...")

            features, drivers = self.create_prediction_features(weather_scenario)

            # Simple prediction based on current form + Spa performance + weather
            predictions = []

            for i, driver in enumerate(drivers):
                driver_perf = self.driver_performance[
                    self.driver_performance["driver"] == driver
                ].iloc[0]

                base_position = driver_perf["avg_position"]
                spa_adjustment = (
                    self.spa_team_performance.get(driver_perf["team"], 10.0)
                    - base_position
                ) * 0.3
                weather_adjustment = features[i][-1] - 1.0  # Weather impact

                predicted_pos = base_position + spa_adjustment + weather_adjustment * 2
                predicted_pos = max(1, min(20, predicted_pos))  # Clamp 1-20

                predictions.append(
                    {
                        "driver": driver,
                        "team": driver_perf["team"],
                        "predicted_position": round(predicted_pos, 1),
                        "current_avg_position": round(base_position, 1),
                    }
                )

            predictions_df = pd.DataFrame(predictions).sort_values("predicted_position")
            all_predictions[weather_scenario] = predictions_df

        self.predictions = all_predictions
        return all_predictions

    def display_predictions(self):
        """Display predictions for both scenarios"""
        if not self.predictions:
            print("❌ No predictions available!")
            return

        print("\n" + "=" * 70)
        print("🏆 BELGIAN GRAND PRIX 2025 - RACE PREDICTIONS")
        print("=" * 70)
        print(f"📅 Date: July 27, 2025")
        print(f"🏁 Circuit: Spa-Francorchamps")
        print(
            f"🌤️ Forecast: {self.weather_data['temperature_min']:.1f}°C-{self.weather_data['temperature_max']:.1f}°C"
        )
        print(
            f"🌧️ Rain Probability: {self.weather_data['precipitation_probability']:.0f}%"
        )
        print("=" * 70)

        for scenario in self.predictions:
            print(f"\n🌧️ {scenario.upper()} CONDITIONS PREDICTION:")
            print("-" * 50)

            for i, row in self.predictions[scenario].iterrows():
                pos = int(row["predicted_position"])
                print(
                    f"{pos:2d}. {row['driver']:<18} ({row['team']:<12}) [Avg: {row['current_avg_position']:4.1f}]"
                )

        print("\n" + "=" * 70)
        print("📝 Notes:")
        print("- Weather data from Open Meteo API")
        print("- Wet weather specialists get advantage in rain")
        print("- Historical Spa performance factored in")
        print("- Both scenarios prepared for race day")
        print("=" * 70)

    def save_predictions(self):
        """Save predictions to files"""
        if self.predictions:
            os.makedirs("models", exist_ok=True)
            for scenario, pred_df in self.predictions.items():
                filename = f"./models/belgian_gp_{scenario}_predictions.csv"
                pred_df.to_csv(filename, index=False)
                print(f"💾 {scenario.capitalize()} predictions saved to {filename}")

    def run_full_analysis(self):
        """Run complete prediction pipeline"""
        print("🚀 Starting Belgian GP 2025 Prediction Analysis")
        print("=" * 60)

        # Core pipeline
        self.load_local_data()
        self.fetch_spa_historical_data()
        self.get_weather_forecast()
        self.analyze_current_season_performance()
        self.make_predictions("both")
        self.display_predictions()
        self.save_predictions()

        print("\n🏁 Analysis complete!")
        return self.predictions


if __name__ == "__main__":
    predictor = BelgianGPPredictor()
    predictions = predictor.run_full_analysis()

    print("\n📊 Key Insights:")
    print("- Weather scenarios prepared for robustness")
    print("- Open Meteo API provides real-time forecasts")
    print("- Wet weather specialists identified")
    print("- Spa circuit characteristics included")
