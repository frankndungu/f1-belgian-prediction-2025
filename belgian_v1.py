import pandas as pd
import numpy as np
import fastf1
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
        self.qualifying_data = None
        self.predictions = None
        self.spa_team_performance = {}
        self.wet_weather_specialists = [
            "Max Verstappen",
            "Lewis Hamilton",
            "George Russell",
            "Fernando Alonso",
        ]

    def load_local_data(self):
        """Load 2025 season data and qualifying results"""
        print("📂 Loading local race data...")

        self.current_season_data = pd.read_csv(
            "./data/compiled_f1_race_results_2025.csv"
        )
        self.current_season_data.columns = self.current_season_data.columns.str.strip()

        print(f"✅ Loaded {len(self.current_season_data)} race entries")
        self._clean_race_data()

        # Load qualifying data
        self._load_qualifying_data()

    def _load_qualifying_data(self):
        """Load and process qualifying data"""
        print("🏁 Loading qualifying data...")

        try:
            self.qualifying_data = pd.read_csv("./data/spa_qualis.csv")
            self.qualifying_data.columns = self.qualifying_data.columns.str.strip()

            # Clean driver names to match race data format
            self.qualifying_data["Driver"] = self.qualifying_data["Driver"].str.strip()

            print(f"✅ Loaded qualifying data for {len(self.qualifying_data)} drivers")
            print("🏁 Qualifying order:")
            for i, row in self.qualifying_data.iterrows():
                print(f"  {row['Position']:2d}. {row['Driver']} ({row['Team']})")

        except FileNotFoundError:
            print("❌ Qualifying data not found at ./data/spa_qualis.csv")
            self.qualifying_data = None
        except Exception as e:
            print(f"❌ Error loading qualifying data: {e}")
            self.qualifying_data = None

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
        """Use actual Spa weather data from today"""
        print("🌤️ Using actual Spa weather data...")

        # Actual Spa weather from screenshot
        self.weather_data = {
            "temperature_max": 17,
            "temperature_min": 12,
            "precipitation_probability": 60,
            "wind_speed": 10,
            "conditions": "wet",
        }

        print(
            f"  🌡️ Temperature: {self.weather_data['temperature_min']:.1f}°C - {self.weather_data['temperature_max']:.1f}°C"
        )
        print(
            f"  🌧️ Rain probability: {self.weather_data['precipitation_probability']:.0f}%"
        )
        print(f"  💨 Wind speed: {self.weather_data['wind_speed']:.1f} km/h")

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

            # Calculate qualifying vs race performance
            quali_vs_race_diff = self._calculate_quali_race_performance(driver)

            team = driver_races["team"].iloc[0]

            driver_stats.append(
                {
                    "driver": driver,
                    "team": team,
                    "avg_position": avg_position,
                    "total_points": total_points,
                    "dnf_rate": dnf_rate,
                    "podium_rate": podium_rate,
                    "quali_race_diff": quali_vs_race_diff,
                }
            )

        self.driver_performance = pd.DataFrame(driver_stats)
        print(f"✅ Computed performance for {len(self.driver_performance)} drivers")

    def _calculate_quali_race_performance(self, driver):
        """Calculate how well a driver typically performs in races vs qualifying"""
        # This is a simplified version - in reality you'd want historical quali vs race data
        # For now, return a default value, but this could be enhanced with more historical data
        return 0.0

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

    def get_driver_qualifying_position(self, driver_name):
        """Get qualifying position for a driver"""
        if self.qualifying_data is None:
            return 10  # Default middle position

        # Try exact match first
        match = self.qualifying_data[self.qualifying_data["Driver"] == driver_name]
        if not match.empty:
            return match.iloc[0]["Position"]

        # Try partial match (for name variations)
        for _, row in self.qualifying_data.iterrows():
            if (
                driver_name.split()[-1] in row["Driver"]
                or row["Driver"].split()[-1] in driver_name
            ):
                return row["Position"]

        return 10  # Default if not found

    def create_prediction_features(self, weather_scenario="actual"):
        """Create features for prediction model including qualifying data"""
        print(
            f"🔧 Creating features for {weather_scenario} scenario (with qualifying data)..."
        )

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

            # Qualifying position (very important!)
            quali_pos = self.get_driver_qualifying_position(driver_name)
            feature_vector.append(quali_pos)

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
        """Generate predictions for different weather scenarios with qualifying data"""
        print("🔮 Generating race predictions (incorporating qualifying results)...")

        scenarios = ["dry", "wet"] if scenario == "both" else [scenario]
        all_predictions = {}

        for weather_scenario in scenarios:
            print(f"  📊 Predicting for {weather_scenario} conditions...")

            features, drivers = self.create_prediction_features(weather_scenario)

            predictions = []

            for i, driver in enumerate(drivers):
                driver_perf = self.driver_performance[
                    self.driver_performance["driver"] == driver
                ].iloc[0]

                # Start with qualifying position as base
                quali_pos = self.get_driver_qualifying_position(driver)

                # Season form adjustment
                season_form_adjustment = (10 - driver_perf["avg_position"]) * 0.2

                # Spa-specific team performance
                spa_adjustment = (
                    self.spa_team_performance.get(driver_perf["team"], 10.0) - 10.0
                ) * 0.3

                # Weather adjustment
                weather_adjustment = (features[i][-1] - 1.0) * 2

                # DNF risk adjustment
                dnf_risk = driver_perf["dnf_rate"] * 3

                # Calculate predicted position
                predicted_pos = (
                    quali_pos
                    + season_form_adjustment
                    + spa_adjustment
                    + weather_adjustment
                    + dnf_risk
                )

                predicted_pos = max(1, min(20, predicted_pos))  # Clamp 1-20

                predictions.append(
                    {
                        "driver": driver,
                        "team": driver_perf["team"],
                        "qualifying_position": quali_pos,
                        "predicted_position": round(predicted_pos, 1),
                        "current_avg_position": round(driver_perf["avg_position"], 1),
                        "position_change": round(predicted_pos - quali_pos, 1),
                    }
                )

            predictions_df = pd.DataFrame(predictions).sort_values("predicted_position")
            all_predictions[weather_scenario] = predictions_df

        self.predictions = all_predictions
        return all_predictions

    def display_predictions(self):
        """Display predictions for both scenarios with qualifying comparison"""
        if not self.predictions:
            print("❌ No predictions available!")
            return

        print("\n" + "=" * 80)
        print("🏆 BELGIAN GRAND PRIX 2025 - RACE PREDICTIONS (WITH QUALIFYING)")
        print("=" * 80)
        print(f"📅 Date: July 27, 2025")
        print(f"🏁 Circuit: Spa-Francorchamps")
        print(
            f"🌤️ Forecast: {self.weather_data['temperature_min']:.1f}°C-{self.weather_data['temperature_max']:.1f}°C"
        )
        print(
            f"🌧️ Rain Probability: {self.weather_data['precipitation_probability']:.0f}%"
        )
        print("=" * 80)

        for scenario in self.predictions:
            print(f"\n🌧️ {scenario.upper()} CONDITIONS PREDICTION:")
            print("-" * 70)
            print(
                f"{'Pos':<4} {'Driver':<18} {'Team':<12} {'Quali':<6} {'Change':<7} {'Season Avg'}"
            )
            print("-" * 70)

            for i, row in self.predictions[scenario].iterrows():
                pos = int(row["predicted_position"])
                change = row["position_change"]
                change_str = f"{change:+.1f}" if change != 0 else "±0.0"

                print(
                    f"{pos:2d}.  {row['driver']:<18} {row['team']:<12} "
                    f"P{row['qualifying_position']:2d}   {change_str:<7} {row['current_avg_position']:4.1f}"
                )

        # Show biggest movers
        print(
            f"\n📈 BIGGEST EXPECTED MOVERS ({list(self.predictions.keys())[0].upper()}):"
        )
        print("-" * 50)
        first_scenario = list(self.predictions.values())[0]
        biggest_gainers = first_scenario.nsmallest(3, "position_change")
        biggest_losers = first_scenario.nlargest(3, "position_change")

        print("🔺 Biggest Gainers:")
        for _, row in biggest_gainers.iterrows():
            if row["position_change"] < 0:
                print(
                    f"  {row['driver']}: P{row['qualifying_position']} → P{int(row['predicted_position'])} ({row['position_change']:.1f})"
                )

        print("🔻 Biggest Losers:")
        for _, row in biggest_losers.iterrows():
            if row["position_change"] > 0:
                print(
                    f"  {row['driver']}: P{row['qualifying_position']} → P{int(row['predicted_position'])} ({row['position_change']:+.1f})"
                )

        print("\n" + "=" * 80)
        print("📝 Notes:")
        print("- Predictions incorporate actual qualifying positions")
        print("- Weather data from Open Meteo API")
        print("- Wet weather specialists get advantage in rain")
        print("- Historical Spa performance factored in")
        print("- Position change shows expected movement from qualifying")
        print("=" * 80)

    def save_predictions(self):
        """Save predictions to files"""
        if self.predictions:
            os.makedirs("models", exist_ok=True)
            for scenario, pred_df in self.predictions.items():
                filename = f"./models/belgian_gp_{scenario}_predictions_with_quali.csv"
                pred_df.to_csv(filename, index=False)
                print(f"💾 {scenario.capitalize()} predictions saved to {filename}")

    def run_full_analysis(self):
        """Run complete prediction pipeline"""
        print(
            "🚀 Starting Belgian GP 2025 Prediction Analysis (Enhanced with Qualifying)"
        )
        print("=" * 70)

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
    print("- Qualifying positions incorporated as primary factor")
    print("- 60% rain probability favors wet weather specialists")
    print("- Cool temperatures (12-17°C) expected")
    print("- Expected position changes calculated")
