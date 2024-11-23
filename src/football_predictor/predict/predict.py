import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

import asyncio
import aiohttp

from understat import Understat

def get_betting_suggestion(pred_supremacy, home_team, away_team):
    """
    Suggest a highly conservative betting market selection using Asian handicaps, including quarter handicaps.

    :param pred_supremacy: The predicted supremacy value.
    :param home_team: The name of the home team.
    :param away_team: The name of the away team.
    :return: A string suggesting the bet to place.
    """
    # Define thresholds for conservative suggestions with quarter handicaps
    if pred_supremacy > 2.0:
        # Strong home team superiority
        return f"{home_team} -0.5 (Win or Asian Handicap)"
    elif 1.75 < pred_supremacy <= 2.0:
        # Moderate home team superiority
        return f"{home_team} -0.25 (Asian Handicap)"
    elif 1.5 < pred_supremacy <= 1.75:
        # Slightly strong home team superiority
        return f"{home_team} +0.5 (Double Chance)"
    elif 1.25 < pred_supremacy <= 1.5:
        # Slight home team superiority
        return f"{home_team} +0.25 (Asian Handicap)"
    elif 0.75 < pred_supremacy <= 1.25:
        # Marginal home team superiority
        return f"{home_team} +1.0 (Asian Handicap)"
    elif 0.5 < pred_supremacy <= 0.75:
        # Very marginal home team superiority
        return f"{home_team} +1.25 (Asian Handicap)"
    elif -0.5 <= pred_supremacy <= 0.5:
        # Essentially even match
        return "No Bet (Uncertain Supremacy)"
    elif -0.75 <= pred_supremacy < -0.5:
        # Very marginal away team superiority
        return f"{away_team} +1.25 (Asian Handicap)"
    elif -1.25 <= pred_supremacy < -0.75:
        # Marginal away team superiority
        return f"{away_team} +1.0 (Asian Handicap)"
    elif -1.5 <= pred_supremacy < -1.25:
        # Slight away team superiority
        return f"{away_team} +0.25 (Asian Handicap)"
    elif -1.75 <= pred_supremacy < -1.5:
        # Slightly strong away team superiority
        return f"{away_team} +0.5 (Double Chance)"
    elif -2.0 <= pred_supremacy < -1.75:
        # Moderate away team superiority
        return f"{away_team} -0.25 (Asian Handicap)"
    else:
        # Strong away team superiority
        return f"{away_team} -0.5 (Win or Asian Handicap)"

async def predict_upcoming_games(model):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        current_season = await understat.get_league_fixtures("epl", 2024)
        next_gw = current_season[:10]

        predictions_df = build_features(next_gw)
        feature_columns = predictions_df.columns.tolist()
        columns_to_drop = ['result', 'goal_diff', 'xG_diff', 'supremacy', 'home_xG_overperformance', 'away_xG_overperformance', 'home_xG_ratio', 'away_xG_ratio', 'datetime', 'season', 'home_team', 'away_team', 'match_id', 'home_goals', 'away_goals', 'home_xG', 'away_xG']
        feature_columns = [col for col in feature_columns if col not in columns_to_drop]
        print("feature_columns:", feature_columns)

        # Prepare features for prediction
        X_pred = predictions_df[feature_columns].values
        print("Predict shape")
        print(X_pred.shape)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_pred)
        print("Predict features:")
        print(X_train_scaled)
        predictions = model.predict(X_train_scaled)

        print("\n🎯 Upcoming Fixture Predictions")
        print("=" * 120)
        print(f"{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Pred Supremacy':>15} {'Betting Suggestion':>40}")
        print("-" * 120)

        for i, (_, match) in enumerate(predictions_df.iterrows()):
            date_str = pd.to_datetime(match['datetime']).strftime('%Y-%m-%d')
            pred_supremacy = predictions[i][0]
            betting_suggestion = get_betting_suggestion(pred_supremacy, match['home_team'], match['away_team'])

            print(f"{date_str:<12} {match['home_team']:<25} {match['away_team']:<25} "
                  f"{pred_supremacy:>15.2f} {betting_suggestion:>40}")


def build_features(next_gw):
    # Load historical data
    hist_df = pd.read_csv('../data/output/engineered_football_data.csv')
    hist_df['datetime'] = pd.to_datetime(hist_df['datetime'])

    # Take the last row as a template - it will have all our engineered features
    template_row = hist_df.iloc[-1].copy()

    # Create DataFrame for upcoming fixtures using the template
    upcoming_df = pd.DataFrame([template_row for _ in next_gw])
    upcoming_df.reset_index(drop=True, inplace=True)

    # Update basic info for each match
    for idx, match in enumerate(next_gw):
        upcoming_df.loc[idx, 'datetime'] = pd.to_datetime(match['datetime'])
        upcoming_df['day_of_week'] = upcoming_df['datetime'].dt.dayofweek.astype(int)
        upcoming_df['month'] = upcoming_df['datetime'].dt.month.astype(int)
        upcoming_df['is_weekend'] = upcoming_df['day_of_week'].isin([5, 6]).astype(int)
        upcoming_df.loc[idx, 'season'] = '2024'
        upcoming_df.loc[idx, 'home_team'] = match['h']['title']
        upcoming_df.loc[idx, 'away_team'] = match['a']['title']

        # Get latest stats for home team
        home_last = hist_df[hist_df['home_team'] == match['h']['title']].iloc[-1]
        away_last = hist_df[hist_df['away_team'] == match['a']['title']].iloc[-1]
        try:
            h2h_last = hist_df[
                (hist_df['home_team'] == match['h']['title']) &
                (hist_df['away_team'] == match['a']['title'])
                ].iloc[-1]
        except IndexError:
            # Define a DataFrame of zeros with the same columns as hist_df
            h2h_last = pd.Series(0, index=hist_df.columns)

        # Update home team features
        home_cols = [col for col in hist_df.columns if col.startswith('home_')]
        for col in home_cols:
            upcoming_df.loc[idx, col] = home_last[col]

        # Update away team features
        away_cols = [col for col in hist_df.columns if col.startswith('away_')]
        for col in away_cols:
            upcoming_df.loc[idx, col] = away_last[col]

        # Update h2h features
        h2h_cols = [col for col in hist_df.columns if col.startswith('h2h_')]
        for col in h2h_cols:
            upcoming_df.loc[idx, col] = h2h_last[col]

        upcoming_df.loc[idx, 'elo_diff'] = upcoming_df.loc[idx, 'home_elo'] - upcoming_df.loc[idx, 'away_elo']

    return upcoming_df



if __name__ == "__main__":
    print("""
        ⚽️ ==================================================== ⚽️

             _____          _           ____            _ 
            |  ___|__  ___ | |_ _   _  |  _ \ _ __ ___| |
            | |_ / _ \/ _ \| __| | | | | |_) | '__/ _ \ |
            |  _|  __/ (_) | |_| |_| | |  __/| | |  __/ |
            |_|  \___|\___/ \__|\__, | |_|   |_|  \___|_|
                                |___/                      

                 👟 Lewis's Supremacy Predictor 👟                    

        ⚽️ ==================================================== ⚽️

        """)

    # Load the model
    model = load_model("../model/best_model.h5")
    print("✓ Model loaded successfully")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(predict_upcoming_games(model))
