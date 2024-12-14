import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

import asyncio
import aiohttp

from understat import Understat

from football_predictor.data.features import features_order


def get_betting_suggestion(pred_supremacy, home_team, away_team):
    """
    Suggest a highly conservative betting market selection using only positive Asian handicaps
    aligned with the supremacy direction.

    :param pred_supremacy: The predicted supremacy value.
    :param home_team: The name of the home team.
    :param away_team: The name of the away team.
    :return: A string suggesting the bet to place.
    """
    # Define thresholds for conservative suggestions with positive handicaps only
    if pred_supremacy > 2.0:
        # Very strong home team superiority
        return f"{home_team} +0.25 (Asian Handicap)"
    elif 1.5 < pred_supremacy <= 2.0:
        # Strong home team superiority
        return f"{home_team} +0.5 (Asian Handicap)"
    elif 1.0 < pred_supremacy <= 1.5:
        # Moderate home team superiority
        return f"{home_team} +1.0 (Asian Handicap)"
    elif 0.5 < pred_supremacy <= 1.0:
        # Slight home team superiority
        return f"{home_team} +1.25 (Asian Handicap)"
    elif 0.0 < pred_supremacy <= 0.5:
        # Very marginal home team superiority
        return f"{home_team} +1.5 (Asian Handicap)"
    elif -0.5 <= pred_supremacy <= 0.0:
        # Very marginal away team superiority
        return f"{away_team} +1.5 (Asian Handicap)"
    elif -1.0 <= pred_supremacy < -0.5:
        # Slight away team superiority
        return f"{away_team} +1.25 (Asian Handicap)"
    elif -1.5 <= pred_supremacy < -1.0:
        # Moderate away team superiority
        return f"{away_team} +1.0 (Asian Handicap)"
    elif -2.0 <= pred_supremacy < -1.5:
        # Strong away team superiority
        return f"{away_team} +0.5 (Asian Handicap)"
    else:
        # Very strong away team superiority
        return f"{away_team} +0.25 (Asian Handicap)"

async def predict_upcoming_games(model):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        current_season = await understat.get_league_fixtures("epl", 2024)
        next_gw = current_season[:10]

        predictions_df = build_features(next_gw)
        feature_columns = predictions_df.columns.tolist()
        columns_to_drop = ['result', 'goal_diff', 'xG_diff', 'supremacy', 'home_xG_overperformance', 'away_xG_overperformance', 'home_xG_ratio', 'away_xG_ratio', 'datetime', 'season', 'home_team', 'away_team', 'match_id', 'home_goals', 'away_goals', 'home_xG', 'away_xG']
        feature_columns = [col for col in feature_columns if col not in columns_to_drop]

        # Prepare features for prediction
        sorted_predictions_df = predictions_df[features_order]
        print("feature_columns:", sorted_predictions_df.columns)
        X_pred = sorted_predictions_df.values
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

    # Initialize DataFrame for upcoming fixtures
    upcoming_df = pd.DataFrame([{
        'datetime': pd.to_datetime(match['datetime']),
        'season': '2024',
        'home_team': match['h']['title'],
        'away_team': match['a']['title']
    } for match in next_gw])

    # Add time-based features
    upcoming_df['day_of_week'] = upcoming_df['datetime'].dt.dayofweek.astype(int)
    upcoming_df['month'] = upcoming_df['datetime'].dt.month.astype(int)
    upcoming_df['is_weekend'] = upcoming_df['day_of_week'].isin([5, 6]).astype(int)

    # Group features by type
    feature_groups = {
        'team_specific': {
            'home': [col for col in hist_df.columns if
                     col.startswith('home_') and not col.endswith(('team', 'goals', 'xG'))],
            'away': [col for col in hist_df.columns if
                     col.startswith('away_') and not col.endswith(('team', 'goals', 'xG'))]
        },
        'elo': ['home_elo', 'away_elo'],
        'position': ['home_position', 'away_position'],
        'h2h': [col for col in hist_df.columns if col.startswith('h2h_')],
        'exact_h2h': [col for col in hist_df.columns if col.startswith('exact_')]
    }

    # Process each upcoming match
    for idx, row in upcoming_df.iterrows():
        # Get latest team data
        home_last = hist_df[hist_df['home_team'] == row['home_team']].iloc[-1]
        away_last = hist_df[hist_df['away_team'] == row['away_team']].iloc[-1]

        # Get h2h data
        h2h_mask = (
                ((hist_df['home_team'] == row['home_team']) & (hist_df['away_team'] == row['away_team'])) |
                ((hist_df['home_team'] == row['away_team']) & (hist_df['away_team'] == row['home_team']))
        )
        h2h_matches = hist_df[h2h_mask]

        exact_h2h_mask = (
                (hist_df['home_team'] == row['home_team']) &
                (hist_df['away_team'] == row['away_team'])
        )
        exact_h2h_matches = hist_df[exact_h2h_mask]

        # Add team-specific features
        for col in feature_groups['team_specific']['home']:
            upcoming_df.loc[idx, col] = home_last[col]
        for col in feature_groups['team_specific']['away']:
            upcoming_df.loc[idx, col] = away_last[col]

        # Add Elo and position features
        for col in feature_groups['elo'] + feature_groups['position']:
            if col.startswith('home_'):
                upcoming_df.loc[idx, col] = home_last[col]
            else:
                upcoming_df.loc[idx, col] = away_last[col]

        # Add h2h features
        if len(h2h_matches) > 0:
            for col in feature_groups['h2h']:
                upcoming_df.loc[idx, col] = h2h_matches.iloc[-1][col]
        else:
            for col in feature_groups['h2h']:
                upcoming_df.loc[idx, col] = 0

        # Add exact h2h features
        if len(exact_h2h_matches) > 0:
            for col in feature_groups['exact_h2h']:
                upcoming_df.loc[idx, col] = exact_h2h_matches.iloc[-1][col]
        else:
            for col in feature_groups['exact_h2h']:
                upcoming_df.loc[idx, col] = 0

        # Calculate derived features
        upcoming_df.loc[idx, 'elo_diff'] = upcoming_df.loc[idx, 'home_elo'] - upcoming_df.loc[idx, 'away_elo']
        upcoming_df.loc[idx, 'position_diff'] = upcoming_df.loc[idx, 'away_position'] - upcoming_df.loc[
            idx, 'home_position']

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
