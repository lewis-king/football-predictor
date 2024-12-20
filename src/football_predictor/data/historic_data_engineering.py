import asyncio
import glob
import json

import numpy as np
import pandas as pd
import os
from datetime import datetime

import aiohttp

from understat import Understat

def extract_season(filename):
    # Extract season from filename (e.g., 'epl_league_results_2017.json' -> '2017')
    return int(filename.split('_')[-1].split('.')[0])

async def pre_process():
    async with aiohttp.ClientSession() as session:
        if resources_folder is None:
            print("Error: Could not find the 'resources' folder.")
        else:
            understat_folder = os.path.join(resources_folder, 'historic', 'understat')

            all_data = []

            # Get all JSON file paths and sort them
            file_paths = sorted(glob.glob(os.path.join(understat_folder, '*.json')))

            for file_path in file_paths:
                print(f"Processing file: {file_path}")
                filename = os.path.basename(file_path)
                season = extract_season(filename)
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                processed_data = [process_match_data(match, season) for match in json_data]
                all_data.extend(processed_data)

            all_league_results_df = pd.DataFrame(all_data)

            print(all_league_results_df.tail())
            understat = Understat(session)
            current_season = await understat.get_league_results("epl", 2024)
            #print(json.dumps(current_season, indent=2))
            current_season_df = json_to_dataframe(current_season, 2024)
            all_league_results_df = pd.concat([all_league_results_df, current_season_df], ignore_index=True)

            print(all_league_results_df.tail(5))

            engineered_features = engineer_features(all_league_results_df)
            engineered_features = round_dataframe(engineered_features)
            print(type(engineered_features))
            print(engineered_features.head(200))

            # Save the engineered DataFrame to a CSV file
            output_directory = 'output'  # Specify your desired output directory
            os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
            output_file = os.path.join(output_directory, 'engineered_football_data.csv')

            engineered_features.to_csv(output_file, index=False)
            print(f"\nEngineered data saved to: {output_file}")


def engineer_features(df):
    print("Initial DataFrame columns:", df.columns)
    print("Initial DataFrame shape:", df.shape)
    print("Initial DataFrame types:\n", df.dtypes)

    df = df.sort_values(['season', 'datetime'])

    # Ensure we're using the correct column names
    home_goals_col = 'home_goals'
    away_goals_col = 'away_goals'
    home_xg_col = 'home_xG'
    away_xg_col = 'away_xG'

    # Check if the required columns exist
    required_cols = [home_goals_col, away_goals_col, home_xg_col, away_xg_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df['result'] = np.sign(df[home_goals_col] - df[away_goals_col]).astype(int)

    stats_cols = [home_goals_col, away_goals_col]
    df = add_team_stats(df, 'home_team', stats_cols)
    df = add_team_stats(df, 'away_team', stats_cols)

    df = add_recent_form(df, 'home_team', 'result')
    df = add_recent_form(df, 'away_team', 'result')

    df = add_head_to_head_stats(df, 'home_team', 'away_team', stats_cols)

    df['goal_diff'] = (df[home_goals_col] - df[away_goals_col]).astype(int)
    df['xG_diff'] = (df[home_xg_col] - df[away_xg_col]).astype(float)
    df['supremacy'] = (df[home_goals_col] - df[away_goals_col]).astype(float)

    df['day_of_week'] = df['datetime'].dt.dayofweek.astype(int)
    df['month'] = df['datetime'].dt.month.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df = calculate_league_positions(df)

    df = calculate_elo_ratings(df, use_xG=True)

    df['position_diff'] = (df['away_position'] - df['home_position']).astype(int)
    df['elo_diff'] = (df['home_elo'] - df['away_elo']).astype(float)

    df = add_xG_features(df)

    print("Final DataFrame columns:", df.columns)
    print("Final DataFrame shape:", df.shape)
    print("Final DataFrame types:\n", df.dtypes)

    return df

def process_match_data(match, season):
    return {
        'season': season,
        'datetime': datetime.strptime(match['datetime'], '%Y-%m-%d %H:%M:%S'),
        'home_team': match['h']['title'],
        'away_team': match['a']['title'],
        'home_goals': int(match['goals']['h']),
        'away_goals': int(match['goals']['a']),
        'home_xG': float(match['xG']['h']),
        'away_xG': float(match['xG']['a']),
    }


def json_to_dataframe(json_data, season):
    processed_data = [process_match_data(match, season) for match in json_data]
    return pd.DataFrame(processed_data)


def find_resources_folder(start_path, target_folder='resources'):
    current_path = start_path
    while True:
        if os.path.exists(os.path.join(current_path, target_folder)):
            return os.path.join(current_path, target_folder)
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:  # We've reached the root directory
            return None
        current_path = parent_path

def add_team_stats(df, team_col, stats_cols, window_sizes=[3, 5, 10]):
    result_df = df.copy()
    for size in window_sizes:
        for col in stats_cols:
            # Calculate rolling mean for this specific window size
            rolling_stats = (df.groupby(['season', team_col])[col]
                             .rolling(window=size, min_periods=1)
                             .mean()
                             .reset_index(level=[0, 1], drop=True))

            new_col = f'{team_col}_{col}_last_{size}'
            result_df[new_col] = rolling_stats

    return result_df


def add_recent_form(df, team_col, result_col, window_sizes=[3, 5, 10]):
    result_df = df.copy()

    print(f"\nCalculating recent form for team: {team_col}")

    for size in window_sizes:
        # Debugging: Print head of DataFrame
        print("DataFrame head before filtering:")
        print(df.head())

        # Calculate form when this team plays at home
        home_games = df[df['home_team'].isin(df[team_col])]
        print(f"Home games for {team_col}: {len(home_games)} rows")

        home_games_form = (
            home_games.groupby(['season', 'home_team'])[result_col]
            .rolling(window=size, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        print("Indexes of df['away_team']:", df['away_team'].index)
        print(f"Indexes of df[{team_col}]:", df[team_col].index)

        # Calculate form when this team plays away
        away_games = df[df['away_team'].isin(df[team_col])]
        print(f"Away games for {team_col}: {len(away_games)} rows")

        # Invert result_col for away games
        away_games[result_col] = -away_games[result_col]

        away_games_form = (
            away_games.groupby(['season', 'away_team'])[result_col]
            .rolling(window=size, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        # Combine home and away games to calculate overall form
        all_games = pd.concat([home_games, away_games]).sort_values(['season', 'datetime'])
        print(f"Total games for {team_col}: {len(all_games)} rows")

        # Map the forms back to the original DataFrame
        home_form_series = pd.Series(index=df.index, dtype='float64')
        away_form_series = pd.Series(index=df.index, dtype='float64')
        overall_form_series = pd.Series(index=df.index, dtype='float64')

        home_form_series.loc[home_games.index] = home_games_form
        away_form_series.loc[away_games.index] = away_games_form

        # Add all forms to the result DataFrame
        result_df[f'{team_col}_home_form_last_{size}'] = home_form_series
        result_df[f'{team_col}_away_form_last_{size}'] = away_form_series

        print(f"Added columns for window size {size}")
        print(f"Home form NaN count: {result_df[f'{team_col}_home_form_last_{size}'].isna().sum()}")
        print(f"Away form NaN count: {result_df[f'{team_col}_away_form_last_{size}'].isna().sum()}")

    return result_df


def add_head_to_head_stats(df, home_col, away_col, stats_cols, window_sizes=[3, 5, 10]):
    # Exact fixture ID (home vs away), keep original match_id
    df['exact_match_id'] = df[home_col] + '_vs_' + df[away_col]
    df['match_id'] = df['exact_match_id'].copy()

    # Two-way fixture ID (includes reverse fixtures)
    df['h2h_match_id'] = df.apply(lambda x: '_vs_'.join(sorted([x[home_col], x[away_col]])), axis=1)

    # Calculate stats for exact fixtures (remove the second rolling mean)
    for size in window_sizes:
        for col in stats_cols:
            exact_stats = df.groupby('exact_match_id')[col].rolling(window=size, min_periods=1).mean()
            exact_stats.reset_index(level=0, drop=True, inplace=True)
            df[f'exact_fixture_{col}_last_{size}'] = exact_stats

            h2h_stats = df.groupby('h2h_match_id')[col].rolling(window=size, min_periods=1).mean()
            h2h_stats.reset_index(level=0, drop=True, inplace=True)
            df[f'h2h_{col}_last_{size}'] = h2h_stats

    df.drop(['exact_match_id', 'h2h_match_id'], axis=1, inplace=True)
    return df


def calculate_league_positions(df):
    df = df.sort_values(['season', 'datetime'])
    team_stats = {}

    def update_stats(season, team, points, goals_scored, goals_conceded, xG, xGA):
        if season not in team_stats:
            team_stats[season] = {}
        if team not in team_stats[season]:
            team_stats[season][team] = {'points': 0, 'goal_diff': 0, 'xG_diff': 0}
        team_stats[season][team]['points'] += points
        team_stats[season][team]['goal_diff'] += goals_scored - goals_conceded
        team_stats[season][team]['xG_diff'] += xG - xGA

    for _, row in df.iterrows():
        season = row['season']
        home_team, away_team = row['home_team'], row['away_team']
        home_goals, away_goals = row['home_goals'], row['away_goals']
        home_xG, away_xG = row['home_xG'], row['away_xG']

        if home_goals > away_goals:
            update_stats(season, home_team, 3, home_goals, away_goals, home_xG, away_xG)
            update_stats(season, away_team, 0, away_goals, home_goals, away_xG, home_xG)
        elif home_goals < away_goals:
            update_stats(season, home_team, 0, home_goals, away_goals, home_xG, away_xG)
            update_stats(season, away_team, 3, away_goals, home_goals, away_xG, home_xG)
        else:
            update_stats(season, home_team, 1, home_goals, away_goals, home_xG, away_xG)
            update_stats(season, away_team, 1, away_goals, home_goals, away_xG, home_xG)

        league_table = pd.DataFrame([
            {'season': s, 'team': t, 'points': stats['points'], 'goal_diff': stats['goal_diff'],
             'xG_diff': stats['xG_diff']}
            for s, teams in team_stats.items()
            for t, stats in teams.items()
            if s == season
        ]).sort_values(['points', 'goal_diff', 'xG_diff'], ascending=[False, False, False])

        league_table['position'] = league_table.groupby('season').cumcount() + 1

        df.loc[_, 'home_position'] = \
        league_table[(league_table['season'] == season) & (league_table['team'] == home_team)]['position'].values[0]
        df.loc[_, 'away_position'] = \
        league_table[(league_table['season'] == season) & (league_table['team'] == away_team)]['position'].values[0]

    return df


def calculate_elo_ratings(df, k_factor=20, home_advantage=100, use_xG=True, reset_factor=0.75):
    df = df.sort_values(['datetime'])
    initial_elo = 1500
    elo_ratings = {}

    def expected_score(rating1, rating2):
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def update_elo(old_elo, expected_score, actual_score, k_factor):
        return old_elo + k_factor * (actual_score - expected_score)

    previous_season = None

    for _, row in df.iterrows():
        season = row['season']
        home_team, away_team = row['home_team'], row['away_team']
        home_goals, away_goals = row['home_goals'], row['away_goals']
        home_xG, away_xG = row['home_xG'], row['away_xG']

        # Initialize ratings for new teams
        if home_team not in elo_ratings:
            elo_ratings[home_team] = initial_elo
        if away_team not in elo_ratings:
            elo_ratings[away_team] = initial_elo

        # Soft reset at the start of a new season
        if season != previous_season and previous_season is not None:
            for team in elo_ratings:
                elo_ratings[team] = (elo_ratings[team] - initial_elo) * reset_factor + initial_elo

        home_elo, away_elo = elo_ratings[home_team], elo_ratings[away_team]

        home_expected = expected_score(home_elo + home_advantage, away_elo)
        away_expected = expected_score(away_elo, home_elo + home_advantage)

        if use_xG:
            home_actual = home_xG / (home_xG + away_xG)
            away_actual = away_xG / (home_xG + away_xG)
        else:
            if home_goals > away_goals:
                home_actual, away_actual = 1, 0
            elif home_goals < away_goals:
                home_actual, away_actual = 0, 1
            else:
                home_actual = away_actual = 0.5

        elo_ratings[home_team] = update_elo(home_elo, home_expected, home_actual, k_factor)
        elo_ratings[away_team] = update_elo(away_elo, away_expected, away_actual, k_factor)

        df.loc[_, 'home_elo'] = home_elo
        df.loc[_, 'away_elo'] = away_elo

        previous_season = season

    return df


def add_xG_features(df):
    # Basic xG derived features
    df['home_xG_overperformance'] = df['home_goals'] - df['home_xG']
    df['away_xG_overperformance'] = df['away_goals'] - df['away_xG']

    # Handle potential division by zero in ratios by using fillna
    df['home_xG_ratio'] = (df['home_xG'] / (df['home_xG'] + df['away_xG'])).fillna(0.5)  # 0.5 for equal chance when no data
    df['away_xG_ratio'] = (df['away_xG'] / (df['home_xG'] + df['away_xG'])).fillna(0.5)

    window_sizes = [3, 5, 10]
    for size in window_sizes:
        # Create a helper DataFrame for team performances
        team_performances = []

        # Add home game performances
        home_perf = df[['season', 'datetime', 'home_team', 'home_xG', 'away_xG']].copy()
        home_perf.columns = ['season', 'datetime', 'team', 'xG_for', 'xG_against']
        team_performances.append(home_perf)

        # Add away game performances
        away_perf = df[['season', 'datetime', 'away_team', 'away_xG', 'home_xG']].copy()
        away_perf.columns = ['season', 'datetime', 'team', 'xG_for', 'xG_against']
        team_performances.append(away_perf)

        # Combine all performances and sort
        all_performances = pd.concat(team_performances).sort_values(['season', 'datetime'])

        # Calculate rolling averages for each team
        numeric_cols = ['xG_for', 'xG_against']
        team_stats = all_performances.groupby(['season', 'team'])[numeric_cols].rolling(window=size, min_periods=1).mean()
        team_stats = team_stats.reset_index()

        # Shift values and fill NaN with team's mean or 0
        for col in ['xG_for', 'xG_against']:
            team_stats[col] = team_stats.groupby(['season', 'team'])[col].shift(1)
            # Fill NaN with team's mean, if no mean available use overall mean, if still NaN use 0
            team_stats[col] = team_stats[col].fillna(team_stats.groupby(['season', 'team'])[col].transform('mean'))
            team_stats[col] = team_stats[col].fillna(team_stats[col].mean())
            team_stats[col] = team_stats[col].fillna(0)

        # Map stats back to home teams
        home_merge = df.merge(
            team_stats[['season', 'team', 'xG_for', 'xG_against']],
            left_on=['season', 'home_team'],
            right_on=['season', 'team'],
            how='left'
        )

        df[f'home_team_overall_xG_last_{size}'] = home_merge['xG_for'].fillna(0)
        df[f'home_team_xG_conceded_last_{size}'] = home_merge['xG_against'].fillna(0)

        # Map stats back to away teams
        away_merge = df.merge(
            team_stats[['season', 'team', 'xG_for', 'xG_against']],
            left_on=['season', 'away_team'],
            right_on=['season', 'team'],
            how='left'
        )

        df[f'away_team_overall_xG_last_{size}'] = away_merge['xG_for'].fillna(0)
        df[f'away_team_xG_conceded_last_{size}'] = away_merge['xG_against'].fillna(0)

        # Calculate venue-specific stats with NaN handling
        home_only = df.groupby(['season', 'home_team'])['home_xG'].rolling(window=size, min_periods=1).mean()
        home_only = home_only.fillna(method='bfill').fillna(0)  # backfill then fill remaining with 0
        home_only.reset_index(level=[0, 1], drop=True, inplace=True)
        df[f'home_team_home_xG_last_{size}'] = home_only

        away_only = df.groupby(['season', 'away_team'])['away_xG'].rolling(window=size, min_periods=1).mean()
        away_only = away_only.fillna(method='bfill').fillna(0)  # backfill then fill remaining with 0
        away_only.reset_index(level=[0, 1], drop=True, inplace=True)
        df[f'away_team_away_xG_last_{size}'] = away_only

    return df


def round_dataframe(df, decimal_places=2, columns=None):
    """
    Round numeric columns in a DataFrame to a specified number of decimal places.

    :param df: pandas DataFrame
    :param decimal_places: number of decimal places to round to (default: 2)
    :param columns: list of column names to round (if None, rounds all numeric columns)
    :return: DataFrame with rounded values
    """
    # If no specific columns are provided, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    # Round the specified columns
    for col in columns:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].round(decimal_places)

    return df

# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))
resources_folder = find_resources_folder(script_dir)

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(pre_process())
