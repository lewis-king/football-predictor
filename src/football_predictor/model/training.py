import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import BatchNormalization, Dropout, Dense
from keras.src.optimizers import Adam
from keras.src.regularizers import l1_l2
from keras_tuner import Hyperband
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.python.keras import regularizers


def prepare_data(df):
    print("Initial DataFrame shape:", df.shape)
    # Store original match information before preprocessing
    match_info = df[['datetime', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals']].copy()

    # Drop non-feature columns and handle any missing values
    feature_columns = df.columns.tolist()
    columns_to_drop = ['datetime', 'season', 'home_team', 'away_team', 'home_goals',
                       'away_goals', 'result', 'match_id', 'goal_diff', 'supremacy',
                       'home_xG_overperformance', 'away_xG_overperformance', 'xG_diff',
                       'home_xG', 'away_xG', 'home_xG_ratio', 'away_xG_ratio']
    feature_columns = [col for col in feature_columns if col not in columns_to_drop]
    print("feature_columns:", feature_columns)
    # Create feature matrix X and target variable y
    X = df[feature_columns].copy()
    y = df['supremacy']  # supremacy as target

    # Print shapes before split
    print("X shape before split:", X.shape)
    print("y shape before split:", y.shape)

    # Split data ensuring temporal order is maintained
    train_size = 0.8
    split_idx = int(len(df) * train_size)

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    match_info_test = match_info[split_idx:].reset_index(drop=True)  # Reset index for proper alignment

    # Print shapes after split
    print("Training shapes - X:", X_train.shape, "y:", y_train.shape)
    print("Testing shapes - X:", X_test.shape, "y:", y_test.shape)
    print("Match info test shape:", match_info_test.shape)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, match_info_test


def build_model(hp):
    """
    Model builder function for hyperparameter tuning.
    """
    model = Sequential()

    # First layer
    model.add(Dense(
        units=hp.Int('units_1', min_value=32, max_value=256, step=32),
        activation=hp.Choice('activation_1', values=['relu', 'elu']),
        kernel_regularizer=l1_l2(
            l1=hp.Float('l1_1', min_value=1e-5, max_value=1e-2, sampling='log'),
            l2=hp.Float('l2_1', min_value=1e-5, max_value=1e-2, sampling='log')
        ),
        input_dim=X_train.shape[1]
    ))

    # Add BatchNormalization and Dropout after first layer
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5)))

    # Second layer (optional)
    if hp.Boolean('second_layer'):
        model.add(Dense(
            units=hp.Int('units_2', min_value=16, max_value=128, step=16),
            activation=hp.Choice('activation_2', values=['relu', 'elu']),
            kernel_regularizer=l1_l2(
                l1=hp.Float('l1_2', min_value=1e-5, max_value=1e-2, sampling='log'),
                l2=hp.Float('l2_2', min_value=1e-5, max_value=1e-2, sampling='log')
            )
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5)))

    # Output layer
    model.add(Dense(1))

    # Compile model with tunable learning rate
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_model_with_tuning(X_train, X_test, y_train, y_test):
    print("Starting hyperparameter tuning...")
    print("Training shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # Define callbacks for the final model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=1e-4
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    model_checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_weights_only=True
    )

    # Initialize tuner
    tuner = Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=100,
        factor=3,
        directory='keras_tuner',
        project_name='supremacy_prediction'
    )

    # Search for best hyperparameters
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters:")
    for hp in best_hps.values:
        print(f"{hp}: {best_hps.values[hp]}")

    # Build model with best hyperparameters
    model = tuner.hypermodel.build(best_hps)

    # Train final model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test, feature_columns):
    # Make predictions
    print("Training predict shape")
    print(X_test.shape)
    y_pred = model.predict(X_test)

    # Print shapes for debugging
    print("Shape of y_test:", y_test.shape)
    print("Shape of y_pred before flatten:", y_pred.shape)
    print("Shape of y_pred after flatten:", y_pred.flatten().shape)

    y_pred_flat = y_pred.flatten()
    if len(y_test) != len(y_pred_flat):
        raise ValueError(f"Shape mismatch: y_test {y_test.shape} vs y_pred {y_pred_flat.shape}")

    # Calculate metrics
    mse = np.mean((y_test - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_test - y_pred_flat))
    r2 = r2_score(y_test, y_pred_flat)

    print("\nModel Performance Metrics:")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Feature importance
    weights = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': weights
    }).sort_values('importance', ascending=False)

    # Print detailed error analysis
    print("\nError Distribution:")
    errors = y_test - y_pred_flat
    print(f"Error Mean: {errors.mean():.4f}")
    print(f"Error Std: {errors.std():.4f}")
    print(f"Error 25th percentile: {np.percentile(errors, 25):.4f}")
    print(f"Error 75th percentile: {np.percentile(errors, 75):.4f}")

    return feature_importance


def plot_training_history(history):
    print("Available metrics in history:", history.history.keys())

    # Determine how many metrics we have (excluding validation metrics)
    metrics = [m for m in history.history.keys() if not m.startswith('val_')]
    n_metrics = len(metrics)

    # Create a figure with enough subplots for each metric
    plt.figure(figsize=(6 * n_metrics, 4))

    # Plot each metric and its validation equivalent if it exists
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, n_metrics, i)
        plt.plot(history.history[metric], label=f'Training {metric}')

        # Plot validation metric if it exists
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validation {metric}')

        plt.title(f'Model {metric.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Print final values
    print("\nFinal metric values:")
    for metric in metrics:
        final_train = history.history[metric][-1]
        print(f"Training {metric}: {final_train:.4f}")

        val_metric = f'val_{metric}'
        if val_metric in history.history:
            final_val = history.history[val_metric][-1]
            print(f"Validation {metric}: {final_val:.4f}")
        print()


def get_accuracy_indicator(pred_diff, actual_supremacy):
    """
    Determine accuracy indicator based on predicted and actual supremacy
    Returns multiple ticks/crosses based on prediction accuracy
    """
    # For draws (actual_supremacy = 0)
    if actual_supremacy == 0:
        if abs(pred_diff) <= 0.25:
            return "✓✓✓"  # Very accurate draw prediction
        elif abs(pred_diff) <= 0.5:
            return "✓✓"  # Good draw prediction
        elif abs(pred_diff) <= 0.75:
            return "✓"  # Acceptable draw prediction
        return "✗"  # Poor draw prediction

    # For home wins (actual_supremacy > 0)
    elif actual_supremacy > 0:
        if pred_diff <= 0:
            return "✗"  # Wrong direction
        elif actual_supremacy == 1:  # One goal win
            if 0.5 <= pred_diff <= 1.5:
                return "✓✓✓"  # Predicting close to a 1-goal win
            elif 0.25 <= pred_diff < 0.5:
                return "✓✓"  # Predicting a close game with home advantage
            elif pred_diff > 1.5:
                return "✓"  # Right direction but overconfident
        elif actual_supremacy == 2:  # Two goal win
            if 1.25 <= pred_diff <= 2.25:
                return "✓✓✓"  # Centered around 2-goal prediction
            elif 0.75 <= pred_diff < 1.25 or 2.25 < pred_diff <= 2.75:
                return "✓✓"  # Reasonable range either side
            elif pred_diff > 0:
                return "✓"  # Right direction
        elif actual_supremacy >= 3:  # Three or more goal win
            if abs(pred_diff - actual_supremacy) <= 0.75:
                return "✓✓✓"  # Very close to actual large margin
            elif abs(pred_diff - actual_supremacy) <= 1.25:
                return "✓✓"  # Reasonably close to large margin
            elif pred_diff > 1.5:  # At least predicted a clear win
                return "✓"
        return "✗"

    # For away wins (actual_supremacy < 0)
    else:
        if pred_diff >= 0:
            return "✗"  # Wrong direction
        elif actual_supremacy == -1:  # One goal away win
            if -1.5 <= pred_diff <= -0.5:
                return "✓✓✓"  # Predicting close to a 1-goal away win
            elif -0.5 < pred_diff <= -0.25:
                return "✓✓"  # Predicting a close game with away advantage
            elif pred_diff < -1.5:
                return "✓"  # Right direction but overconfident
        elif actual_supremacy == -2:  # Two goal away win
            if -2.25 <= pred_diff <= -1.25:
                return "✓✓✓"  # Centered around 2-goal prediction
            elif -2.75 <= pred_diff < -1.25 or -1.25 < pred_diff <= -0.75:
                return "✓✓"  # Reasonable range either side
            elif pred_diff < 0:
                return "✓"  # Right direction
        elif actual_supremacy <= -3:  # Three or more goal away win
            if abs(pred_diff - actual_supremacy) <= 0.75:
                return "✓✓✓"  # Very close to actual large margin
            elif abs(pred_diff - actual_supremacy) <= 1.25:
                return "✓✓"  # Reasonably close to large margin
            elif pred_diff < -1.5:  # At least predicted a clear win
                return "✓"
        return "✗"

def display_predictions(model, X_test, match_info_test, num_matches=5):
    """
    Display predictions with match context
    """
    recent_matches = X_test[-num_matches:]
    print("Predict features:")
    print(recent_matches)
    predictions = model.predict(recent_matches)
    recent_info = match_info_test.iloc[-num_matches:]

    print("\nRecent Match Predictions:")
    print("-" * 100)
    print(f"{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Actual Score':<15} {'Predicted Supremacy':>15}")
    print("-" * 100)

    for i, (pred, (_, match)) in enumerate(zip(predictions, recent_info.iterrows())):
        date_str = pd.to_datetime(match['datetime']).strftime('%Y-%m-%d')
        actual_score = f"{match['home_goals']}-{match['away_goals']}"
        actual_supremacy = match['home_goals'] - match['away_goals']
        pred_diff = pred[0]

        accuracy = get_accuracy_indicator(pred_diff, actual_supremacy)

        # Add color coding based on accuracy (optional)
        accuracy_width = 5  # Width for alignment including multi-character ticks

        print(f"{date_str:<12} {match['home_team']:<25} {match['away_team']:<25} "
              f"{actual_score:<15} {pred_diff:>15.2f} {accuracy:>{accuracy_width}}")

    print("-" * 100)

    # Calculate and display prediction metrics (updated to count any tick as correct)
    recent_actual = recent_info['home_goals'] - recent_info['away_goals']
    recent_pred = predictions.flatten()
    mae = np.mean(np.abs(recent_actual - recent_pred))

    # Count predictions with any ticks as correct
    correct_predictions = sum(1 for actual, pred in zip(recent_actual, recent_pred)
                              if "✓" in get_accuracy_indicator(pred, actual))

    print(f"\nPrediction Metrics for last {num_matches} matches:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Correct Result Direction: {correct_predictions}/{num_matches} "
          f"({(correct_predictions / num_matches) * 100:.1f}%)")

    # Optional: Add detailed breakdown
    tick_counts = {"✓✓✓": 0, "✓✓": 0, "✓": 0, "✗": 0}
    for actual, pred in zip(recent_actual, recent_pred):
        accuracy = get_accuracy_indicator(pred, actual)
        if "✓✓✓" in accuracy:
            tick_counts["✓✓✓"] += 1
        elif "✓✓" in accuracy:
            tick_counts["✓✓"] += 1
        elif "✓" in accuracy:
            tick_counts["✓"] += 1
        else:
            tick_counts["✗"] += 1

    print("\nPrediction Accuracy Breakdown:")
    print(f"Perfect Predictions (✓✓✓): {tick_counts['✓✓✓']} ({(tick_counts['✓✓✓'] / num_matches) * 100:.1f}%)")
    print(f"Good Predictions (✓✓): {tick_counts['✓✓']} ({(tick_counts['✓✓'] / num_matches) * 100:.1f}%)")
    print(f"Acceptable Predictions (✓): {tick_counts['✓']} ({(tick_counts['✓'] / num_matches) * 100:.1f}%)")
    print(f"Incorrect Predictions (✗): {tick_counts['✗']} ({(tick_counts['✗'] / num_matches) * 100:.1f}%)")

if __name__ == "__main__":
    # Load your engineered data
    df = pd.read_csv('../data/output/engineered_football_data.csv')
    print("Original DataFrame shape:", df.shape)

    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, feature_columns, match_info_test = prepare_data(df)

        model, history = train_model_with_tuning(X_train, X_test, y_train, y_test)
        feature_importance = evaluate_model(model, X_test, y_test, feature_columns)

        # Plot training history
        plot_training_history(history)

        # Print feature importance
        print("\nOrdered Most Important Features:")
        print("-" * 50)
        for idx, row in feature_importance.head(100).iterrows():
            print(f"{row['feature']:<30} {row['importance']:.4f}")

        # Predictions
        display_predictions(model, X_test, match_info_test, num_matches=10)  # Show last 10 matches


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
