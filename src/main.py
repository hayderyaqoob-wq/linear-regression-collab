# src/main.py (updated snippet)
import argparse
import os
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    parser = argparse.ArgumentParser(description="Train Linear Regression and predict Y.")
    parser.add_argument("--x", type=float, default=None, help="Input X value to predict Y")
    args = parser.parse_args()

    # Check for x value: CLI arg > env var > None
    x_value = args.x
    if x_value is None and os.getenv("PREDICT_X"):
        try:
            x_value = float(os.getenv("PREDICT_X"))
            print(f"📥 Read x={x_value} from environment variable PREDICT_X")
        except ValueError:
            print("⚠️  PREDICT_X env var is not a valid float, skipping prediction")

    print("🚀 Starting Linear Regression Training...")
    
    # Generate data and train (same as before)
    X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print("\n--- Model Results ---")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept:   {model.intercept_:.4f}")
    print(f"MSE:         {mse:.4f}")
    print(f"R² Score:    {r2:.4f}")
    print("---------------------\n")
    
    # Predict if x_value is provided
    if x_value is not None:
        x_input = np.array([[x_value]])
        y_predicted = model.predict(x_input)[0]
        print(f"🎯 PREDICTION: For x = {x_value}, predicted y = {y_predicted:.4f}")
        # Output in a parseable format for GitHub Actions
        print(f"::set-output name=predicted_y::{y_predicted:.4f}")
    else:
        print("💡 No x value provided. Run with --x <val> or set PREDICT_X env var.")

if __name__ == "__main__":
    main()