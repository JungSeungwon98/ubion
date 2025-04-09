import argparse
from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor
from model import VolatilityPredictor
import subprocess
import sys
import os

def run_streamlit():
    """Run the Streamlit app"""
    print("Starting Streamlit app...")
    print("Access the dashboard at: http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

def main():
    parser = argparse.ArgumentParser(description='Stock Market Volatility Prediction')
    parser.add_argument('--mode', choices=['train', 'streamlit'], default='streamlit',
                      help='Mode to run the application in')
    parser.add_argument('--symbol', default='AAPL',
                      help='Stock symbol to analyze')
    parser.add_argument('--days', type=int, default=365,
                      help='Number of days of historical data to analyze')
    
    args = parser.parse_args()
    
    if args.mode == 'streamlit':
        run_streamlit()
        return
    
    # Load and preprocess data
    loader = MarketDataLoader()
    try:
        data = loader.load_market_data(args.symbol, args.days)
        features = loader.prepare_features(data)
        
        # Prepare data for training
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(features)
        
        # Train model
        model = VolatilityPredictor()
        model.train(X_train, y_train)
        
        # Evaluate model
        accuracy = model.evaluate(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Save model
        model.save_model(f"models/{args.symbol}_model.pkl")
        print(f"Model saved as models/{args.symbol}_model.pkl")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 