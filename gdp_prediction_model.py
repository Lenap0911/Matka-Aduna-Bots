"""
Simple GDP Prediction Model
============================
This model predicts GDP for the next 5 years (101-105) based on historical data.
You can expand this by adding more sophisticated features later.

GDP Formula: GDP = Sum of all income from population
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class GDPPredictorModel:
    """Simple GDP prediction model with multiple forecasting methods"""
    
    def __init__(self, csv_path):
        """
        Initialize the model with historical GDP data
        
        Args:
            csv_path: Path to statistics CSV file containing year and gdp columns
        """
        self.df = pd.read_csv(csv_path)
        self.models = {}
        self.predictions = {}
        
    def prepare_data(self):
        """Prepare data for modeling"""
        self.X = self.df['year'].values.reshape(-1, 1)
        self.y = self.df['gdp'].values
        return self.X, self.y
    
    def train_linear_model(self):
        """Train simple linear regression model"""
        X, y = self.prepare_data()
        
        model = LinearRegression()
        model.fit(X, y)
        
        self.models['linear'] = model
        print(f"✓ Linear Model trained")
        print(f"  Slope: {model.coef_[0]:.2f}")
        print(f"  Intercept: {model.intercept_:.2f}")
        return model
    
    def train_polynomial_model(self, degree=2):
        """Train polynomial regression model"""
        X, y = self.prepare_data()
        
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        self.models['polynomial'] = {
            'model': model,
            'poly_features': poly_features,
            'degree': degree
        }
        print(f"✓ Polynomial Model (degree={degree}) trained")
        return model
    
    def train_moving_average(self, window=10):
        """Calculate moving average trend"""
        self.models['moving_average'] = {
            'window': window,
            'last_values': self.y[-window:]
        }
        print(f"✓ Moving Average (window={window}) calculated")
    
    def predict_future(self, years_ahead=5):
        """
        Predict GDP for future years using all trained models
        
        Args:
            years_ahead: Number of years to predict (default: 5)
        
        Returns:
            DataFrame with predictions from all models
        """
        last_year = self.df['year'].max()
        future_years = np.arange(last_year + 1, last_year + 1 + years_ahead).reshape(-1, 1)
        
        predictions_df = pd.DataFrame({'year': future_years.flatten()})
        
        # Linear prediction
        if 'linear' in self.models:
            linear_pred = self.models['linear'].predict(future_years)
            predictions_df['linear_prediction'] = linear_pred
        
        # Polynomial prediction
        if 'polynomial' in self.models:
            poly_model = self.models['polynomial']
            X_poly_future = poly_model['poly_features'].transform(future_years)
            poly_pred = poly_model['model'].predict(X_poly_future)
            predictions_df['polynomial_prediction'] = poly_pred
        
        # Moving average prediction
        if 'moving_average' in self.models:
            ma_model = self.models['moving_average']
            ma_value = np.mean(ma_model['last_values'])
            predictions_df['moving_average_prediction'] = ma_value
        
        # Ensemble (average of all models)
        pred_columns = [col for col in predictions_df.columns if 'prediction' in col]
        predictions_df['ensemble_prediction'] = predictions_df[pred_columns].mean(axis=1)
        
        self.predictions = predictions_df
        return predictions_df
    
    def calculate_growth_rate(self):
        """Calculate historical average growth rate"""
        growth_rates = []
        for i in range(1, len(self.y)):
            growth = ((self.y[i] - self.y[i-1]) / self.y[i-1]) * 100
            growth_rates.append(growth)
        
        avg_growth = np.mean(growth_rates)
        std_growth = np.std(growth_rates)
        
        print(f"\n📊 Historical Growth Statistics:")
        print(f"  Average growth rate: {avg_growth:.2f}%")
        print(f"  Std deviation: {std_growth:.2f}%")
        print(f"  Min growth: {np.min(growth_rates):.2f}%")
        print(f"  Max growth: {np.max(growth_rates):.2f}%")
        
        return avg_growth, std_growth
    
    def visualize_predictions(self, save_path='gdp_predictions.png'):
        """Create visualization of historical data and predictions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Full historical data + predictions
        ax1.plot(self.df['year'], self.df['gdp'], 'b-', linewidth=2, label='Historical GDP', alpha=0.7)
        
        if not self.predictions.empty:
            ax1.plot(self.predictions['year'], self.predictions['ensemble_prediction'], 
                    'r--', linewidth=2, marker='o', markersize=8, label='Ensemble Prediction')
            
            if 'linear_prediction' in self.predictions.columns:
                ax1.plot(self.predictions['year'], self.predictions['linear_prediction'], 
                        'g:', linewidth=1.5, alpha=0.5, label='Linear Model')
            
            if 'polynomial_prediction' in self.predictions.columns:
                ax1.plot(self.predictions['year'], self.predictions['polynomial_prediction'], 
                        'm:', linewidth=1.5, alpha=0.5, label='Polynomial Model')
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('GDP', fontsize=12)
        ax1.set_title('GDP Historical Data and Predictions', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Last 20 years + predictions (zoomed in)
        recent_data = self.df.tail(20)
        ax2.plot(recent_data['year'], recent_data['gdp'], 'b-', linewidth=2, 
                marker='o', markersize=5, label='Recent Historical GDP')
        
        if not self.predictions.empty:
            ax2.plot(self.predictions['year'], self.predictions['ensemble_prediction'], 
                    'r--', linewidth=2, marker='o', markersize=8, label='Predictions')
        
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('GDP', fontsize=12)
        ax2.set_title('Recent Trend and Predictions (Zoomed)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Visualization saved to: {save_path}")
        plt.show()
    
    def export_predictions(self, output_path='gdp_predictions.csv'):
        """Export predictions to CSV file"""
        if not self.predictions.empty:
            self.predictions.to_csv(output_path, index=False)
            print(f"\n💾 Predictions saved to: {output_path}")
        else:
            print("No predictions to export. Run predict_future() first.")
    
    def print_summary(self):
        """Print summary of predictions"""
        if self.predictions.empty:
            print("No predictions available. Run predict_future() first.")
            return
        
        print("\n" + "="*60)
        print("GDP PREDICTIONS SUMMARY")
        print("="*60)
        print(self.predictions.to_string(index=False))
        print("="*60)
        
        # Show recommended prediction (ensemble)
        print("\n🎯 RECOMMENDED PREDICTIONS (Ensemble Model):")
        for _, row in self.predictions.iterrows():
            print(f"  Year {int(row['year'])}: ${row['ensemble_prediction']:,.2f}")


def main():
    """Main function to run the GDP prediction model"""
    print("="*60)
    print("GDP PREDICTION MODEL")
    print("="*60)
    
    # Initialize model with your data
    csv_path = 'data/data/statistics_all_teams_year100.csv'
    model = GDPPredictorModel(csv_path)
    
    # Train all models
    print("\n🔧 Training Models...")
    model.train_linear_model()
    model.train_polynomial_model(degree=2)
    model.train_moving_average(window=10)
    
    # Calculate growth statistics
    model.calculate_growth_rate()
    
    # Make predictions for next 5 years (101-105)
    print("\n🔮 Generating Predictions...")
    predictions = model.predict_future(years_ahead=5)
    
    # Show results
    model.print_summary()
    
    # Visualize
    print("\n📊 Creating Visualizations...")
    model.visualize_predictions('gdp_predictions.png')
    
    # Export to CSV
    model.export_predictions('gdp_predictions.csv')
    
    print("\n Model complete! You can now:")
    print("   1. Review predictions in gdp_predictions.csv")
    print("   2. View visualization in gdp_predictions.png")
    print("   3. Modify the model to add more features")
    print("   4. Use individual models or the ensemble prediction")


if __name__ == "__main__":
    main()
