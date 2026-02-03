"""
Advanced Multi-Variable GDP Prediction Model
=============================================
Uses multiple predictors/features to forecast GDP for years 101-105.
This model combines demographic, economic, and profession-related features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class AdvancedGDPPredictor:
    """Advanced GDP prediction using multiple features"""
    
    def __init__(self, predictors_csv_path):
        """
        Initialize with predictors dataset
        
        Args:
            predictors_csv_path: Path to predictors.csv with all features
        """
        self.df = pd.read_csv(predictors_csv_path)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.predictions_df = None
        
    def prepare_features(self, exclude_cols=None):
        """
        Prepare feature matrix and target variable
        
        Args:
            exclude_cols: Columns to exclude from features
        """
        if exclude_cols is None:
            exclude_cols = ['gdp', 'total_income']  # total_income is basically GDP
        
        # Separate features and target
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols].values
        y = self.df['gdp'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✓ Features prepared: {len(feature_cols)} predictors")
        print(f"  Feature names: {feature_cols[:10]}... (showing first 10)")
        
        return X_scaled, y, feature_cols
    
    def train_models(self):
        """Train multiple regression models"""
        print("\n🔧 Training Models...")
        
        X, y, feature_cols = self.prepare_features()
        
        # 1. Linear Regression
        linear = LinearRegression()
        linear.fit(X, y)
        self.models['linear'] = {
            'model': linear,
            'name': 'Linear Regression'
        }
        print("✓ Linear Regression trained")
        
        # 2. Ridge Regression (with regularization)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        self.models['ridge'] = {
            'model': ridge,
            'name': 'Ridge Regression'
        }
        print("✓ Ridge Regression trained")
        
        # 3. Lasso Regression (feature selection)
        lasso = Lasso(alpha=10.0)
        lasso.fit(X, y)
        self.models['lasso'] = {
            'model': lasso,
            'name': 'Lasso Regression'
        }
        print("✓ Lasso Regression trained")
        
        # 4. Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        self.models['random_forest'] = {
            'model': rf,
            'name': 'Random Forest'
        }
        print("✓ Random Forest trained")
        
        # 5. Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        gb.fit(X, y)
        self.models['gradient_boosting'] = {
            'model': gb,
            'name': 'Gradient Boosting'
        }
        print("✓ Gradient Boosting trained")
        
        # Store feature names
        self.feature_cols = feature_cols
        
        # Calculate feature importance from Random Forest
        self.calculate_feature_importance()
        
    def calculate_feature_importance(self):
        """Calculate and display feature importance from Random Forest"""
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']['model']
            importance = rf_model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = feature_importance_df
            
            print("\n📊 Top 10 Most Important Features:")
            for idx, row in feature_importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
    
    def evaluate_models(self):
        """Evaluate model performance using cross-validation"""
        print("\n📈 Model Performance (Cross-Validation Score):")
        
        X, y, _ = self.prepare_features()
        
        for key, model_info in self.models.items():
            model = model_info['model']
            name = model_info['name']
            
            # 5-fold cross-validation
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            mean_score = scores.mean()
            std_score = scores.std()
            
            print(f"   {name}: R² = {mean_score:.4f} (±{std_score:.4f})")
            self.models[key]['cv_score'] = mean_score
    
    def create_future_features(self):
        """
        Create feature values for years 101-105 based on trends
        This is the tricky part - we need to extrapolate features
        """
        print("\n🔮 Generating future features (years 101-105)...")
        
        future_data = []
        
        for year in range(101, 106):
            # Start with year
            features = {'year': year}
            
            # For each feature, use trend-based extrapolation
            for col in self.df.columns:
                if col in ['year', 'gdp']:
                    continue
                
                # Get last 10 years of data for this feature
                recent_values = self.df[col].tail(10).values
                years_recent = self.df['year'].tail(10).values
                
                # Fit simple linear trend
                if len(recent_values) > 0 and not np.all(recent_values == 0):
                    # Simple linear extrapolation
                    slope = np.polyfit(years_recent, recent_values, 1)[0]
                    last_value = recent_values[-1]
                    years_ahead = year - years_recent[-1]
                    predicted_value = last_value + (slope * years_ahead)
                    
                    # Don't allow negative values for counts
                    if 'count' in col or 'population' in col:
                        predicted_value = max(0, predicted_value)
                    
                    features[col] = predicted_value
                else:
                    features[col] = 0
            
            future_data.append(features)
        
        future_df = pd.DataFrame(future_data)
        
        # Ensure column order matches training data
        feature_cols = [col for col in self.df.columns if col not in ['gdp', 'total_income']]
        future_df = future_df[feature_cols]
        
        print(f"✓ Future features created for years 101-105")
        
        return future_df
    
    def predict_future(self):
        """Predict GDP for years 101-105 using all trained models"""
        
        future_df = self.create_future_features()
        
        # Scale future features
        X_future = self.scaler.transform(future_df.values)
        
        # Make predictions with each model
        predictions = {'year': range(101, 106)}
        
        for key, model_info in self.models.items():
            model = model_info['model']
            name = model_info['name']
            
            pred = model.predict(X_future)
            predictions[f'{key}_prediction'] = pred
        
        predictions_df = pd.DataFrame(predictions)
        
        # Calculate ensemble (weighted average based on CV scores)
        pred_cols = [col for col in predictions_df.columns if 'prediction' in col]
        
        # Simple average ensemble
        predictions_df['ensemble_avg'] = predictions_df[pred_cols].mean(axis=1)
        
        # Weighted ensemble based on CV scores
        if all(key in self.models and 'cv_score' in self.models[key] for key in ['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']):
            weights = []
            for col in pred_cols:
                model_key = col.replace('_prediction', '')
                if model_key in self.models and 'cv_score' in self.models[model_key]:
                    weights.append(max(0, self.models[model_key]['cv_score']))
                else:
                    weights.append(1.0)
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            predictions_df['ensemble_weighted'] = (predictions_df[pred_cols] * weights).sum(axis=1)
        else:
            predictions_df['ensemble_weighted'] = predictions_df['ensemble_avg']
        
        self.predictions_df = predictions_df
        
        return predictions_df
    
    def print_predictions(self):
        """Print final predictions"""
        if self.predictions_df is None:
            print("No predictions available. Run predict_future() first.")
            return
        
        print("\n" + "="*70)
        print("GDP PREDICTIONS FOR YEARS 101-105")
        print("="*70)
        
        # Show all model predictions
        print("\n📊 All Model Predictions:")
        pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
        print(self.predictions_df.to_string(index=False))
        
        # Show recommended predictions
        print("\n" + "="*70)
        print("🎯 RECOMMENDED PREDICTIONS (Weighted Ensemble):")
        print("="*70)
        for _, row in self.predictions_df.iterrows():
            print(f"   Year {int(row['year'])}: ${row['ensemble_weighted']:,.2f}")
        print("="*70)
    
    def visualize_predictions(self, save_path='advanced_gdp_predictions.png'):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Historical GDP + All model predictions
        ax1 = axes[0, 0]
        ax1.plot(self.df['year'], self.df['gdp'], 'b-', linewidth=2, label='Historical GDP', alpha=0.7)
        
        if self.predictions_df is not None:
            colors = ['red', 'green', 'orange', 'purple', 'brown']
            model_keys = ['linear', 'ridge', 'random_forest', 'gradient_boosting', 'lasso']
            
            for i, key in enumerate(model_keys):
                col_name = f'{key}_prediction'
                if col_name in self.predictions_df.columns:
                    ax1.plot(self.predictions_df['year'], self.predictions_df[col_name], 
                            '--', linewidth=1.5, alpha=0.6, label=self.models[key]['name'])
            
            ax1.plot(self.predictions_df['year'], self.predictions_df['ensemble_weighted'], 
                    'r-', linewidth=3, marker='o', markersize=8, label='Ensemble (Recommended)')
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('GDP')
        ax1.set_title('Historical GDP and Multi-Model Predictions', fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zoomed in - Last 20 years + predictions
        ax2 = axes[0, 1]
        recent = self.df.tail(20)
        ax2.plot(recent['year'], recent['gdp'], 'b-', linewidth=2, marker='o', markersize=4, label='Recent GDP')
        
        if self.predictions_df is not None:
            ax2.plot(self.predictions_df['year'], self.predictions_df['ensemble_weighted'], 
                    'r--', linewidth=2, marker='o', markersize=8, label='Predictions')
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel('GDP')
        ax2.set_title('Recent Trend and Predictions', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature Importance
        ax3 = axes[1, 0]
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            ax3.barh(range(len(top_features)), top_features['importance'])
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'], fontsize=8)
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 15 Feature Importance', fontweight='bold')
            ax3.invert_yaxis()
            ax3.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Prediction Comparison
        ax4 = axes[1, 1]
        if self.predictions_df is not None:
            years = self.predictions_df['year']
            width = 0.15
            x = np.arange(len(years))
            
            model_keys = ['linear', 'ridge', 'random_forest', 'gradient_boosting', 'ensemble_weighted']
            colors = ['skyblue', 'lightgreen', 'orange', 'purple', 'red']
            
            for i, key in enumerate(model_keys):
                col_name = f'{key}_prediction' if key != 'ensemble_weighted' else key
                if col_name in self.predictions_df.columns:
                    label = self.models.get(key, {}).get('name', 'Ensemble') if key != 'ensemble_weighted' else 'Ensemble'
                    ax4.bar(x + i*width, self.predictions_df[col_name], width, 
                           label=label, color=colors[i], alpha=0.8)
            
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Predicted GDP')
            ax4.set_title('Model Comparison for Years 101-105', fontweight='bold')
            ax4.set_xticks(x + width * 2)
            ax4.set_xticklabels(years)
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Visualization saved to: {save_path}")
        plt.show()
    
    def export_predictions(self, output_path='gdp_predictions_advanced.csv'):
        """Export predictions to CSV"""
        if self.predictions_df is not None:
            # Create clean output with just years and final predictions
            output_df = self.predictions_df[['year', 'ensemble_weighted']].copy()
            output_df.columns = ['year', 'predicted_gdp']
            output_df.to_csv(output_path, index=False)
            print(f"\n💾 Final predictions saved to: {output_path}")
            
            # Also save detailed predictions
            detailed_path = output_path.replace('.csv', '_detailed.csv')
            self.predictions_df.to_csv(detailed_path, index=False)
            print(f"💾 Detailed predictions saved to: {detailed_path}")
        else:
            print("No predictions to export.")


def main():
    """Main execution function"""
    print("="*70)
    print("ADVANCED MULTI-VARIABLE GDP PREDICTION MODEL")
    print("="*70)
    
    # Check if predictors.csv exists
    try:
        # Initialize model
        model = AdvancedGDPPredictor('predictors.csv')
        
        # Train all models
        model.train_models()
        
        # Evaluate performance
        model.evaluate_models()
        
        # Make predictions
        print("\n🔮 Generating Predictions for Years 101-105...")
        predictions = model.predict_future()
        
        # Display results
        model.print_predictions()
        
        # Visualize
        print("\n📊 Creating Visualizations...")
        model.visualize_predictions()
        
        # Export
        model.export_predictions()
        
        print("\n✅ Advanced model complete!")
        print("\nFinal Output:")
        print("   - gdp_predictions_advanced.csv (final predictions)")
        print("   - gdp_predictions_advanced_detailed.csv (all model outputs)")
        print("   - advanced_gdp_predictions.png (visualization)")
        
    except FileNotFoundError:
        print("\n❌ Error: predictors.csv not found!")
        print("\nPlease run create_predictors.py first to generate the predictors dataset:")
        print("   python create_predictors.py")
        return


if __name__ == "__main__":
    main()
