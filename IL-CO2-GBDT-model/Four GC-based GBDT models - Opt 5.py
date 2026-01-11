# python code of four GC-based GBDT models - Opt. 5


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import randint, uniform, loguniform
import time
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']





print("\n1. Read pre standardized data...")
database = pd.read_excel("CO2_database_GC.xlsx")
print(f"   data dimension: {database.shape}")





print("\n2. Data preparation and standardization...")

numeric_columns = database.select_dtypes(include=[np.number]).columns.tolist()
print(f"   Total number of numerical columns: {len(numeric_columns)}")

exclude_columns = ['No', 'IL', 'ID', 'SMILES', 'x_CO2 (mol/kg sorbent)']
feature_columns = [col for col in numeric_columns if col not in exclude_columns]
target_column = 'x_CO2 (mol/kg sorbent)'

print(f"   Number of feature columns used: {len(feature_columns)}")
print(f"   target variable: {target_column}")


X = database[feature_columns]
y = database[target_column]

print(f"   Dimension of feature matrix: {X.shape}")
print(f"   Target vector dimension: {y.shape}")


print(f"\n   Data Quality Check:")
print(f"   number of Infs in X: {np.isinf(X).sum().sum()}")
print(f"   Number of NaNs in X: {X.isna().sum().sum()}")
print(f"   number of Infs in y: {np.isinf(y).sum()}")
print(f"   Number of NaNs in y: {y.isna().sum()}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=512)

print(f"\n   Data split result:")
print(f"   training set: {X_train.shape}")
print(f"   test set: {X_test.shape}")


print("\n   data standardization...")
experimental_features = ['T (K)', 'P (bar)']
desc_features = [col for col in feature_columns if col not in experimental_features]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()


X_train_scaled[desc_features] = scaler.fit_transform(X_train[desc_features])
X_test_scaled[desc_features] = scaler.transform(X_test[desc_features])

print(f"   Standardizing {len(desc_features)} descriptor features")
print(f"   {len(experimental_features)} experimental features remain in original scale")





def save_standardization_parameters(scaler, desc_features, experimental_features, X_train, save_path="."):
    params_data = []
    
    for i, feature in enumerate(desc_features):
        params_data.append({
            'Feature': feature,
            'Mean_μ': scaler.mean_[i],
            'Std_σ': scaler.scale_[i],
            'Feature_Type': 'Descriptor (Scaled)',
            'Standardization_Formula': f"({feature} - {scaler.mean_[i]:.6f}) / {scaler.scale_[i]:.6f}"
        })
    
    for feature in experimental_features:
        mean_val = X_train[feature].mean()
        std_val = X_train[feature].std()
        params_data.append({
            'Feature': feature,
            'Mean_μ': mean_val,
            'Std_σ': std_val,
            'Feature_Type': 'Experimental (Not Scaled)',
            'Standardization_Formula': 'No Scaling - Original Scale'
        })
    
    params_df = pd.DataFrame(params_data)
    
    params_df['sort_key'] = params_df['Feature_Type'].map({
        'Experimental (Not Scaled)': 0, 
        'Descriptor (Scaled)': 1
    })
    params_df = params_df.sort_values(['sort_key', 'Feature']).drop('sort_key', axis=1)
    
    filename = "Standardization_Parameters_GC.xlsx"
    full_path = os.path.join(save_path, filename)
    params_df.to_excel(full_path, index=False)
    
    print(f"💾 Standardization Parameters Saved:")
    print(f"   File: {filename}")
    print(f"   Path: {os.path.abspath(full_path)}")
    
    return params_df

standardization_params = save_standardization_parameters(
    scaler, 
    desc_features, 
    experimental_features,
    X_train
)





print("\n3. Saving standardized data...")

def save_standardized_data(X_scaled, y, feature_columns, target_column, filename):

    ml_data = X_scaled.copy()
    ml_data[target_column] = y
    
    ml_data.to_excel(filename, index=False)
    absolute_path = os.path.abspath(filename)
    
    print(f"   📁 File saved: {filename}")
    print(f"   📍 Full path: {absolute_path}")
    print(f"   ✅ Data standardized and saved")
    
    return filename


std_filename = "CO2_database_GC_std.xlsx"
save_standardized_data(pd.concat([X_train_scaled, X_test_scaled]), 
                      pd.concat([y_train, y_test]), 
                      feature_columns, target_column, std_filename)





print("\n4. Model definition and parameter configuration...")


param_distributions = {
    'GradientBoosting': {
        'n_estimators': [180, 220, 260],
        'learning_rate': [0.07, 0.09, 0.11],
        'max_depth': [5, 6],
        'min_samples_split': [8, 12, 16],
        'min_samples_leaf': [4, 6, 8],
        'subsample': [0.75, 0.85, 0.9],
        'max_features': [0.8, 0.9, 1.0],
        'validation_fraction': [0.1],
        'n_iter_no_change': [10, 15]
    },
    
    'XGBoost': {
        'n_estimators': [180, 220, 260],
        'learning_rate': [0.07, 0.09, 0.11],
        'max_depth': [5, 6],
        'min_child_weight': [4, 6, 8],
        'subsample': [0.75, 0.85, 0.9],
        'colsample_bytree': [0.75, 0.85, 0.9],
        'colsample_bylevel': [0.75, 0.85],
        'reg_alpha': [0.8, 1.2, 1.6],
        'reg_lambda': [1.5, 2.0, 2.5],
        'gamma': [0.05, 0.1, 0.15]
    },
    
    'LightGBM': {
        'n_estimators': [180, 220, 260],
        'learning_rate': [0.07, 0.09, 0.11],
        'num_leaves': [30, 35, 40],
        'max_depth': [6, 7],
        'min_child_samples': [15, 25, 35],
        'min_child_weight': [0.005, 0.01, 0.02],
        'subsample': [0.75, 0.85, 0.9],
        'colsample_bytree': [0.75, 0.85, 0.9],
        'reg_alpha': [1.0, 2.0, 3.0],
        'reg_lambda': [1.5, 2.5, 3.5],
        'min_split_gain': [0.05, 0.1, 0.15]
    },
    
    'CatBoost': {
        'iterations': [180, 220, 260],
        'learning_rate': [0.07, 0.09, 0.11],
        'depth': [5, 6],
        'l2_leaf_reg': [4, 6, 8],
        'random_strength': [1.5, 2.0, 2.5],
        'bagging_temperature': [0.8, 1.0, 1.2],
        'leaf_estimation_iterations': [3, 4],
        'min_data_in_leaf': [25, 35, 45],
        'grow_policy': ['SymmetricTree']
    }
}


models = {
    'GradientBoosting': GradientBoostingRegressor(random_state=512),
    'XGBoost': XGBRegressor(random_state=512),
    'LightGBM': LGBMRegressor(random_state=512, verbose=-1),
    'CatBoost': CatBoostRegressor(random_state=512, verbose=False, allow_writing_files=False)
}





print("\n5. Defining model evaluation functions...")

def evaluate_model_full(model, X_train, y_train, X_test, y_test, model_name):


    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_aard = 100 * np.mean(np.abs((y_train - y_train_pred) / np.clip(np.abs(y_train), 1e-10, None)))
    

    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_aard = 100 * np.mean(np.abs((y_test - y_test_pred) / np.clip(np.abs(y_test), 1e-10, None)))
    
    print(f"  {model_name} Evaluation Results:")
    print(f"    Training Set - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, AARD%: {train_aard:.2f}%")
    print(f"    Test Set - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, AARD%: {test_aard:.2f}%")
    
    return {
        'train': {'R2': train_r2, 'MAE': train_mae, 'MSE': train_mse, 'RMSE': train_rmse, 'AARD%': train_aard},
        'test': {'R2': test_r2, 'MAE': test_mae, 'MSE': test_mse, 'RMSE': test_rmse, 'AARD%': test_aard},
        'predictions': {'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred}
    }





print("\n6. Starting model optimization...")

results = {}

for model_name, model in models.items():
    print(f"\n🔧 optimize {model_name}...")
    start_time = time.time()
    
    try:

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=50,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=512
        )
        
        random_search.fit(X_train_scaled, y_train)
        best_model = random_search.best_estimator_
        

        metrics = evaluate_model_full(best_model, X_train_scaled, y_train, X_test_scaled, y_test, model_name)
        

        results[model_name] = {
            'best_model': best_model,
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,
            'train_metrics': metrics['train'],
            'test_metrics': metrics['test'],
            'predictions': metrics['predictions']
        }
        
        end_time = time.time()
        print(f"✅ {model_name} optimization completed - Time elapsed: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ {model_name} optimization failed: {e}")
        continue





print("\n" + "="*80)
print("Complete Parameter Summary for All Models (Optimized Parameters + Actual Default Parameters)")
print("="*80)

if results:
    for model_name, res in results.items():
        print(f"\n🏆 {model_name} Complete Parameters:")
        print("-" * 50)
        

        best_model = res['best_model']
        actual_params = best_model.get_params()
        

        optimized_param_names = list(param_distributions[model_name].keys())
        
        print("   🔧 Optimized Parameters:")
        for param in optimized_param_names:
            if param in actual_params:
                value = actual_params[param]
                print(f"     {param}: {value}")
        
        print(f"\n   📋 Actual Default Parameters Used:")

        important_default_params = {
            'GradientBoosting': ['subsample', 'max_features', 'alpha', 'min_impurity_decrease'],
            'XGBoost': ['reg_alpha', 'reg_lambda', 'gamma', 'colsample_bytree', 'colsample_bylevel'],
            'LightGBM': ['subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_split_gain'],
            'CatBoost': ['random_strength', 'bagging_temperature', 'leaf_estimation_iterations', 'bootstrap_type']
        }
        
        for param in important_default_params.get(model_name, []):
            if param in actual_params and param not in optimized_param_names:
                value = actual_params[param]
                print(f"     {param}: {value}")
        
        print(f"\n   📊 Performance Metrics:")
        print(f"     Cross-Validation MSE: {res['best_score']:.6f}")
        print(f"     Training Set R²: {res['train_metrics']['R2']:.4f}")
        print(f"     Test Set R²: {res['test_metrics']['R2']:.4f}")

else:
    print("❌ No models were successfully trained, unable to output parameters")





print("\n7. Performance Summary......")

if results:

    train_comparison = pd.DataFrame()
    test_comparison = pd.DataFrame()
    
    for model_name, res in results.items():
        train_comparison[model_name] = pd.Series(res['train_metrics'])
        test_comparison[model_name] = pd.Series(res['test_metrics'])
    
    train_comparison = train_comparison.T
    test_comparison = test_comparison.T
    
    print("\n📊 Training Set Performance:")
    print(train_comparison.round(4))
    
    print("\n📊 Test Set Performance:")
    print(test_comparison.round(4))
    
    print("\n🔍 Overfitting Analysis")
    overfitting_analysis = pd.DataFrame()
    
    for model_name in results.keys():
        train_metrics = results[model_name]['train_metrics']
        test_metrics = results[model_name]['test_metrics']
        
 

    print("\n" + "="*60)
    print("Saving optimal models and prediction results")
    print("="*60)
    

    for model_name, result in results.items():
        base_name = f"GC-{model_name} Opt. 5"
        print(f"\n💾 Saved {base_name} ...")
        

        try:
            import joblib
            model_filename = f"{base_name}.pkl"
            joblib.dump(result['best_model'], model_filename)
            print(f"   ✅ Model saved: {model_filename}")
        except Exception as e:
            print(f"   ❌ Model saving failed: {e}")

            continue
        

        predictions_data = []
        

        train_indices = X_train_scaled.index
        y_train_pred = result['predictions']['y_train_pred']
        for idx, (true_val, pred_val) in enumerate(zip(y_train, y_train_pred)):
            original_idx = train_indices[idx]
            predictions_data.append({
                'No': database.loc[original_idx, 'No'],
                'IL': database.loc[original_idx, 'IL'],
                'SMILES': database.loc[original_idx, 'SMILES'],
                'Dataset': 'Train',
                'True_Value': true_val,
                'Predicted_Value': pred_val,
                'Error': true_val - pred_val
            })
        

        test_indices = X_test_scaled.index
        y_test_pred = result['predictions']['y_test_pred']
        for idx, (true_val, pred_val) in enumerate(zip(y_test, y_test_pred)):
            original_idx = test_indices[idx]
            predictions_data.append({
                'No': database.loc[original_idx, 'No'],
                'IL': database.loc[original_idx, 'IL'],
                'SMILES': database.loc[original_idx, 'SMILES'],
                'Dataset': 'Test',
                'True_Value': true_val,
                'Predicted_Value': pred_val,
                'Error': true_val - pred_val
            })
        

        predictions_df = pd.DataFrame(predictions_data)
        predictions_filename = f"{base_name}_predictions.xlsx"
        predictions_df.to_excel(predictions_filename, index=False)
        print(f"   ✅ Prediction results saved: {predictions_filename}")
    
    print(f"\n✅ All models and prediction results have been saved successfully")    





print("\n8. Generating visual evaluation charts...")

def calculate_90_percent_range(errors):
    lower_bound = np.percentile(errors, 5)
    upper_bound = np.percentile(errors, 95)
    in_range = np.sum((errors >= lower_bound) & (errors <= upper_bound))
    total = len(errors)
    percentage = (in_range / total) * 100
    return lower_bound, upper_bound, percentage

def calculate_error_in_range(errors, lower_bound, upper_bound):
    in_range = np.sum((errors >= lower_bound) & (errors <= upper_bound))
    percentage = (in_range / len(errors)) * 100
    return in_range, percentage

def generate_model_visualizations(model_name, model_info, y_train, y_test, save_path="."):
    
    y_train_pred = model_info['predictions']['y_train_pred']
    y_test_pred = model_info['predictions']['y_test_pred']
    
    train_errors = y_train - y_train_pred
    test_errors = y_test - y_test_pred
    
    train_lower, train_upper, train_percentage = calculate_90_percent_range(train_errors)
    test_lower, test_upper, test_percentage = calculate_90_percent_range(test_errors)
    
    train_in_025, train_percent_025 = calculate_error_in_range(train_errors, -0.25, 0.25)
    test_in_025, test_percent_025 = calculate_error_in_range(test_errors, -0.25, 0.25)    
    
    border_width = 2.0
    tick_fontsize = 16
    label_fontsize = 18
    title_fontsize = 16
    legend_fontsize = 16
    
    # 1. Predicted vs Experimental Capacity
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)
        spine.set_color('black')
    
    plt.scatter(y_train, y_train_pred, color='none', edgecolor='blue', 
               alpha=0.6, s=30, label="Train", linewidth=1, marker='s')
    plt.scatter(y_test, y_test_pred, color='none', edgecolor='red', 
               alpha=0.6, s=40, label="Test", linewidth=1, marker='o')
    
    min_val = min(min(y_train), min(y_test), min(y_train_pred), min(y_test_pred))
    max_val = max(max(y_train), max(y_test), max(y_train_pred), max(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', 
             linestyle='--', linewidth=2, label='Ideal Fit')
    
    plt.ylim(-2, 11)
    plt.xlim(-1, 12)
    
    plt.xlabel("Experimental Capacity (mol/kg)", fontsize=label_fontsize, fontname='Arial')
    plt.ylabel("Predicted Capacity (mol/kg)", fontsize=label_fontsize, fontname='Arial')
    plt.title(f"Pred. vs Exp. Capacity (GC-{model_name} Opt. 5)", fontsize=title_fontsize, fontweight='bold', fontname='Arial')
    plt.legend(loc='upper left', prop={'family': 'Arial', 'size': legend_fontsize})
    
    plt.tick_params(axis='both', which='major', width=2, color='black')
    plt.xticks(fontname='Arial', fontsize=tick_fontsize)
    plt.yticks(fontname='Arial', fontsize=tick_fontsize)
    
    plt.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    filename1 = f"Pred vs Exp Capacity (GC-{model_name} Opt. 5).png"
    full_path1 = os.path.join(save_path, filename1)
    plt.savefig(full_path1, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   📊 Saved: {filename1}")
    print(f"   📍 Path: {os.path.abspath(full_path1)}")
    
    # 2. Residuals vs Predicted Values
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)
        spine.set_color('black')
    
    plt.scatter(y_train_pred, train_errors, alpha=0.6, s=30, color='none', 
                edgecolor='blue', linewidth=1, label='Train', marker='s')
    plt.scatter(y_test_pred, test_errors, alpha=0.6, s=40, color='none', 
                edgecolor='red', linewidth=1, label='Test', marker='o')
    
    plt.axhspan(train_lower, train_upper, alpha=0.1, color='blue', label='Train 90% Range')
    plt.axhspan(test_lower, test_upper, alpha=0.1, color='red', label='Test 90% Range')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Zero Error')
    
    plt.ylim(-7, 7)
    plt.xlim(-1, 12)
    
    plt.xlabel('Predicted Capacity (mol/kg)', fontsize=label_fontsize, fontname='Arial')
    plt.ylabel('Residual (mol/kg)', fontsize=label_fontsize, fontname='Arial')
    plt.title(f'Residual Plot (GC-{model_name} Opt. 5)', fontsize=title_fontsize, fontweight='bold', fontname='Arial')
    plt.legend(loc='lower right', prop={'family': 'Arial', 'size': legend_fontsize})
    
    plt.tick_params(axis='both', which='major', width=2, color='black')
    plt.xticks(fontname='Arial', fontsize=tick_fontsize)
    plt.yticks(fontname='Arial', fontsize=tick_fontsize)
    
    plt.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    filename2 = f"Residual Plot (GC-{model_name} Opt. 5).png"
    full_path2 = os.path.join(save_path, filename2)
    plt.savefig(full_path2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   📊 Saved: {filename2}")
    print(f"   📍 Path: {os.path.abspath(full_path2)}")
    
    # 3. Error Distribution
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)
        spine.set_color('black')
    
    bin_edges = []
    current = 0
    max_error = max(np.abs(train_errors.max()), np.abs(test_errors.max()))
    
    while current <= max_error + 0.25:
        bin_edges.extend([-current - 0.25, -current])
        if current != 0:
            bin_edges.extend([current, current + 0.25])
        current += 0.25
    
    bin_edges = sorted(set(bin_edges))
    
    plt.hist(train_errors, bins=bin_edges, alpha=0.7, color='blue', 
             edgecolor='white', label='Train', density=False, hatch='\\\\\\\\')
    plt.hist(test_errors, bins=bin_edges, alpha=0.7, color='red', 
             edgecolor='white', label='Test', density=False, hatch='////')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero Error')
    
    plt.ylim(0, 1300)
    plt.xlim(-5.5, 5.5)
    
    plt.xlabel('Residual (mol/kg)', fontsize=label_fontsize, fontname='Arial')
    plt.ylabel('Number of Data Points', fontsize=label_fontsize, fontname='Arial')
    plt.title(f'Error Distribution (GC-{model_name} Opt. 5)', fontsize=title_fontsize, fontweight='bold', fontname='Arial')
    plt.legend(loc='upper right', prop={'family': 'Arial', 'size': legend_fontsize})
    
    plt.tick_params(axis='both', which='major', width=2, color='black')
    plt.xticks(fontname='Arial', fontsize=tick_fontsize)
    plt.yticks(fontname='Arial', fontsize=tick_fontsize)
    
    plt.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    filename4 = f"Error Distribution (GC-{model_name} Opt. 5).png"
    full_path4 = os.path.join(save_path, filename4)
    plt.savefig(full_path4, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   📊 Saved: {filename4}")
    print(f"   📍 Path: {os.path.abspath(full_path4)}")
    
    print(f"\n   GC-{model_name} Opt. 5 Detailed Statistics:")
    print(f"   Training Set R²: {model_info['train_metrics']['R2']:.4f}")
    print(f"   Test Set R²: {model_info['test_metrics']['R2']:.4f}")
    print(f"   Training Set RMSE: {model_info['train_metrics']['RMSE']:.4f} mol/kg")
    print(f"   Test Set RMSE: {model_info['test_metrics']['RMSE']:.4f} mol/kg")
    print(f"   Training Set 90% Range: [{train_lower:.4f}, {train_upper:.4f}] mol/kg ({train_percentage:.1f}%)")
    print(f"   Test Set 90% Range: [{test_lower:.4f}, {test_upper:.4f}] mol/kg ({test_percentage:.1f}%)")
    print(f"\n   📊 ±0.25 mol/kg Error Analysis:")
    print(f"   Training Set: {train_in_025}/{len(train_errors)} data points within ±0.25 range ({train_percent_025:.1f}%)")
    print(f"   Test Set: {test_in_025}/{len(test_errors)} data points within ±0.25 range ({test_percent_025:.1f}%)")
    
    return [full_path1, full_path2, full_path4]

if results:
    print("Generating visual charts for each model")
    
    all_image_paths = {}
    for model_name, model_info in results.items():
        print(f"\n📈 Generating visual charts for GC-{model_name} Opt. 5...")
        image_paths = generate_model_visualizations(model_name, model_info, y_train, y_test)
        all_image_paths[model_name] = image_paths





print("\n9. Feature Importance and SHAP Analysis...")

def create_shap_summary_plot(model_name, shap_values, X_sample, feature_columns, save_path="."):
    try:
        import shap
        
        border_width = 2.0
        tick_fontsize = 16
        label_fontsize = 18
        title_fontsize = 16
        
        plt.figure(figsize=(8, 10))
        
        shap.summary_plot(shap_values, X_sample, feature_names=feature_columns, show=False)
        
        ax = plt.gca()
        
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)
            spine.set_color('black')
        
        plt.title(f'SHAP Summary Plot (GC-{model_name} Opt. 5)', 
                 fontsize=title_fontsize, fontweight='bold', fontname='Arial')
        
        plt.xlabel('SHAP value (impact on model output)', 
                  fontsize=label_fontsize, fontname='Arial')
        
        plt.tick_params(axis='both', which='major', width=2, color='black', labelsize=tick_fontsize)
        plt.xticks(fontname='Arial')
        plt.yticks(fontname='Arial')
        
        plt.tight_layout()
        
        filename = f"SHAP Summary Plot (GC-{model_name} Opt. 5).png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   📊 Saved: {filename}")
        print(f"   📍 Path: {os.path.abspath(full_path)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create SHAP summary plot: {e}")
        import traceback
        print(f"   Detailed error: {traceback.format_exc()}")
        return False

def create_shap_importance_plot(model_name, shap_values, feature_columns, save_path="."):
    try:
        import numpy as np
        
        border_width = 2.0
        tick_fontsize = 16
        label_fontsize = 18
        title_fontsize = 16
        legend_fontsize = 16
        
        shap_importance = np.mean(np.abs(shap_values), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=True)
        
        shap_importance_df = importance_df.sort_values('shap_importance', ascending=False)
        shap_importance_filename = f"SHAP Feature Importance (GC-{model_name} Opt. 5).xlsx"
        shap_importance_full_path = os.path.join(save_path, shap_importance_filename)
        shap_importance_df.to_excel(shap_importance_full_path, index=False)
        print(f"   💾 Path: {os.path.abspath(shap_importance_full_path)}")
        
        print(f"   {model_name} SHAP top 20 important features:")
        for i, row in shap_importance_df.head(20).iterrows():
            print(f"     {row['feature']}: {row['shap_importance']:.6f}")
        

        top_20_features = importance_df.tail(20)
        
        plt.figure(figsize=(8, 10))
        ax = plt.gca()
        
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)
            spine.set_color('black')
        
        bars = ax.barh(range(len(top_20_features)), top_20_features['shap_importance'], 
                      color='#39FF14', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(top_20_features)))
        ax.set_yticklabels(top_20_features['feature'], fontname='Arial', fontsize=tick_fontsize)
        
        ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)', 
                     fontsize=label_fontsize, fontname='Arial')
        ax.set_xlim(0, top_20_features['shap_importance'].max() * 1.1)
        
        for i, (idx, row) in enumerate(top_20_features.iterrows()):
            ax.text(row['shap_importance'] + 0.005, i, f'{row["shap_importance"]:.3f}', 
                   va='center', fontsize=legend_fontsize, fontname='Arial')
        
        ax.set_title(f'SHAP Feature Importance (GC-{model_name} Opt. 5)', 
                    fontsize=title_fontsize, fontweight='bold', fontname='Arial')
        
        ax.tick_params(axis='both', which='major', width=2, color='black')
        ax.tick_params(axis='x', which='major', labelsize=tick_fontsize)
        for label in ax.get_xticklabels():
            label.set_fontname('Arial')
        
        ax.grid(True, alpha=0.3, axis='x', color='gray', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        filename = f"SHAP Feature Importance (GC-{model_name} Opt. 5).png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   📊 Saved: {filename}")
        print(f"   📍 Path: {os.path.abspath(full_path)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create SHAP importance plot: {e}")
        return False

if results:
    print("Starting SHAP analysis")
    
    for model_name, model_info in results.items():
        print(f"\n🔬 Analyzing GC-{model_name} Opt. 5 model...")
        
        try:
            import shap
        except ImportError:
            print("   ❌ SHAP library not installed, please run: pip install shap")
            continue
        
        model = model_info['best_model']
        
        try:
            if model_name in ['GradientBoosting', 'XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.TreeExplainer(model)
            
            sample_size = min(500, len(X_train_scaled))
            X_sample = X_train_scaled.iloc[:sample_size]
            
            print(f"   Computing SHAP values (using {sample_size} samples)...")
            shap_values = explainer.shap_values(X_sample)
            
            summary_success = create_shap_summary_plot(
                model_name, shap_values, X_sample, feature_columns, "."
            )
            
            importance_success = create_shap_importance_plot(
                model_name, shap_values, feature_columns, "."
            )
            
            if summary_success and importance_success:
                print(f"   ✅ GC-{model_name} Opt. 5 SHAP analysis completed")
            else:
                print(f"   ⚠️ GC-{model_name} Opt. 5 SHAP analysis had issues")
                
        except Exception as e:
            print(f"   ❌ GC-{model_name} Opt. 5 SHAP analysis failed: {e}")
            import traceback
            print(f"   Detailed error: {traceback.format_exc()}")




# end

