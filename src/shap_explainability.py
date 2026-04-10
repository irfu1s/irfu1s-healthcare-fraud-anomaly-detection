import shap
import pandas as pd
import numpy as np

DRIVER_BULLET_MAP = {
    'Total Claim Amount': 'Claim amount is materially above the provider baseline.',
    'Deductible': 'Deductible pattern is unusual for this type of claim.',
    'Coinsurance': 'Coinsurance pattern differs from comparable claims.',
    'Provider Baseline': "Submitted amount is drifting away from the provider's usual level.",
    'Provider Variance': 'Provider billing variability is elevated for this claim.',
    'Provider Deviation': 'Recent provider behavior differs from historical norms.',
    'Clinical Mismatch': 'Diagnosis and procedure context do not align with typical billing behavior.',
    '7-Day Rolling Mean': 'Recent provider billing volume is above its short-term trend.',
    '7-Day Rolling Volatility': 'Recent provider billing activity is unusually volatile.',
    'Length of Stay': 'Visit duration is inconsistent with expected behavior.',
    'Patient Age': 'Patient context differs from comparable claims.',
    'Diabetes Context': 'Clinical history contributes to an uncommon claim profile.',
    'CHF Context': 'Clinical history contributes to an uncommon claim profile.',
    'Oncology Context': 'Clinical history contributes to an uncommon claim profile.',
    'COPD Context': 'Clinical history contributes to an uncommon claim profile.',
    'Patient Hoarding (Collusion Risk)': 'Patient-provider interaction pattern looks unusually concentrated.',
}


def _extract_shap_row(shap_values):
    shap_array = np.asarray(shap_values[0] if isinstance(shap_values, list) else shap_values)
    if shap_array.ndim == 1:
        return shap_array
    return shap_array[0]


def _extract_expected_value(explainer):
    expected_value = getattr(explainer, "expected_value", 0.0)
    expected_array = np.asarray(expected_value)
    if expected_array.size == 0:
        return 0.0
    return float(expected_array.reshape(-1)[0])


def _format_feature_value(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.2f}"
    return str(value)


# Passed model and scaler as arguments to avoid slow disk reads!
def explain_transaction(transaction, static_cols, model, scaler):
    try:
        # 1. Grab the raw input data
        df_eval = pd.DataFrame([transaction])[static_cols]
        X_scaled = scaler.transform(df_eval.values)
        
        # 2. Run SHAP math
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        shap_row = _extract_shap_row(shap_values)
        
        # 3. Attach the ACTUAL input value (0 or 1) to the explanation
        contributions = pd.DataFrame({
            'Feature': static_cols,
            'Value': df_eval.iloc[0].values,  
            'Contribution': shap_row
        })
        
        contributions['Abs_Contribution'] = contributions['Contribution'].abs()
        contributions = contributions.sort_values(by='Abs_Contribution', ascending=False)
        return contributions[['Feature', 'Value', 'Contribution']]
        
    except Exception as e:
        print(f"SHAP Explanation Error: {e}")
        return pd.DataFrame()

def explain_in_words(explanation_df):
    if explanation_df.empty:
        return "Explanation unavailable."

    feature_statement_map = {
        'billing_amount_log': 'claim amount is materially above the provider baseline',
        'deductible_context': 'deductible pattern is unusual for this claim profile',
        'coinsurance_context': 'coinsurance pattern differs from comparable claims',
        'provider_avg_billing': "submitted amount differs from the provider's typical billing level",
        'provider_std_billing': 'provider billing variability is elevated',
        'provider_deviation_score': 'provider behavior differs from historical norms',
        'dx_px_combo_spike': 'diagnosis and procedure context appear mismatched',
        'rolling_mean_7': 'recent provider billing activity is above trend',
        'rolling_std_7': 'recent provider billing activity is unusually volatile',
        'service_duration': 'visit duration is inconsistent with expected behavior',
        'age': 'patient context differs from comparable claims',
        'has_diabetes': 'clinical history contributes to an uncommon claim profile',
        'has_chf': 'clinical history contributes to an uncommon claim profile',
        'has_cancer': 'clinical history contributes to an uncommon claim profile',
        'has_copd': 'clinical history contributes to an uncommon claim profile',
        'patient_hoarding_index': 'patient-provider interaction is unusually concentrated',
    }

    reasons = []
    seen = set()
    for _, row in explanation_df.iterrows():
        if len(reasons) >= 2:
            break

        feature = row['Feature']
        val = row['Value'] if 'Value' in row else 1
        if 'has_' in feature and val == 0:
            continue

        statement = feature_statement_map.get(feature)
        if not statement or statement in seen:
            continue

        reasons.append(statement)
        seen.add(statement)

    if not reasons:
        return "Primary drivers indicate a strong deviation from the provider baseline."

    return "Primary drivers indicate that " + " and ".join(reasons) + "."

# Passed model and scaler as arguments here too!
def get_shap_data_for_plotly(transaction, static_cols, model, scaler):
    try:
        df_eval = pd.DataFrame([transaction])[static_cols]
        X_scaled = scaler.transform(df_eval.values)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        shap_row = _extract_shap_row(shap_values)
        base_value = _extract_expected_value(explainer)
        
        # UPDATED: Cleaned UI dictionary
        feature_map = {
            'billing_amount_log': 'Total Claim Amount',
            'deductible_context': 'Deductible',
            'coinsurance_context': 'Coinsurance',
            'provider_avg_billing': 'Provider Baseline',
            'provider_std_billing': 'Provider Variance',
            'provider_deviation_score': 'Provider Deviation', # NEW
            'dx_px_combo_spike': 'Clinical Mismatch',         # NEW
            'rolling_mean_7': '7-Day Rolling Mean',
            'rolling_std_7': '7-Day Rolling Volatility',
            'service_duration': 'Length of Stay',
            'age': 'Patient Age',
            'has_diabetes': 'Diabetes Context',
            'has_chf': 'CHF Context',
            'has_cancer': 'Oncology Context',
            'has_copd': 'COPD Context',
            'patient_hoarding_index': 'Patient Hoarding (Collusion Risk)'
        }
        
        df_chart = pd.DataFrame({
            'Feature': static_cols,
            'Value': df_eval.iloc[0].values,
            'Contribution': shap_row
        })
        
        clean_features = []
        clean_contributions = []
        clean_values = []
        
        for _, row in df_chart.iterrows():
            feat = row['Feature']
            val = row['Value']
            cont = row['Contribution']
            
            if 'has_' in feat and val == 0:
                continue
                
            clean_features.append(feature_map.get(feat, feat))
            clean_contributions.append(float(cont))
            clean_values.append(_format_feature_value(val))
            
        final_df = pd.DataFrame(
            {
                'Feature': clean_features,
                'Feature_Value': clean_values,
                'Contribution': clean_contributions,
            }
        )
        final_df['Abs'] = final_df['Contribution'].abs()
        final_df = final_df.sort_values(by='Abs', ascending=False).head(5)
        final_df = final_df.sort_values(by='Contribution')
        
        return {
            "base_value": float(base_value),
            "component_score": float(base_value + final_df['Contribution'].sum()),
            "features": final_df['Feature'].tolist(),
            "feature_values": final_df['Feature_Value'].tolist(),
            "contributions": final_df['Contribution'].tolist(),
        }
    except Exception as e:
        print(f"Error generating SHAP for UI: {e}")
        return None


def get_driver_bullets_from_shap_data(shap_data, limit=3):
    if not shap_data:
        return []

    features = shap_data.get("features", [])
    contributions = shap_data.get("contributions", [])
    if not features or not contributions or len(features) != len(contributions):
        return []

    ranked = sorted(zip(features, contributions), key=lambda item: item[1], reverse=True)
    positive = [item for item in ranked if item[1] > 0]
    selected = positive[:limit] if positive else sorted(
        zip(features, contributions),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:limit]

    return [
        DRIVER_BULLET_MAP.get(feature, f"{feature} is pushing the claim toward anomaly.")
        for feature, _ in selected
    ]
