import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedInsuranceClaimAnalyzer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fraud_model = None
        self.status_model = None
        self.feature_importance_df = None
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
    def calculate_customer_score(self, df):
        """
        Calculate customer risk score based on historical patterns
        """
        customer_scores = pd.DataFrame()
        customer_scores['avg_claim_amount'] = df.groupby('Customer_ID')['Claim_Amount'].transform('mean')
        customer_scores['claim_frequency'] = df.groupby('Customer_ID')['Claim_ID'].transform('count')
        customer_scores['approval_rate'] = df.groupby('Customer_ID')['Claim_Status'].transform(
            lambda x: (x == 'Approved').mean()
        )
        customer_scores['fraud_rate'] = df.groupby('Customer_ID')['Fraud_Suspected'].transform('mean')
        
        # Normalize scores
        for column in customer_scores.columns:
            min_val = customer_scores[column].min()
            max_val = customer_scores[column].max()
            if max_val > min_val:
                customer_scores[column] = (customer_scores[column] - min_val) / (max_val - min_val)
            else:
                customer_scores[column] = 0
        
        # Calculate final score
        customer_scores['customer_risk_score'] = (
            customer_scores['avg_claim_amount'] * 0.3 +
            customer_scores['claim_frequency'] * 0.2 +
            customer_scores['approval_rate'] * -0.2 +
            customer_scores['fraud_rate'] * 0.3
        )
        
        return customer_scores['customer_risk_score']

    def create_time_features(self, df):
        """
        Create advanced time-based features
        """
        # Convert all date columns to datetime if they aren't already
        date_columns = ['Date_of_Claim', 'Date_of_Incident', 'Policy_Start_Date', 
                       'Policy_End_Date', 'Adjuster_Decision_Date', 'Resolution_Date']
        for col in date_columns:
            if df[col].dtype != 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])

        df['policy_duration'] = (df['Policy_End_Date'] - df['Policy_Start_Date']).dt.days
        df['claim_day_of_week'] = df['Date_of_Claim'].dt.dayofweek
        df['incident_day_of_week'] = df['Date_of_Incident'].dt.dayofweek
        df['weekend_incident'] = df['incident_day_of_week'].isin([5, 6]).astype(int)
        df['weekend_claim'] = df['claim_day_of_week'].isin([5, 6]).astype(int)
        df['claim_month'] = df['Date_of_Claim'].dt.month
        df['incident_month'] = df['Date_of_Incident'].dt.month
        df['month_end_claim'] = (df['Date_of_Claim'].dt.day >= 25).astype(int)
        
        return df

    def preprocess_data(self, df):
        """
        Enhanced preprocessing with advanced feature engineering
        """
        df_processed = df.copy()
        
        # Create time features
        df_processed = self.create_time_features(df_processed)
        
        # Calculate customer risk score
        df_processed['customer_risk_score'] = self.calculate_customer_score(df_processed)
        
        # Create categorical encodings
        categorical_columns = ['Policy_Type', 'Customer_Gender', 'Customer_Region',
                             'Incident_Type', 'Incident_Severity', 'Reported_By',
                             'Claim_Filing_Channel', 'Payment_Method']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
        
        # Advanced feature engineering
        df_processed['claim_amount_per_day'] = df_processed['Claim_Amount'] / df_processed['policy_duration'].clip(lower=1)
        df_processed['premium_per_day'] = df_processed['Premium_Amount'] / df_processed['policy_duration'].clip(lower=1)
        df_processed['claim_premium_ratio'] = df_processed['Claim_Amount'] / df_processed['Premium_Amount'].clip(lower=1)
        
        # Create interaction features
        df_processed['severity_claim_ratio'] = df_processed['Incident_Severity'] * df_processed['claim_premium_ratio']
        df_processed['region_severity_interaction'] = df_processed['Customer_Region'] * df_processed['Incident_Severity']
        
        # Historical patterns
        df_processed['previous_claim_frequency'] = df_processed['Previous_Claims_Count'] / df_processed['policy_duration'].clip(lower=1)
        df_processed['avg_previous_claim_amount'] = df_processed['Previous_Claims_Amount'] / (df_processed['Previous_Claims_Count'] + 1)
        
        return df_processed

    def prepare_features(self, df_processed):
        """
        Prepare enhanced feature set
        """
        feature_columns = [
            'Policy_Type', 'Premium_Amount', 'Customer_Age', 'Customer_Income',
            'Customer_Credit_Score', 'Customer_Region', 'Incident_Type',
            'Incident_Severity', 'Previous_Claims_Count', 'Processing_Time',
            'customer_risk_score', 'claim_amount_per_day', 'premium_per_day',
            'claim_premium_ratio', 'severity_claim_ratio', 'region_severity_interaction',
            'previous_claim_frequency', 'avg_previous_claim_amount', 'weekend_incident',
            'weekend_claim', 'month_end_claim'
        ]
        
        X = df_processed[feature_columns]
        
        # Generate polynomial features
        X_poly = self.poly_features.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        
        # Generate feature names (compatible with newer scikit-learn versions)
        try:
            # For newer scikit-learn versions
            feature_names = self.poly_features.get_feature_names_out(feature_columns)
        except AttributeError:
            # Fallback for older versions
            feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
        
        return pd.DataFrame(X_scaled, columns=feature_names)

    def train_models(self, df):
        """
        Train enhanced models with class balancing and XGBoost
        """
        df_processed = self.preprocess_data(df)
        X = self.prepare_features(df_processed)
        
        y_fraud = df_processed['Fraud_Suspected'].astype(int)
        y_status = (df_processed['Claim_Status'] == 'Approved').astype(int)
        
        X_train, X_test, y_fraud_train, y_fraud_test, y_status_train, y_status_test = train_test_split(
            X, y_fraud, y_status, test_size=0.2, random_state=42, stratify=y_fraud
        )
        
        # Enhanced fraud detection model
        self.fraud_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        
        # XGBoost for claim status prediction
        self.status_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=sum(y_status_train == 0) / sum(y_status_train == 1),
            random_state=42
        )
        
        # Train models
        self.fraud_model.fit(X_train, y_fraud_train)
        self.status_model.fit(X_train, y_status_train)
        
        # Generate predictions
        fraud_pred = self.fraud_model.predict(X_test)
        status_pred = self.status_model.predict(X_test)
        
        results = {
            'fraud_detection': {
                'accuracy': accuracy_score(y_fraud_test, fraud_pred),
                'classification_report': classification_report(y_fraud_test, fraud_pred),
                'confusion_matrix': confusion_matrix(y_fraud_test, fraud_pred)
            },
            'claim_status': {
                'accuracy': accuracy_score(y_status_test, status_pred),
                'classification_report': classification_report(y_status_test, status_pred),
                'confusion_matrix': confusion_matrix(y_status_test, status_pred)
            }
        }
        
        return results

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('synthetic_insurance_claims.csv')
    
    # Initialize and train enhanced analyzer
    analyzer = EnhancedInsuranceClaimAnalyzer()
    results = analyzer.train_models(df)
    
    # Print results
    print("\nEnhanced Model Performance:")
    print(f"Fraud Detection Accuracy: {results['fraud_detection']['accuracy']:.2%}")
    print(f"Claim Status Accuracy: {results['claim_status']['accuracy']:.2%}")
    print("\nFraud Detection Classification Report:")
    print(results['fraud_detection']['classification_report'])
    print("\nClaim Status Classification Report:")
    print(results['claim_status']['classification_report'])
