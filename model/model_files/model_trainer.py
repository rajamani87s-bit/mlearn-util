import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Tuple, Optional

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Simple Logistic Regression": LogisticRegression(random_state=42),
            "Decision Tree Model": DecisionTreeClassifier(random_state=42),
            "K-NN Model": KNeighborsClassifier(),
            "Naive Bayes Guassian Model": GaussianNB(),
            "Naive Bayes Multinomial Model": MultinomialNB(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost Classifier": XGBClassifier(random_state=42)
        }
        self.trained_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_name = None
    
    def get_available_models(self):
        return self.models.keys()

    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_name: str, 
                   test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Args:
            X: Feature dataframe
            y: Target series
            model_name: Name of the model to train
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dict containing training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
                
            # Get and train the model
            model = self.models[model_name]
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_precision = classification_report(y_train, y_pred_train, output_dict=True)['weighted avg']['precision']
            train_recall = classification_report(y_train, y_pred_train, output_dict=True)['weighted avg']['recall'] 
            test_precision = classification_report(y_test, y_pred_test, output_dict=True)['weighted avg']['precision']
            test_recall = classification_report(y_test, y_pred_test, output_dict=True)['weighted avg']['recall']

            train_f1_score = classification_report(y_train, y_pred_train, output_dict=True)['weighted avg']['f1-score']
            test_f1_score = classification_report(y_test, y_pred_test, output_dict=True)['weighted avg']['f1-score']
            
            # Calculate MCC scores
            train_mcc_score = matthews_corrcoef(y_train, y_pred_train)
            test_mcc_score = matthews_corrcoef(y_test, y_pred_test)
            
            # Calculate AUC scores
            # AUC requires probability predictions
            try:
                if hasattr(model, 'predict_proba'):
                    y_train_proba = model.predict_proba(X_train)
                    y_test_proba = model.predict_proba(X_test)
                    
                    train_auc_score = roc_auc_score(y_train, y_train_proba[:, 1])
                    test_auc_score = roc_auc_score(y_test, y_test_proba[:, 1])
                    
                else:
                    # Some models don't support predict_proba (e.g., SVC without probability=True)
                    train_auc_score = None
                    test_auc_score = None
            except Exception as e:
                st.warning(f"Could not calculate AUC score: {str(e)}")
                train_auc_score = None
                test_auc_score = None

            # Store trained model
            self.trained_model = model
            self.model_name = model_name
            
            # Prepare results
            results = {
                'model_name': model_name,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'train_f1_score': train_f1_score,
                'test_f1_score': test_f1_score,
                'train_mcc_score': train_mcc_score,
                'test_mcc_score': test_mcc_score,
                'train_auc_score': train_auc_score,
                'test_auc_score': test_auc_score,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'classification_report': classification_report(y_test, y_pred_test),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test),
                'predictions': {
                    'y_test': y_test,
                    'y_pred': y_pred_test
                }
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return {}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature dataframe
            
        Returns:
            np.ndarray: Predictions
        """
        if self.trained_model is None:
            raise ValueError("No trained model available. Please train a model first.")
        
       # Make predictions
        predictions = self.trained_model.predict(X)
        
        # Decode predictions if necessary
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """
        Get model parameters and hyperparameters
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict containing model parameters
        """
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        return model.get_params()
    
    def update_model_parameters(self, model_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Update model parameters
        
        Args:
            model_name: Name of the model
            parameters: Dictionary of parameters to update
            
        Returns:
            bool: Success status
        """
        try:
            if model_name not in self.models:
                return False
            
            self.models[model_name].set_params(**parameters)
            return True
        except Exception as e:
            st.error(f"Error updating parameters: {str(e)}")
            return False

# Alias for compatibility with streamlit_app.py
ModelManager = ModelTrainer