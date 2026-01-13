import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    
    