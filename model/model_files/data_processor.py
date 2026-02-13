"""
Data Processing Business Logic
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
   
    def __init__(self):
        self.data = None
        self.original_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pd.DataFrame or None: Loaded dataframe or None if error
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                self.data = df.copy()
                self.original_data = df.copy()
                return df
            else:
                st.error("Unsupported file format. Please upload a CSV file.")
                return None
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict containing data information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'duplicate_rows': df.duplicated().sum()
        }
        return info
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset based on provided options
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        cleaned_df = df.copy()
        
        # Handle missing values

        cleaned_df = cleaned_df.dropna()
        

        # Fill numeric columns with mean
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        
        # Fill categorical columns with mode
        cat_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
        
        # Remove duplicates

        cleaned_df = cleaned_df.drop_duplicates()

        
        # Separate features and target
        X = cleaned_df.drop(columns=['Status'])
        y = cleaned_df['Status']
        
        # Handle categorical variables (simple one-hot encoding)
        categorical_columns = X.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    
    def get_column_statistics(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific column
        
        Args:
            df: Input dataframe
            column: Column name
            
        Returns:
            Dict containing column statistics
        """
        if column not in df.columns:
            return {}
        
        col_data = df[column]
        stats = {
            'name': column,
            'dtype': str(col_data.dtype),
            'non_null_count': col_data.count(),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique()
        }
        
        if col_data.dtype in ['int64', 'float64']:
            stats.update({
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75)
            })
        else:
            # For categorical data
            value_counts = col_data.value_counts().head(10)
            stats.update({
                'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                'frequency': value_counts.iloc[0] if not value_counts.empty else 0,
                'top_values': value_counts.to_dict()
            })
        
        return stats

