"""
Visualization and Charting Utilities
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, Any, Optional, List

class ChartCreator:
    """Creates various types of charts and visualizations"""
    
    @staticmethod
    def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                           color_col: Optional[str] = None, title: str = None) -> go.Figure:
        """
        Create an interactive scatter plot
        
        Args:
            df: Input dataframe
            x_col: X-axis column name
            y_col: Y-axis column name
            color_col: Optional color grouping column
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        if title is None:
            title = f"{x_col} vs {y_col}"
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title,
            hover_data=df.columns.tolist()
        )
        
        fig.update_layout(
            title_x=0.5,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_histogram(df: pd.DataFrame, column: str, bins: int = 30, 
                        title: str = None) -> go.Figure:
        """
        Create a histogram
        
        Args:
            df: Input dataframe
            column: Column name for histogram
            bins: Number of bins
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        if title is None:
            title = f"Distribution of {column}"
        
        fig = px.histogram(
            df, 
            x=column, 
            nbins=bins,
            title=title,
            marginal="box"  # Add box plot on top
        )
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    @staticmethod
    def create_box_plot(df: pd.DataFrame, column: str, group_by: Optional[str] = None,
                       title: str = None) -> go.Figure:
        """
        Create a box plot
        
        Args:
            df: Input dataframe
            column: Column for box plot
            group_by: Optional grouping column
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        if title is None:
            title = f"Box Plot of {column}"
            if group_by:
                title += f" by {group_by}"
        
        fig = px.box(
            df,
            x=group_by,
            y=column,
            title=title
        )
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
        """
        Create a correlation heatmap for numeric columns
        
        Args:
            df: Input dataframe
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            # Return empty figure if no numeric columns
            fig = go.Figure()
            fig.add_annotation(
                text="No numeric columns found for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                        title: str = None) -> go.Figure:
        """
        Create a bar chart
        
        Args:
            df: Input dataframe
            x_col: X-axis column name
            y_col: Y-axis column name
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        if title is None:
            title = f"{y_col} by {x_col}"
        
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title
        )
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str,
                         title: str = None) -> go.Figure:
        """
        Create a line chart
        
        Args:
            df: Input dataframe
            x_col: X-axis column name
            y_col: Y-axis column name
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        if title is None:
            title = f"{y_col} over {x_col}"
        
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title
        )
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    @staticmethod
    def create_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str] = None,
                                       title: str = "Confusion Matrix") -> go.Figure:
        """
        Create a confusion matrix heatmap
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=class_names,
            y=class_names,
            title=title,
            color_continuous_scale='Blues'
        )
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(cm[i][j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
                )
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    @staticmethod
    def create_feature_importance_chart(importance_df: pd.DataFrame, 
                                       top_n: int = 10,
                                       title: str = "Feature Importance") -> go.Figure:
        """
        Create a feature importance chart
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            title: Chart title
            
        Returns:
            plotly Figure object
        """
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=title
        )
        
        fig.update_layout(
            title_x=0.5,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig


def create_charts(chart_type: str, df: pd.DataFrame, **kwargs) -> Optional[go.Figure]:
    """
    Factory function to create different types of charts
    
    Args:
        chart_type: Type of chart to create
        df: Input dataframe
        **kwargs: Additional parameters for chart creation
        
    Returns:
        plotly Figure object or None
    """
    creator = ChartCreator()
    
    chart_functions = {
        'scatter': creator.create_scatter_plot,
        'histogram': creator.create_histogram,
        'box': creator.create_box_plot,
        'correlation': creator.create_correlation_heatmap,
        'bar': creator.create_bar_chart,
        'line': creator.create_line_chart
    }
    
    if chart_type in chart_functions:
        try:
            return chart_functions[chart_type](df, **kwargs)
        except Exception as e:
            st.error(f"Error creating {chart_type} chart: {str(e)}")
            return None
    else:
        st.error(f"Unknown chart type: {chart_type}")
        return None