import streamlit as st
import plotly.express as px
import pandas as pd
from model.model_files.model_trainer import ModelManager
from model.model_files.data_processor import DataProcessor

st.sidebar.title("MLearn Util App")
st.sidebar.write("Download the dataset from here..")

if 'all_results' not in st.session_state:
    st.session_state['all_results'] = {}

# Provide download button for sample dataset
try:
    with open("Breast_Cancer.csv", "rb") as file:
        csv_data = file.read()
    st.sidebar.download_button(
        label="Download Sample Dataset",
        data=csv_data,
        file_name="Breast_Cancer.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.sidebar.warning("Sample dataset file not found")

st.sidebar.subheader("Upload your dataset")

uploaded_file = st.sidebar.file_uploader("Upload a file", type=['csv'])

if uploaded_file is not None:
    # Load the data into session state
    try:
        data = pd.read_csv(uploaded_file)
        st.session_state['data'] = data
        st.session_state['file_name'] = uploaded_file.name
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
        st.sidebar.write(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

options = ["--Select--", "Simple Logistic Regression", "Decision Tree Model", "K-NN Model", "Naive Bayes Guassian Model", "Random Forest", "XGBoost Classifier"]
st.sidebar.subheader("Select the model you want to train")
selected_model = st.sidebar.selectbox("Please select the model: ", options)

# Store selected model in session state
if selected_model != "--Select--":
    st.session_state['selected_model'] = selected_model
else:
    st.session_state['selected_model'] = None

# Only show parameters if data is loaded and model is selected
if 'data' in st.session_state and st.session_state.get('selected_model'):
    st.sidebar.subheader("Configuration")
    test_split = st.sidebar.slider("Test Split Ratio", 0.1, 0.5, 0.2)
    st.session_state['test_split'] = test_split
    

st.title("Welcome to MLearn Util App")
st.write("Use the sidebar to upload data, select a model, and train it.")

# Show training results if available
if 'training_results' in st.session_state:
    results = st.session_state['training_results']
    st.success(f"✅ Previously trained: **{results['model_name']}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train Accuracy", f"{results['train_accuracy']:.4f}")
    with col2:
        st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")

st.title("📊 Dashboard")

# Check if data is available
if 'data' not in st.session_state:
    st.warning("No data available. Please upload data using the sidebar.")
    st.stop()

data = st.session_state['data']
st.success(f"Working with: {st.session_state.get('file_name', 'Unknown file')}")

# Create main tabs for Data Exploration, Model Training, and Model Comparison
main_tab1, main_tab2, main_tab3 = st.tabs(["Data Exploration", "Model Training Metrics", "Model Comparison Metrics"])

with main_tab1:
    # Data overview section
    st.markdown("### Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", data.shape[0])
    with col2:
        st.metric("Total Columns", data.shape[1])
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())

    # Data exploration section
    st.markdown("### Data Exploration")

    tab2, tab3 = st.tabs(["Summary", "Details"])

    with tab2:
        st.markdown("#### Statistical Summary")
        st.dataframe(data.describe())
        
        st.markdown("#### Target Variable: Status")

        st.markdown("#### Data Types")
        st.dataframe(data.dtypes.to_frame(name='Data Type'))

    with tab3:
        st.markdown("#### Raw Data")
        st.dataframe(data)

with main_tab2:
    # Model training section
    if st.session_state.get('selected_model') :
        st.markdown("### Model Training")
        model_name = st.session_state['selected_model']
        
        st.info(f"**Model:** {model_name} ")
        
        if st.sidebar.button(f"Train {model_name}"):
            with st.spinner("Training model..."):
                try:
                    # Prepare data
                    X, y = DataProcessor.prepare_data(DataProcessor(),df=data)
                    
                    # Initialize model manager and train
                    model_manager = ModelManager()
                    test_size = st.session_state.get('test_split', 0.2)
                    
                    results = model_manager.train_model(X, y, model_name, test_size=test_size)

                    if results:
                        st.session_state['training_results'] = results
                        st.session_state['all_results'][model_name] = results
                        
                        st.success(f"✅ {model_name} trained successfully!")
                        
                        # Store results in session state
                        st.session_state['training_results'] = results
                        
                        # Display results
                        st.markdown("### Training Results")
                        
                        # Main metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Train Accuracy", f"{results['train_accuracy']:.4f}")
                            st.metric("Training Samples", results['train_size'])
                        with col2:
                            st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
                            st.metric("Test Samples", results['test_size'])
                        
                        # Additional performance metrics
                        st.markdown("#### Additional Performance Metrics")
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric("Train Precision", f"{results['train_precision']:.4f}")
                            st.metric("Test Precision", f"{results['test_precision']:.4f}")
                        
                        with metric_col2:
                            st.metric("Train Recall", f"{results['train_recall']:.4f}")
                            st.metric("Test Recall", f"{results['test_recall']:.4f}")
                        
                        with metric_col3:
                            st.metric("Train F1-Score", f"{results['train_f1_score']:.4f}")
                            st.metric("Test F1-Score", f"{results['test_f1_score']:.4f}")
                        
                        with metric_col4:
                            st.metric("Train MCC", f"{results['train_mcc_score']:.4f}")
                            st.metric("Test MCC", f"{results['test_mcc_score']:.4f}")
                        
                        # AUC scores (if available)
                        if results['train_auc_score'] is not None and results['test_auc_score'] is not None:
                            auc_col1, auc_col2 = st.columns(2)
                            with auc_col1:
                                st.metric("Train AUC", f"{results['train_auc_score']:.4f}")
                            with auc_col2:
                                st.metric("Test AUC", f"{results['test_auc_score']:.4f}")
                        
                        # Classification Report
                        st.markdown("#### Classification Report")
                        st.text(results['classification_report'])
                        
                        # Confusion Matrix
                        st.markdown("#### Confusion Matrix")
                        import plotly.figure_factory as ff
                        cm = results['confusion_matrix']
                        
                        fig = px.imshow(cm, 
                                       labels=dict(x="Predicted", y="Actual", color="Count"),
                                       text_auto=True,
                                       color_continuous_scale='Blues')
                        fig.update_layout(title="Confusion Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Model training failed. Please check your data and try again.")
                        
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("Please select a model from the sidebar to start training.")

with main_tab3:
    # Model Comparison section
    st.markdown("### Comparison of All Trained Models")
    
    if 'all_results' in st.session_state and st.session_state['all_results']:
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in st.session_state['all_results'].items():
            comparison_data.append({
                'Model': result['model_name'],
                'Test Accuracy': f"{result['test_accuracy']:.4f}",
                'Test Precision': f"{result['test_precision']:.4f}",
                'Test Recall': f"{result['test_recall']:.4f}",
                'Test F1-Score': f"{result['test_f1_score']:.4f}",
                'Test MCC': f"{result['test_mcc_score']:.4f}",
                'Test AUC': f"{result['test_auc_score']:.4f}" if result['test_auc_score'] is not None else "N/A"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.markdown("#### Evaluation Metrics Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualize comparison with bar charts
        st.markdown("#### Visual Comparison")
        
        # Prepare numeric data for visualization
        numeric_comparison = []
        for model_name, result in st.session_state['all_results'].items():
            numeric_comparison.append({
                'Model': result['model_name'],
                'Accuracy': result['test_accuracy'],
                'Precision': result['test_precision'],
                'Recall': result['test_recall'],
                'F1-Score': result['test_f1_score'],
                'MCC': result['test_mcc_score'],
                'AUC': result['test_auc_score'] if result['test_auc_score'] is not None else 0
            })
        
        numeric_df = pd.DataFrame(numeric_comparison)
        
        # Create grouped bar chart
        melted_df = numeric_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
        
        fig = px.bar(melted_df, x='Model', y='Score', color='Metric', 
                     barmode='group',
                     title='Model Performance Comparison Across All Metrics',
                     labels={'Score': 'Score Value', 'Model': 'Model Name'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual metric comparisons
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig_acc = px.bar(numeric_df, x='Model', y='Accuracy',
                            title='Test Accuracy Comparison',
                            color='Accuracy',
                            color_continuous_scale='Blues')
            fig_acc.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # F1-Score comparison
            fig_f1 = px.bar(numeric_df, x='Model', y='F1-Score',
                           title='Test F1-Score Comparison',
                           color='F1-Score',
                           color_continuous_scale='Greens')
            fig_f1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_f1, use_container_width=True)
        
        with col2:
            # MCC comparison
            fig_mcc = px.bar(numeric_df, x='Model', y='MCC',
                            title='Test MCC Comparison',
                            color='MCC',
                            color_continuous_scale='Oranges')
            fig_mcc.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_mcc, use_container_width=True)
            
            # AUC comparison (if available)
            if numeric_df['AUC'].max() > 0:
                fig_auc = px.bar(numeric_df, x='Model', y='AUC',
                                title='Test AUC Comparison',
                                color='AUC',
                                color_continuous_scale='Purples')
                fig_auc.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_auc, use_container_width=True)
        
        # Best model recommendation
        st.markdown("#### Best Model Recommendation")
        best_accuracy_idx = numeric_df['Accuracy'].idxmax()
        best_f1_idx = numeric_df['F1-Score'].idxmax()
        best_mcc_idx = numeric_df['MCC'].idxmax()
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        with rec_col1:
            st.metric("Best Accuracy", 
                     numeric_df.loc[best_accuracy_idx, 'Model'],
                     f"{numeric_df.loc[best_accuracy_idx, 'Accuracy']:.4f}")
        with rec_col2:
            st.metric("Best F1-Score", 
                     numeric_df.loc[best_f1_idx, 'Model'],
                     f"{numeric_df.loc[best_f1_idx, 'F1-Score']:.4f}")
        with rec_col3:
            st.metric("Best MCC", 
                     numeric_df.loc[best_mcc_idx, 'Model'],
                     f"{numeric_df.loc[best_mcc_idx, 'MCC']:.4f}")
        
    else:
        st.info("No models have been trained yet. Please train at least one model to see comparisons.")