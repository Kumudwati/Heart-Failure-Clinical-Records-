import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to load the heart failure clinical records dataset
def load_dataset():
    # URL of the heart failure clinical records dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"

    # Load the dataset into a pandas DataFrame
    data = pd.read_csv(url)

    # Select specific columns
    selected_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                       'platelets', 'serum_creatinine', 'serum_sodium', 
                         'time', 'DEATH_EVENT']

    # Filter the dataset to include only selected columns
    data = data[selected_columns]

    return data

def main():
    # Load the dataset
    data = load_dataset()
    explore_data(data)

def explore_data(data):
    # Drop NaN values
    data = data.dropna()

    # Sidebar for interactive controls
    st.sidebar.title("Heart Failure Dataset Exploration")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    display_option = st.sidebar.radio("Select Display Option", ["Dataset Overview", "Data Visualization", "About"])

    if display_option == "Dataset Overview":
        # Display dataset overview
        st.title("Heart Failure Clinical Records Dataset Overview")
        st.write("Dataset Information:")
        st.write(data.describe())  # Display summary statistics

        # Display a few sample rows of the dataset
        st.write("Sample Data:")
        st.write(data.head())

    elif display_option == "Data Visualization":
        # Data Visualization
        st.title("Heart Failure Clinical Records Dataset Visualization")

        # Select visualization type
        plot_type = st.sidebar.selectbox("Select Plot Type", ["Count Plot", "Correlation Heatmap", 
                                                              "Scatter Plot (Age vs Serum Creatinine)",
                                                              "Pair Plot (Simplified)", "Box Plot"])

        if plot_type == "Count Plot":
            # Count plot for death events
            st.subheader("Count of Death Events")
            st.write("This plot shows the count of death events.")
            sns.set_palette("husl")
            sns.countplot(x='DEATH_EVENT', data=data)
            st.pyplot()

        elif plot_type == "Correlation Heatmap":
            # Correlation heatmap
            st.subheader("Correlation Matrix")
            st.write("This heatmap shows the correlation between different features.")
            corr_matrix = data.corr()
            sns.set_palette("viridis")
            fig, ax = plt.subplots(figsize=(10, 8))  # Increase the figsize for better readability
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)  # Add fmt=".2f" to display correlation values with two decimal places
            bottom, top = ax.get_ylim()  # Fix for seaborn issue with heatmap cut-off
            ax.set_ylim(bottom + 0.5, top - 0.5)  # Fix for seaborn issue with heatmap cut-off
            plt.tight_layout()  # Adjust layout to prevent overlap
            st.pyplot(fig)

        elif plot_type == "Scatter Plot (Age vs Serum Creatinine)":
            # Scatter plot: Age vs Serum Creatinine
            st.subheader("Scatter Plot: Age vs Serum Creatinine")
            st.write("This scatter plot shows the relationship between age and serum creatinine, with colors indicating death events.")
            sns.set_palette("Set2")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='age', y='serum_creatinine', hue='DEATH_EVENT', data=data, ax=ax)
            ax.set_xlabel("Age")
            ax.set_ylabel("Serum Creatinine")
            st.pyplot(fig)

        elif plot_type == "Pair Plot (Simplified)":
            # Pair plot for simplified features
            st.subheader("Pair Plot for Selected Features (Simplified)")
            st.write("This pair plot shows pairwise relationships between a subset of features, with histograms along the diagonal.")
            simplified_features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'DEATH_EVENT']
            simplified_data = data[simplified_features]
            sns.set_palette("pastel")
            sns.pairplot(data=simplified_data, hue='DEATH_EVENT', diag_kind='hist')
            st.pyplot()

        elif plot_type == "Box Plot":
            # Box plot for selected features
            st.subheader("Box Plot for Selected Features")
            st.write("This box plot shows the distribution of selected features, with colors indicating death events.")
            selected_features = st.multiselect("Select Features for Box Plot", data.columns.tolist())
            if selected_features:
                melted_data = data.melt(id_vars='DEATH_EVENT', value_vars=selected_features)
                sns.set_palette("Dark2")
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='variable', y='value', hue='DEATH_EVENT', data=melted_data)
                plt.xticks(rotation=45)
                plt.xlabel("Feature")
                plt.ylabel("Value")
                st.pyplot()

    elif display_option == "About":
        # About section
        st.title("About Heart Failure Clinical Records Analysis")
        st.markdown("""
        This Streamlit application explores the **Heart Failure Clinical Records Dataset** collected from patients 
        during their follow-up period. The dataset includes various clinical features and outcomes, such as age, 
        blood-related parameters, medical conditions, and survival outcomes.

        **Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

        **Project Purpose**:
        This project aims to analyze and visualize the heart failure dataset to understand the relationships 
        between different clinical features and the occurrence of death events. By exploring this dataset, 
        we can gain insights into predictive factors associated with heart failure outcomes.

        Explore the dataset using the sidebar options to view dataset overview, visualize data, and gain 
        valuable insights into heart failure clinical records.
        """)

if __name__ == "__main__":
    main()
