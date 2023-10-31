import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Disable the Matplotlib warning
st.set_option('deprecation.showPyplotGlobalUse', False)
#scaling functions
# Function to perform standardization and display plots
def perform_standardization(df):
    st.subheader("Standardization (Z-score Normalization)")
    selected_column = st.selectbox("Select a Numerical Column", df.select_dtypes(include='number').columns)
    if selected_column:
        # Original Plot
        st.subheader("Original Plot")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[selected_column], kde=True)
        st.pyplot()

        # Standardization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[[selected_column]])
        df_standardized = pd.DataFrame(data=scaled_data, columns=[selected_column + '_standardized'])

        # Transformed Plot
        st.subheader("Transformed Plot (Standardized)")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_standardized[selected_column + '_standardized'], kde=True)
        st.pyplot()

        # Display code
        st.code(
            f"""
            selected_column = '{selected_column}'
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[[selected_column]])
            df_standardized = pd.DataFrame(data=scaled_data, columns=[selected_column + '_standardized'])
            """
        )

# Function to perform normalization and display plots
def perform_normalization(df):
    st.subheader("Min-Max Normalization (Scaling)")
    selected_column = st.selectbox("Select a Numerical Column", df.select_dtypes(include='number').columns)
    if selected_column:
        # Original Plot
        st.subheader("Original Plot")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[selected_column], kde=True)
        st.pyplot()

        # Normalization
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[[selected_column]])
        df_normalized = pd.DataFrame(data=scaled_data, columns=[selected_column + '_normalized'])

        # Transformed Plot
        st.subheader("Transformed Plot (Normalized)")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_normalized[selected_column + '_normalized'], kde=True)
        st.pyplot()

        # Display code
        st.code(
            f"""
            selected_column = '{selected_column}'
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[[selected_column]])
            df_normalized = pd.DataFrame(data=scaled_data, columns=[selected_column + '_normalized'])
            """
        )



# Function to display basic statistics
def display_basic_stats(df):
    st.subheader("Basic Statistics")
    st.write(df.describe())
    st.code(
        """
def display_basic_stats(df):
    st.subheader("Basic Statistics")
    st.write(df.describe())"""
    )

# Function to display column-wise summary
def display_column_summary(df):
    st.subheader("Column Summary")
    selected_column = st.selectbox("Select a Column", df.columns)
    st.write(df[selected_column].describe())
    st.code("""
def display_column_summary(df):
    st.subheader("Column Summary")
    selected_column = st.selectbox("Select a Column", df.columns)
    st.write(df[selected_column].describe())
            """)

# Function to display missing values
def display_missing_values(df):
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    st.write(missing_data)
    st.code("""
def display_missing_values(df):
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    st.write(missing_data)
        """)

# Function to display value counts for a selected column
# def display_value_counts(df):
#     st.subheader("Value Counts")
#     selected_column = st.selectbox("Select a Column", df.columns)
#     value_counts = df[selected_column].value_counts()
#     st.write(value_counts)
# Streamlit app title and file upload



# Function to display a bar chart for a selected categorical column
def display_bar_chart(df):
    st.subheader("Bar Chart")
    selected_column = st.selectbox("Select a Categorical Column", df.select_dtypes(include='object').columns)
    if selected_column:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=selected_column, data=df)
        plt.xticks(rotation=45)
        st.pyplot()

# Function to display a histogram for a selected numerical column
def display_histogram(df):
    st.subheader("Histogram")
    selected_column = st.selectbox("Select a Numerical Column", df.select_dtypes(include='number').columns)
    if selected_column:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[selected_column], kde=True)
        st.pyplot()

# Function to display a scatter plot for two selected numerical columns
def display_scatter_plot(df):
    st.subheader("Scatter Plot")
    x_column = st.selectbox("Select X-Axis Column", df.select_dtypes(include='number').columns)
    y_column = st.selectbox("Select Y-Axis Column", df.select_dtypes(include='number').columns)
    if x_column and y_column:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_column, y=y_column, data=df)
        st.pyplot()


# Function to display a bar chart for a selected categorical column
def display_bar_chart(df):
    st.subheader("Bar Chart")
    selected_column = st.selectbox("Select a Categorical Column", df.select_dtypes(include='object').columns, key="bar_chart")
    if selected_column:
        plt.figure(figsize=(10, 6))  # Set the size of the plot
        sns.countplot(x=selected_column, data=df)
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot()
        st.code("""
def display_bar_chart(df):
    st.subheader("Bar Chart")
    selected_column = st.selectbox("Select a Categorical Column", df.select_dtypes(include='object').columns, key="bar_chart")
    if selected_column:
        plt.figure(figsize=(10, 6))  # Set the size of the plot
        sns.countplot(x=selected_column, data=df)
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf()) 
                """)


# Function to display a histogram for a selected numerical column
def display_histogram(df):
    st.subheader("Histogram")
    numerical_columns = df.select_dtypes(include='number').columns
    selected_column = st.selectbox("Select a Numerical Column", numerical_columns, key="unique_key_for_selectbox")
    # selected_column = st.selectbox("Select a Numerical Column", df.select_dtypes(include='number').columns)
    if selected_column:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[selected_column], kde=True)
        st.pyplot()
    st.code(
        """
        def display_histogram(df):
            st.subheader("Histogram")
            selected_column = st.selectbox("Select a Numerical Column", df.select_dtypes(include='number').columns)
            if selected_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[selected_column], kde=True, ax=ax)
                st.pyplot(fig)
        """
    )

# Function to display a pie chart for a selected categorical column
def display_pie_chart(df):
    st.subheader("Pie Chart")
    selected_column = st.selectbox("Select a Categorical Column", df.select_dtypes(include='object').columns)
    if selected_column:
        counts = df[selected_column].value_counts()
        fig, ax = plt.subplots()
        ax(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    # Display code
    st.code(
        """
        def display_pie_chart(df):
            st.subheader("Pie Chart")
            selected_column = st.selectbox("Select a Categorical Column", df.select_dtypes(include='object').columns)
            if selected_column:
                counts = df[selected_column].value_counts()
                fig, ax = plt.subplots()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
        """
    )

# Function to display a scatter plot for two selected numerical columns
# def display_scatter_plot(df):
#     st.subheader("Scatter Plot")
#     x_column = st.selectbox("Select X-Axis Column", df.select_dtypes(include='number').columns)
#     y_column = st.selectbox("Select Y-Axis Column", df.select_dtypes(include='number').columns)
#     hue_col = st.selectbox("Select Hue Column", df.select_dtypes(include='object').columns)
#     if x_column and y_column:
#         plt.figure(figsize=(10, 6))
#         sns.scatterplot(x=x_column, y=y_column, data=df,hue=hue_col)
#         st.pyplot()
def display_scatter_plot(df):
    st.subheader("Scatter Plot")
    x_column = st.selectbox("Select X-Axis Column", df.select_dtypes(include='number').columns)
    y_column = st.selectbox("Select Y-Axis Column", df.select_dtypes(include='number').columns)
    
    # Checkbox to enable/disable hue
    use_hue = st.checkbox("Use Hue", value=True)
    hue_col = None
    
    if use_hue:
        hue_col = st.selectbox("Select Hue Column", df.select_dtypes(include='object').columns)
    
    if x_column and y_column:
        plt.figure(figsize=(10, 6))
        
        if use_hue:
            sns.scatterplot(x=x_column, y=y_column, hue=hue_col, data=df, palette='Set1')
        else:
            sns.scatterplot(x=x_column, y=y_column, data=df)
            
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        
        if use_hue:
            plt.title(f'Scatter Plot between {x_column} and {y_column} (Hued by {hue_col})')
        else:
            plt.title(f'Scatter Plot between {x_column} and {y_column}')
            
        st.pyplot()
def display_pie_chart(df):
    st.subheader("Pie Chart")
    selected_column = st.selectbox("Select a Categorical Column", df.select_dtypes(include='object').columns, key="pie_chart")
    if selected_column:
        data_counts = df[selected_column].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot()

def display_count_plot(df):
    st.subheader("Count Plot")
    selected_column = st.selectbox("Select a Categorical Column", df.select_dtypes(include='object').columns,key="count_plot")
    if selected_column:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=selected_column, data=df)
        plt.xticks(rotation=45)
        st.pyplot()



# Function to perform grouping and display plots
def perform_grouping(df):
    st.subheader("Grouping Data")
    group_by_column = st.selectbox("Select a Column to Group By", df.columns)
    aggregation_column = st.selectbox("Select a Column for Aggregation", df.select_dtypes(include='number').columns)
    aggregation_function = st.selectbox("Select an Aggregation Function", ["mean", "sum", "count"])

    if group_by_column and aggregation_column:
        grouped_data = df.groupby(group_by_column)[aggregation_column].agg(aggregation_function).reset_index()

        # Display grouped data
        st.subheader("Grouped Data")
        st.write(grouped_data)

        # Plot grouped data
        st.subheader("Grouped Data Plot")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=group_by_column, y=aggregation_column, data=grouped_data)
        plt.xticks(rotation=45)
        st.pyplot()

        # Display code
        st.code(
            f"""
            group_by_column = '{group_by_column}'
            aggregation_column = '{aggregation_column}'
            aggregation_function = '{aggregation_function}'
            grouped_data = df.groupby(group_by_column)[aggregation_column].agg(aggregation_function).reset_index()
            """
        )

# Streamlit app title and file upload
st.title("CSV File Data Analysis App")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv","xlsx"])

# Check if file is uploaded
if uploaded_file is not None:
    # Perform analysis when the file is uploaded

    try:
        # Try reading the file with UTF-8 encoding
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        # If a UnicodeDecodeError occurs, try using a different encoding (e.g., 'iso-8859-1')
        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')

    
    # Show the raw data
    st.subheader("Raw Data")
    st.write(df)
    st.code(""" 
    df = pd.read_csv(uploaded_file)
    # Show the raw data
    st.subheader("Raw Data")
    st.write(df)
            """)
    # analysis_and_transformation_types = ["Categorical Analysis", "Numerical Analysis", "Data Transformation"]
    # selected_analysis_and_transformation = st.radio("Select Analysis/Transformation Type",analysis_and_transformation_types)


    analysis_options = ["Categorical Analysis", "Numerical Analysis", "Data Transformation", "Grouping"]
    selected_analysis_option = st.radio("Select Analysis/Transformation/Grouping Type", analysis_options)

    if selected_analysis_option == "Categorical Analysis":
        # Display categorical analysis options...
        pass
    elif selected_analysis_option == "Numerical Analysis":
        # Display numerical analysis options...
        pass
    elif selected_analysis_option == "Data Transformation":
        # Display data transformation options...
        pass
    elif selected_analysis_option == "Grouping":
        perform_grouping(df)
    elif selected_analysis_option == "Data Transformation":
        transformation_options = ["Standardization (Z-score Normalization)", "Min-Max Normalization (Scaling)"]
        selected_transformation = st.selectbox("Select Transformation Type", transformation_options)

        if selected_transformation == "Standardization (Z-score Normalization)":
            perform_standardization(df)
        elif selected_transformation == "Min-Max Normalization (Scaling)":
            perform_normalization(df)


    

     


    # Perform various analyses based on user selection
    selected_analysis = ["Basic Statistics", "Column Summary", "Missing Values", "Value Counts"]
    # selected_analysis = st.multiselect("Select Analysis Options", analysis_options)

    if "Basic Statistics" in selected_analysis:
        display_basic_stats(df)

    if "Column Summary" in selected_analysis:
        display_column_summary(df)

    if "Missing Values" in selected_analysis:
        display_missing_values(df)

    # if "Value Counts" in selected_analysis:
    #     display_value_counts(df)

    # Visualization options
    selected_visualization = ["Bar Chart", "Histogram", "Scatter Plot","pie","count_plot"]
    # selected_visualization = st.multiselect("Select Visualization Options", visualization_options)

    if "Bar Chart" in selected_visualization:
        display_bar_chart(df)

    if "Histogram" in selected_visualization:
        display_histogram(df)

 
    if "pie" in selected_visualization:
        display_pie_chart(df)
    if "count_plot" in selected_visualization:
        display_count_plot(df)
    if "Scatter Plot" in selected_visualization:
        display_scatter_plot(df)