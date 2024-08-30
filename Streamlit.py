import streamlit as st
import teradatasql
import pandas as pd
import sweetviz as sv
import os

# Function to connect to Teradata
def teradata_connect(host, username, password):
    try:
        # Specify the port number directly in the host string
        port = 1025  # Replace with the appropriate port number
        connection = teradatasql.connect(
            host=host,
            user=username,
            password=password
        )
        return connection
    except Exception as e:
        st.error(f"Failed to connect to Teradata: {e}")
        return None

# Streamlit App
st.title("Teradata EDW EDA Tool with SweetViz")

# Session State for Handling User Input
if 'host' not in st.session_state:
    st.session_state['host'] = ''
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'password' not in st.session_state:
    st.session_state['password'] = ''
if 'user_query' not in st.session_state:
    st.session_state['user_query'] = ''
if 'date_column' not in st.session_state:
    st.session_state['date_column'] = ''
if 'columns' not in st.session_state:
    st.session_state['columns'] = []
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'conn' not in st.session_state:
    st.session_state['conn'] = None

# User Input for Credentials
st.sidebar.header("User Login")
st.session_state['host'] = st.sidebar.text_input("Host", value=st.session_state['host'])
st.session_state['username'] = st.sidebar.text_input("Username", value=st.session_state['username'])
st.session_state['password'] = st.sidebar.text_input("Password", type="password", value=st.session_state['password'])

# Login Button
if st.sidebar.button("Login"):
    st.session_state['conn'] = teradata_connect(st.session_state['host'], st.session_state['username'], st.session_state['password'])
    if st.session_state['conn'] is not None:
        st.session_state['logged_in'] = True
        st.success("Successfully connected to Teradata EDW")

# Check if user is logged in
if st.session_state['logged_in']:
    
    # User Input for Query
    st.header("Enter your SQL Query")
    st.session_state['user_query'] = st.text_area("SQL Query", value=st.session_state['user_query'], height=200)

    # Run Query to Fetch Columns
    if st.button("Fetch Columns"):
        if st.session_state['user_query'] and st.session_state['conn'] is not None:
            try:
                df = pd.read_sql(st.session_state['user_query'], st.session_state['conn'])
                st.session_state['columns'] = df.columns.tolist()
                st.write(f"Columns in the query result: {st.session_state['columns']}")

                # Display dropdown to select date column
                if st.session_state['columns']:
                    st.session_state['date_column'] = st.selectbox(
                        "Select the Date Column for EDA",
                        st.session_state['columns']
                    )
                    st.write(f"Selected Date Column: {st.session_state['date_column']}")

            except Exception as e:
                st.error(f"Error fetching columns: {e}")
        else:
            st.error("Please enter a SQL query to fetch columns.")
    
    # Run EDA
    if st.session_state['date_column'] and st.button("Run EDA"):
        if st.session_state['user_query'] and st.session_state['date_column'] and st.session_state['conn'] is not None:
            try:
                df = pd.read_sql(st.session_state['user_query'], st.session_state['conn'])
                st.write(f"Data fetched: {df.shape[0]} rows and {df.shape[1]} columns.")
                
                # Filter data for the last 3 years based on the provided date column
                df[st.session_state['date_column']] = pd.to_datetime(df[st.session_state['date_column']])
                df_filtered = df[df[st.session_state['date_column']] >= pd.to_datetime("today") - pd.DateOffset(years=3)]
                
                st.write(f"Data after filtering for the last 3 years: {df_filtered.shape[0]} rows and {df_filtered.shape[1]} columns.")
                
                # Generate SweetViz Report
                st.header("Exploratory Data Analysis (EDA) Report with SweetViz")
                report = sv.analyze(df_filtered)
                
                # Save report to a temporary HTML file
                report_file = "sweetviz_report.html"
                report.show_html(report_file, open_browser=False)
                
                # Display the report in the Streamlit app
                with open(report_file, 'r') as f:
                    html_content = f.read()
                    st.components.v1.html(html_content, width=800, height=600, scrolling=True)

                # Clean up the temporary file
                if os.path.exists(report_file):
                    os.remove(report_file)

            except Exception as e:
                st.error(f"Error running EDA: {e}")
        else:
            st.error("Please provide both a SQL query and select a date column.")

# Closing connection if exists
if st.session_state['conn'] is not None:
    try:
        st.session_state['conn'].close()
        st.session_state['conn'] = None
    except Exception as e:
        st.error(f"Error closing the connection: {e}")
