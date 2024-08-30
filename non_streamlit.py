import teradatasql
import pandas as pd
import sweetviz as sv
import os

# Function to connect to Teradata
def teradata_connect(host, username, password):
    try:
        connection = teradatasql.connect(
            host=host,
            user=username,
            password=password
        )
        print("Successfully connected to Teradata EDW")
        return connection
    except Exception as e:
        print(f"Failed to connect to Teradata: {e}")
        return None

# Function to read SQL query from a text file
def read_sql_query(file_path):
    try:
        with open(file_path, 'r') as file:
            query = file.read()
        print(f"Successfully read SQL query from {file_path}")
        return query
    except Exception as e:
        print(f"Failed to read SQL query from file: {e}")
        return None

# Main logic
def main():
    host = "tddp.tdc.vzwcorp.com"  # Replace with actual host
    username = "UFINNTLBD1"  # Replace with actual username
    password = "$1SI6Nkb"  # Replace with actual password

    # Establish connection
    conn = teradata_connect(host, username, password)
    if conn is None:
        return

    # Path to the SQL query text file
    sql_file_path = "your_query.sql"  # Replace with the actual path to your SQL file

    # Read the SQL query from the text file
    user_query = read_sql_query(sql_file_path)
    if user_query is None:
        return

    try:
        df = pd.read_sql(user_query, conn)
        print(f"Data fetched: {df.shape[0]} rows and {df.shape[1]} columns.")
        print(df.head())

        # Select the date column
        date_column = "DATE"  # Replace with your date column

        # Filter data for the last 3 years based on the provided date column
        df[date_column] = pd.to_datetime(df[date_column])
        df_filtered = df[df[date_column] >= pd.to_datetime("today") - pd.DateOffset(years=3)]

        print(f"Data after filtering for the last 3 years: {df_filtered.shape[0]} rows and {df_filtered.shape[1]} columns.")

        # Generate SweetViz Report
        report = sv.analyze(df_filtered)

        # Save report to a temporary HTML file
        report_file = "sweetviz_report.html"
        report.show_html(report_file, open_browser=False)

        print(f"SweetViz report generated: {report_file}")

    except Exception as e:
        print(f"Error running query or generating report: {e}")

    finally:
        # Closing connection
        if conn is not None:
            try:
                conn.close()
                print("Connection closed.")
            except Exception as e:
                print(f"Error closing the connection: {e}")

if __name__ == "__main__":
    main()
