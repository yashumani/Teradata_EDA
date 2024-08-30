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

# Main logic
def main():
    host = "your_host_here"  # Replace with actual host
    username = "your_username_here"  # Replace with actual username
    password = "your_password_here"  # Replace with actual password

    # Establish connection
    conn = teradata_connect(host, username, password)
    if conn is None:
        return

    # SQL Query
    user_query = """
    SELECT order_id, customer_id, product_id, order_date, quantity, total_amount
    FROM sales_data
    WHERE order_date >= '2021-01-01'
    """  # Replace with your actual SQL query

    try:
        df = pd.read_sql(user_query, conn)
        print(f"Data fetched: {df.shape[0]} rows and {df.shape[1]} columns.")
        print(df.head())

        # Select the date column
        date_column = "order_date"  # Replace with your date column

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
