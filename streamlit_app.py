import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set page title and description
st.set_page_config(page_title="Sprint Analysis", layout="wide")
st.title("Sprint Performance Dashboard")
st.write("Visualization of athlete performance in sprint events")

# Function to load data from CSV
def load_data():
    csv_path = "sprint_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error("sprint_data.csv not found. Please upload the file.")
        return None

# Load data
df = load_data()

if df is not None:
    # Display raw data in expandable section
    with st.expander("View Raw Data"):
        st.dataframe(df)

    # Filter for 10m Sprint
    df_10m = df[df["Test"] == "10m Sprint"].copy()

    if not df_10m.empty:

        # Create bar chart for 10m Sprint
        st.subheader("10m Sprint Times Comparison")

        fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
        bars = ax.barh(df_10m["Name"], df_10m["Result"].astype(float), color="skyblue")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("10m Sprint Performance")
        ax.invert_yaxis()  # Display fastest athlete at the top

        # Add time labels to the end of each bar
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}s',
                    va='center')

        # Highlight the fastest time
        fastest_idx = df_10m["Result"].astype(float).idxmin()
        bars[fastest_idx].set_color("gold")

        # Add analysis
        st.pyplot(fig)

        # Display statistics for 10m Sprint
        st.subheader("10m Sprint Performance Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            fastest_time = df_10m["Result"].astype(float).min()
            fastest_athlete = df_10m.loc[df_10m["Result"].astype(float) == fastest_time, "Name"].values[0]
            st.metric("Fastest Time", f"{fastest_time:.2f}s", f"by {fastest_athlete}")

        with col2:
            average_time = df_10m["Result"].astype(float).mean()
            st.metric("Average Time", f"{average_time:.2f}s")

        with col3:
            slowest_time = df_10m["Result"].astype(float).max()
            slowest_athlete = df_10m.loc[df_10m["Result"].astype(float) == slowest_time, "Name"].values[0]
            st.metric("Slowest Time", f"{slowest_time:.2f}s", f"by {slowest_athlete}")

        # Add sidebar filters for 10m Sprint
        st.sidebar.header("Filters")
        time_threshold = st.sidebar.slider(
            "Filter by maximum time (10m Sprint)",
            min_value=float(df_10m["Result"].astype(float).min()),
            max_value=float(df_10m["Result"].astype(float).max()),
            value=float(df_10m["Result"].astype(float).max())
        )

        filtered_df_10m = df_10m[df_10m["Result"].astype(float) <= time_threshold]

        if len(filtered_df_10m) < len(df_10m):
            st.subheader(f"Athletes with 10m sprint times under {time_threshold:.2f}s")
            st.dataframe(filtered_df_10m)

    else:
        st.warning("No 10m Sprint data found in the uploaded CSV.")

    # Add download button for the data
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Sprint Data",
        data=csv,
        file_name="sprint_data.csv",
        mime="text/csv"
    )
else:
    st.write("Please upload the sprint_data.csv file to view the analysis.")
