import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set page title and description
st.set_page_config(page_title="10m Sprint Analysis", layout="wide")
st.title("10m Sprint Performance Dashboard")
st.write("Visualization of athlete performance in 10m sprint events")

# Function to load data from CSV (or create sample data if file doesn't exist)
def load_or_create_data():
    csv_path = "sprint_data.csv"
    
    # If CSV doesn't exist, create sample data
    if not os.path.exists(csv_path):
        # Create sample data with 10 athletes
        names = [
            "Usain Bolt", "Tyson Gay", "Asafa Powell", "Justin Gatlin", 
            "Yohan Blake", "Christian Coleman", "Noah Lyles", "Andre De Grasse",
            "Trayvon Bromell", "Fred Kerley"
        ]
        
        # Generate realistic 10m sprint times (between 1.7 and 2.2 seconds)
        times = np.round(np.random.uniform(1.7, 2.2, size=10), 2)
        
        # Create DataFrame
        data = pd.DataFrame({
            "Athlete": names,
            "Time (seconds)": times
        })
        
        # Save to CSV
        data.to_csv(csv_path, index=False)
        st.sidebar.success("Sample data created successfully!")
    
    # Read the CSV file
    return pd.read_csv(csv_path)

# Load data
df = load_or_create_data()

# Display raw data in expandable section
with st.expander("View Raw Data"):
    st.dataframe(df)

# Create bar chart
st.subheader("10m Sprint Times Comparison")

fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
bars = ax.barh(df["Athlete"], df["Time (seconds)"], color="skyblue")
ax.set_xlabel("Time (seconds)")
ax.set_title("10m Sprint Performance")
ax.invert_yaxis()  # Display fastest athlete at the top

# Add time labels to the end of each bar
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}s', 
            va='center')

# Highlight the fastest time
fastest_idx = df["Time (seconds)"].idxmin()
bars[fastest_idx].set_color("gold")

# Add analysis
st.pyplot(fig)

# Display statistics
st.subheader("Performance Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    fastest_time = df["Time (seconds)"].min()
    fastest_athlete = df.loc[df["Time (seconds)"] == fastest_time, "Athlete"].values[0]
    st.metric("Fastest Time", f"{fastest_time:.2f}s", f"by {fastest_athlete}")

with col2:
    average_time = df["Time (seconds)"].mean()
    st.metric("Average Time", f"{average_time:.2f}s")

with col3:
    slowest_time = df["Time (seconds)"].max()
    slowest_athlete = df.loc[df["Time (seconds)"] == slowest_time, "Athlete"].values[0]
    st.metric("Slowest Time", f"{slowest_time:.2f}s", f"by {slowest_athlete}")

# Add sidebar filters
st.sidebar.header("Filters")
time_threshold = st.sidebar.slider(
    "Filter by maximum time",
    min_value=float(df["Time (seconds)"].min()),
    max_value=float(df["Time (seconds)"].max()),
    value=float(df["Time (seconds)"].max())
)

filtered_df = df[df["Time (seconds)"] <= time_threshold]

if len(filtered_df) < len(df):
    st.subheader(f"Athletes with times under {time_threshold:.2f}s")
    st.dataframe(filtered_df)

# Add download button for the data
csv = df.to_csv(index=False)
st.sidebar.download_button(
    label="Download Sprint Data",
    data=csv,
    file_name="sprint_data.csv",
    mime="text/csv"
)