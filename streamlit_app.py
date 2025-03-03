import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import os

# Set page configuration for wide layout
st.set_page_config(page_title="Sprint Performance Dashboard", layout="wide")
st.title("Sprint Performance Dashboard")
st.write("Compare an athlete's performance against the rest")

# Decrease sidebar width with percentage-based CSS
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 20% !important;
        min-width: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load data from Google Sheets
def load_data_from_sheets():
    try:
        SHEET_ID = '1fjqPnrhIRGeYNxkRmWn4igb81SJ_gxfLCSVb3KZyju4'
        SHEET_NAME = 'Sheet1'
        url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.warning(f"Failed to load from Google Sheets: {str(e)}")
        st.warning("Falling back to local CSV file...")
        return None

# Function to load data from CSV
def load_data_from_csv():
    try:
        csv_path = "sprint_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        else:
            st.error("Local CSV file (sprint_data.csv) not found.")
            return None
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

# Load data (try Google Sheets first, then CSV)
def load_data():
    df = load_data_from_sheets()
    if df is None:
        df = load_data_from_csv()
    return df

# Load data
df = load_data()

if df is not None:
    # Prepare data
    df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
    df = df.dropna(subset=['Result'])

    # Add filter for athlete name at the top
    athlete_names = df['Name'].unique().tolist()
    selected_athlete = st.selectbox("Select an Athlete", options=athlete_names)

    # Filter data for the selected athlete
    df_selected = df[df['Name'] == selected_athlete].copy()
    df_others = df[df['Name'] != selected_athlete].copy()

    if not df_selected.empty:
        # Calculate average results for others per test
        others_avg = df_others.groupby('Test')['Result'].mean().reset_index()
        selected_data = df_selected[['Test', 'Result']].drop_duplicates()

        # Merge data for comparison
        comparison_df = pd.merge(selected_data, others_avg, on='Test', how='left', suffixes=('_selected', '_others'))
        comparison_df = comparison_df.fillna(0)

        # Create comparison chart with enhanced visuals
        st.subheader(f"Performance Comparison: {selected_athlete} vs. Others")
        fig = plt.figure(facecolor='none')
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], facecolor=(0, 0, 0, 0.05))

        # Define gradient colors
        selected_cmap = LinearSegmentedColormap.from_list("selected", ["#00CED1", "#1E90FF"])
        others_cmap = LinearSegmentedColormap.from_list("others", ["#FF6B6B", "#DC143C"])

        bar_width = 0.35
        x = range(len(comparison_df['Test']))
        
        # Gradient bars for selected athlete
        selected_bars = ax.bar([i - bar_width/2 for i in x], comparison_df['Result_selected'], bar_width, 
                               color=[selected_cmap(i / len(x)) for i in range(len(x))], edgecolor='black', linewidth=1.2)
        
        # Gradient bars for others average
        others_bars = ax.bar([i + bar_width/2 for i in x], comparison_df['Result_others'], bar_width, 
                             color=[others_cmap(i / len(x)) for i in range(len(x))], edgecolor='black', linewidth=1.2)

        # Enhance text with outlines
        ax.set_xlabel('Test', color='white', fontsize=12, weight='bold', path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        ax.set_ylabel('Result (seconds)', color='white', fontsize=12, weight='bold', path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        ax.set_title(f'{selected_athlete} vs. Others', color='white', fontsize=14, weight='bold', 
                     path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Test'], rotation=45, ha='right', color='white', fontsize=10, 
                           path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        ax.tick_params(axis='y', colors='white', labelsize=10)
        
        # Add grid lines for better readability
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5, color='white')

        # Customize legend
        ax.legend([selected_bars, others_bars], [selected_athlete, 'Others Average'], fontsize=10, labelcolor='white', 
                  edgecolor='black', facecolor=(0, 0, 0, 0.1), framealpha=0.8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Calculate KPIs
        overall_avg_pr = df['Average_Percentrank'].mean()
        selected_pr = df_selected['Average_Percentrank'].iloc[0]  # Assuming one PR per athlete
        
        # Calculate percentage difference from the overall average
        diff_to_avg_pr_percent = ((selected_pr - overall_avg_pr) / overall_avg_pr) * 100 if overall_avg_pr != 0 else 0

        # Display KPIs with enhanced visuals
        st.subheader("Key Performance Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Average Percentrank", f"{overall_avg_pr:.1f}", 
                      label_visibility="visible", 
                      help="Average Percentrank across all athletes")
        with col2:
            st.metric(f"Difference to Average PR ({selected_athlete})", f"{diff_to_avg_pr_percent:.1f}%", 
                      delta=f"{diff_to_avg_pr_percent:+.1f}%", 
                      delta_color="normal" if diff_to_avg_pr_percent >= 0 else "inverse", 
                      help="Percentage difference of selected athlete's Percentrank from the overall average")

else:
    st.write("Please upload the sprint_data.csv file to view the analysis.")
    st.write("Unable to load data from either Google Sheets or CSV. Please ensure:")
    st.write("- Google Sheet is shared with 'Anyone with the link' can view, or")
    st.write("- sprint_data.csv file exists in the same directory as this script")
