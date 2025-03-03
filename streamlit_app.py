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
        # Calculate PRabs: Average Percentrank per athlete across all tests
        df_prabs = df.groupby('Name')['Average_Percentrank'].mean().reset_index()
        df_prabs = df_prabs.rename(columns={'Average_Percentrank': 'PRabs'})

        # Overall PRabs (mean of all athletes' PRabs)
        overall_prabs = df_prabs['PRabs'].mean()
        # Selected athlete's PRabs
        selected_prabs = df_prabs[df_prabs['Name'] == selected_athlete]['PRabs'].iloc[0]
        diff_to_avg_prabs_percent = ((selected_prabs - overall_prabs) / overall_prabs) * 100 if overall_prabs != 0 else 0

        # Display modified KPIs at the top
        st.subheader("Key Performance Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall PRabs", f"{overall_prabs:.1f}", 
                      label_visibility="visible", 
                      help="Average of all athletes' absolute Percentranks (PRabs = avg of test PRs per athlete)")
        with col2:
            st.metric(f"PRabs ({selected_athlete})", f"{selected_prabs:.1f}", 
                      delta=f"{diff_to_avg_prabs_percent:+.1f}%", 
                      delta_color="normal" if diff_to_avg_prabs_percent >= 0 else "inverse", 
                      help="Selected athlete's absolute Percentrank (avg of all test PRs)")

        # New KPI: Best Result (lowest time) comparison with test filter
        st.subheader("Best Performance Comparison (Lower is Better)")
        test_names = df['Test'].unique().tolist()
        selected_test = st.selectbox("Select a Test", options=test_names, key="test_filter")
        
        # Filter data for the selected test
        df_test_all = df[df['Test'] == selected_test]
        df_test_selected = df_selected[df_selected['Test'] == selected_test]
        
        # Calculate best result for the selected test
        overall_best_result = df_test_all['Result'].min() if not df_test_all.empty else float('inf')
        selected_best_result = df_test_selected['Result'].min() if not df_test_selected.empty else float('inf')
        diff_to_best_percent = ((selected_best_result - overall_best_result) / overall_best_result) * 100 if overall_best_result != 0 else 0

        # Display best result KPI for the selected test
        col3, col4 = st.columns(2)
        with col3:
            st.metric(f"Overall Best Result ({selected_test})", 
                      f"{overall_best_result:.2f}s" if overall_best_result != float('inf') else "N/A", 
                      label_visibility="visible", 
                      help=f"Lowest sprint time for {selected_test} across all athletes")
        with col4:
            st.metric(f"Best Result ({selected_athlete}, {selected_test})", 
                      f"{selected_best_result:.2f}s" if selected_best_result != float('inf') else "N/A", 
                      delta=f"{diff_to_best_percent:+.1f}%" if selected_best_result != float('inf') else "N/A", 
                      delta_color="inverse" if diff_to_best_percent >= 0 else "normal", 
                      help=f"Selected athlete's best time for {selected_test} compared to overall best")

        # Small sorted table with top 3 + selected athlete
        st.subheader("Athlete PRabs Rankings")
        df_prabs_sorted = df_prabs.sort_values('PRabs', ascending=False).reset_index(drop=True)
        df_prabs_sorted['Rank'] = df_prabs_sorted.index + 1
        
        # Get top 3 and selected athlete
        top_3 = df_prabs_sorted.head(3)
        selected_row = df_prabs_sorted[df_prabs_sorted['Name'] == selected_athlete]
        if selected_athlete not in top_3['Name'].values:
            df_prabs_filtered = pd.concat([top_3, selected_row]).reset_index(drop=True)
            df_prabs_filtered['Rank'] = df_prabs_filtered.index + 1  # Reassign ranks
        else:
            df_prabs_filtered = top_3

        # Highlight selected athlete in the table
        def highlight_selected(row):
            return ['background-color: #00CED1' if row['Name'] == selected_athlete else '' for _ in row]

        styled_df = df_prabs_filtered.style.apply(highlight_selected, axis=1).format({'PRabs': "{:.1f}"})
        st.dataframe(styled_df, use_container_width=True, height=150)

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

else:
    st.write("Please upload the sprint_data.csv file to view the analysis.")
    st.write("Unable to load data from either Google Sheets or CSV. Please ensure:")
    st.write("- Google Sheet is shared with 'Anyone with the link' can view, or")
    st.write("- sprint_data.csv file exists in the same directory as this script")
