import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import os
import requests
import re

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

# Function to load data from Google Sheets (Sheet1 for main data)
def load_data_from_sheets():
    try:
        SHEET_ID = '1fjqPnrhIRGeYNxkRmWn4igb81SJ_gxfLCSVb3KZyju4'
        SHEET_NAME = 'Sheet1'
        url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.warning(f"Failed to load from Google Sheets (Sheet1): {str(e)}")
        st.warning("Falling back to local CSV file...")
        return None

# Function to load image data from Sheet2
def load_image_data_from_sheets():
    try:
        SHEET_ID = '1fjqPnrhIRGeYNxkRmWn4igb81SJ_gxfLCSVb3KZyju4'
        SHEET_NAME = 'Sheet2'
        url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
        df_images = pd.read_csv(url)
        return df_images
    except Exception as e:
        st.warning(f"Failed to load image data from Sheet2: {str(e)}")
        return None

# Function to load data from CSV (fallback for Sheet1)
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

# Function to convert Google Drive URL to direct download URL
def convert_google_drive_url(url):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url  # Return original URL if not a Google Drive link

# Function to load image from URL
def load_image_from_url(url):
    try:
        # Handle Google Drive URLs
        download_url = convert_google_drive_url(url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(download_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content  # Return raw bytes
    except Exception:
        return None

# Load main data (Sheet1 or CSV)
df = load_data()

# Load image data (Sheet2)
df_images = load_image_data_from_sheets()

if df is not None:
    # Prepare data
    df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
    df = df.dropna(subset=['Result'])

    # Add filter for athlete name at the top
    athlete_names = df['Name'].unique().tolist()
    selected_athlete = st.selectbox("Select an Athlete", options=athlete_names)

    # Display athlete image from Sheet2
    image_url = None
    if df_images is not None and 'Name' in df_images.columns and 'Image' in df_images.columns:
        # Find matching row in Sheet2
        matching_row = df_images[df_images['Name'] == selected_athlete]
        if not matching_row.empty:
            image_url = matching_row['Image'].iloc[0] if not pd.isna(matching_row['Image'].iloc[0]) else None

    if image_url:
        img = load_image_from_url(image_url)
        if img:
            st.image(img, width=100, caption=f"{selected_athlete}")
        else:
            st.markdown(f'<span style="color:red; font-size:24px;">✗</span> No valid image for {selected_athlete}', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="color:red; font-size:24px;">✗</span> No image URL for {selected_athlete}', unsafe_allow_html=True)

    # Filter data for the selected athlete
    df_selected = df[df['Name'] == selected_athlete].copy()
    df_others = df[df['Name'] != selected_athlete].copy()

    if not df_selected.empty:
        # Calculate PRabs: Average Percentrank per athlete across all tests
        df_prabs = df.groupby('Name')['Average_Percentrank'].mean().reset_index()
        df_prabs = df_prabs.rename(columns={'Average_Percentrank': 'PRabs'})

        # Sort by PRabs and assign ranks
        df_prabs_sorted = df_prabs.sort_values('PRabs', ascending=False).reset_index(drop=True)
        df_prabs_sorted['Rank'] = df_prabs_sorted.index + 1
        total_athletes = len(df_prabs_sorted)

        # Selected athlete's PRabs and rank
        selected_prabs = df_prabs[df_prabs['Name'] == selected_athlete]['PRabs'].iloc[0]
        selected_rank = df_prabs_sorted[df_prabs_sorted['Name'] == selected_athlete]['Rank'].iloc[0]
        rank_display = f"{selected_rank}/{total_athletes}"

        # New KPI: Best result based on highest PR value
        best_pr_row = df_selected.loc[df_selected['Average_Percentrank'].idxmax()]
        best_pr_result = best_pr_row['Result']
        best_pr_test = best_pr_row['Test']
        best_pr_value = best_pr_row['Average_Percentrank']

        # Display modified KPIs at the top
        st.subheader("Key Performance Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"PRabs ({selected_athlete})", f"{selected_prabs:.1f}", 
                      delta=rank_display, 
                      delta_color="off", 
                      label_visibility="visible", 
                      help="Selected athlete's absolute Percentrank and rank (e.g., 3/15 means 3rd highest out of 15)")
        with col2:
            st.metric(f"Best Result by PR ({selected_athlete})", f"{best_pr_result:.2f}s", 
                      delta=f"{best_pr_test} (PR: {best_pr_value:.1f})", 
                      delta_color="off", 
                      label_visibility="visible", 
                      help="Selected athlete's result with the highest Percentrank value across all tests")

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

        # Collapsible Athlete PRabs Rankings with all athletes
        with st.expander("Athlete PRabs Rankings", expanded=False):
            df_prabs_display = df_prabs_sorted[['Name', 'PRabs']]
            def highlight_selected(row):
                return ['background-color: #00CED1' if row['Name'] == selected_athlete else '' for _ in row]
            styled_df = df_prabs_display.style.apply(highlight_selected, axis=1).format({'PRabs': "{:.1f}"})
            st.dataframe(styled_df, use_container_width=True, height=300)

        # Modified chart: Average result per test comparison
        st.subheader(f"Average Test Performance: {selected_athlete} vs. All")
        fig = plt.figure(facecolor='none')
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], facecolor=(0, 0, 0, 0.05))

        # Calculate averages per test
        all_avg = df.groupby('Test')['Result'].mean().reset_index()
        selected_avg = df_selected.groupby('Test')['Result'].mean().reset_index()
        comparison_df = pd.merge(all_avg, selected_avg, on='Test', how='left', suffixes=('_all', '_selected'))
        comparison_df['selected_minus_all'] = comparison_df['Result_selected'] - comparison_df['Result_all']

        # Colors: Light green (#90EE90) if selected is lower (better), red (#FF0000) if worse
        colors = ['#90EE90' if x < 0 else '#FF0000' for x in comparison_df['selected_minus_all'].fillna(0)]

        # Plot bars for selected athlete’s average results
        bar_width = 0.5
        x = range(len(comparison_df['Test']))
        bars = ax.bar(x, comparison_df['Result_selected'], bar_width, color=colors, edgecolor='black', linewidth=1.2, label=f'{selected_athlete} Avg')

        # Plot horizontal line for all athletes’ average per test
        for i, (test, avg_all) in enumerate(zip(comparison_df['Test'], comparison_df['Result_all'])):
            ax.axhline(y=avg_all, xmin=i/len(x) + 0.05, xmax=(i+1)/len(x) - 0.05, color='white', linestyle='--', linewidth=1.5, label='All Avg' if i == 0 else "")

        # Enhance text with outlines
        ax.set_xlabel('Test', color='white', fontsize=12, weight='bold', path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        ax.set_ylabel('Average Result (seconds)', color='white', fontsize=12, weight='bold', path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        ax.set_title(f'{selected_athlete} vs. Average Performance', color='white', fontsize=14, weight='bold', 
                     path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Test'], rotation=45, ha='right', color='white', fontsize=10, 
                           path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        ax.tick_params(axis='y', colors='white', labelsize=10)

        # Add grid lines for better readability
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5, color='white')

        # Customize legend
        ax.legend(fontsize=10, labelcolor='white', edgecolor='black', facecolor=(0, 0, 0, 0.1), framealpha=0.8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

else:
    st.write("Please upload the sprint_data.csv file to view the analysis.")
    st.write("Unable to load data from either Google Sheets or CSV. Please ensure:")
    st.write("- Google Sheet is shared with 'Anyone with the link' can view, or")
    st.write("- sprint_data.csv file exists in the same directory as this script")
