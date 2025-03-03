import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import os
import requests
import re
import base64
import numpy as np

# Set page configuration for wide layout
st.set_page_config(page_title="Sprint Performance Dashboard", layout="wide")
st.title("Sprint Performance Dashboard")
st.write("Compare an athlete's performance against the rest")

# Decrease sidebar width with percentage-based CSS and reduce individual KPI size
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 20% !important;
        min-width: 150px;
    }
    /* Reduce size of individual test KPIs */
    .small-kpi .stMetric {
        font-size: 14px !important;
    }
    .small-kpi .stMetric label {
        font-size: 14px !important;
    }
    .small-kpi .stMetric span {
        font-size: 14px !important;
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

    # Define test scoring directions
    higher_is_better_tests = ['Seilspringen', 'Standweitsprung']  # Higher values are better
    lower_is_better_tests = [test for test in df['Test'].unique() if test not in higher_is_better_tests]  # Lower values are better

    # Define categories
    speed_tests = ['10m Sprint', '20m Sprint', '50m Sprint', 'Gewandtheit']  # Speed category, lower is better
    technique_tests = ['Dribbling', 'Ballkontrolle']  # Technique category, lower is better

    # Add filter for athlete name at the top
    athlete_names = df['Name'].unique().tolist()
    selected_athlete = st.selectbox("Select an Athlete", options=athlete_names)

    # Display athlete image from Sheet2 with custom size
    image_url = None
    if df_images is not None and 'Name' in df_images.columns and 'Image' in df_images.columns:
        matching_row = df_images[df_images['Name'] == selected_athlete]
        if not matching_row.empty:
            image_url = matching_row['Image'].iloc[0] if not pd.isna(matching_row['Image'].iloc[0]) else None

    if image_url:
        img = load_image_from_url(image_url)
        if img:
            base64_img = base64.b64encode(img).decode('utf-8')
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{base64_img}" width="300" height="450" alt="{selected_athlete}">
                    <p>{selected_athlete}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(f'<span style="color:red; font-size:24px;">✗</span> No valid image for {selected_athlete}', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="color:red; font-size:24px;">✗</span> No image URL for {selected_athlete}', unsafe_allow_html=True)

    # Filter data for the selected athlete
    df_selected = df[df['Name'] == selected_athlete].copy()
    df_others = df[df['Name'] != selected_athlete].copy()

    if not df_selected.empty:
        # Display KPIs
        st.subheader("Key Performance Indicators")

        # Calculate Speed Category Performance (average time, lower is better)
        speed_df = df[df['Test'].isin(speed_tests)]
        speed_df_selected = df_selected[df_selected['Test'].isin(speed_tests)]
        overall_speed_avg = speed_df['Result'].mean() if not speed_df.empty else float('inf')
        selected_speed_avg = speed_df_selected['Result'].mean() if not speed_df_selected.empty else float('inf')
        diff_percent_speed = ((selected_speed_avg - overall_speed_avg) / overall_speed_avg * 100) if overall_speed_avg != 0 and overall_speed_avg != float('inf') else 0

        # Display Speed Performance KPI
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Speed Performance (All)", 
                      f"{overall_speed_avg:.2f}s" if overall_speed_avg != float('inf') else "N/A", 
                      label_visibility="visible", 
                      help="Average time for Speed category (10m, 20m, 50m Sprints, Gewandtheit) across all athletes")
        with col2:
            st.metric(f"Speed Performance ({selected_athlete})", 
                      f"{selected_speed_avg:.2f}s" if selected_speed_avg != float('inf') else "N/A", 
                      delta=f"{diff_percent_speed:+.1f}%" if selected_speed_avg != float('inf') else "N/A", 
                      delta_color="normal" if diff_percent_speed >= 0 else "inverse",  # Green for negative (faster), red for positive (slower)
                      label_visibility="visible", 
                      help="Selected athlete's average time for Speed category vs. overall")

        # Calculate Technique Category Performance (average time, lower is better)
        technique_df = df[df['Test'].isin(technique_tests)]
        technique_df_selected = df_selected[df_selected['Test'].isin(technique_tests)]
        overall_technique_avg = technique_df['Result'].mean() if not technique_df.empty else float('inf')
        selected_technique_avg = technique_df_selected['Result'].mean() if not technique_df_selected.empty else float('inf')
        diff_percent_technique = ((selected_technique_avg - overall_technique_avg) / overall_technique_avg * 100) if overall_technique_avg != 0 and overall_technique_avg != float('inf') else 0

        # Display Technique Performance KPI
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Technique Performance (All)", 
                      f"{overall_technique_avg:.2f}s" if overall_technique_avg != float('inf') else "N/A", 
                      label_visibility="visible", 
                      help="Average time for Technique category (Dribbling, Ballkontrolle) across all athletes")
        with col4:
            st.metric(f"Technique Performance ({selected_athlete})", 
                      f"{selected_technique_avg:.2f}s" if selected_technique_avg != float('inf') else "N/A", 
                      delta=f"{diff_percent_technique:+.1f}%" if selected_technique_avg != float('inf') else "N/A", 
                      delta_color="inverse" if diff_percent_technique >= 0 else "normal",  # Red for positive (slower), green for negative (faster)
                      label_visibility="visible", 
                      help="Selected athlete's average time for Technique category vs. overall")

        # Add KPIs for each test's best results in strict 2-column structure with reduced size
        tests = df['Test'].unique()
        for test in tests:
            # Calculate best results
            if test in higher_is_better_tests:
                overall_best = df[df['Test'] == test]['Result'].max() if not df[df['Test'] == test].empty else float('-inf')
                selected_best = df_selected[df_selected['Test'] == test]['Result'].max() if not df_selected[df_selected['Test'] == test].empty else float('-inf')
                diff_percent = ((selected_best - overall_best) / overall_best * 100) if overall_best != 0 and overall_best != float('-inf') else 0
                unit = ""
            else:
                overall_best = df[df['Test'] == test]['Result'].min() if not df[df['Test'] == test].empty else float('inf')
                selected_best = df_selected[df_selected['Test'] == test]['Result'].min() if not df_selected[df_selected['Test'] == test].empty else float('inf')
                diff_percent = ((selected_best - overall_best) / overall_best * 100) if overall_best != 0 and overall_best != float('inf') else 0
                unit = "s"

            # Only display if selected athlete has data for the test
            if selected_best not in [float('inf'), float('-inf')]:
                with st.container():
                    st.markdown('<div class="small-kpi">', unsafe_allow_html=True)
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric(f"Best {test} (All)", 
                                  f"{overall_best:.2f}{unit}", 
                                  label_visibility="visible", 
                                  help=f"Best result for {test} across all athletes")
                    with cols[1]:
                        st.metric(f"Best {test} ({selected_athlete})", 
                                  f"{selected_best:.2f}{unit}", 
                                  delta=f"{diff_percent:+.1f}%", 
                                  delta_color="inverse" if diff_percent >= 0 and test not in higher_is_better_tests else "normal", 
                                  label_visibility="visible", 
                                  help=f"Selected athlete's best result for {test} vs. overall best")
                    st.markdown('</div>', unsafe_allow_html=True)

        # Radar chart: Average result per test comparison
        st.subheader(f"Average Test Performance: {selected_athlete} vs. All")
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor='none')

        # Calculate averages per test
        all_avg = df.groupby('Test')['Result'].mean().reset_index()
        selected_avg = df_selected.groupby('Test')['Result'].mean().reset_index()
        comparison_df = pd.merge(all_avg, selected_avg, on='Test', how='outer', suffixes=('_all', '_selected')).fillna(0)

        # Prepare data for radar chart
        categories = comparison_df['Test'].tolist()
        num_vars = len(categories)
        values_all = comparison_df['Result_all'].tolist()
        values_selected = comparison_df['Result_selected'].tolist()

        # Repeat the first value to close the circle
        values_all += values_all[:1]
        values_selected += values_selected[:1]

        # Compute angle for each category
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Close the plot

        # Plot data
        ax.plot(angles, values_all, linewidth=2, linestyle='solid', label='All Avg', color='#FF6B6B')
        ax.fill(angles, values_all, '#FF6B6B', alpha=0.2)
        ax.plot(angles, values_selected, linewidth=2, linestyle='solid', label=f'{selected_athlete} Avg', color='#00CED1')
        ax.fill(angles, values_selected, '#00CED1', alpha=0.2)

        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='white', fontsize=10, path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        ax.set_yticklabels([])  # Hide radial labels for cleaner look
        ax.set_title(f'{selected_athlete} vs. Average Performance', color='white', fontsize=14, weight='bold', 
                     path_effects=[pe.withStroke(linewidth=3, foreground='black')], pad=20)
        ax.set_facecolor((0, 0, 0, 0.05))
        ax.spines['polar'].set_color('white')
        ax.spines['polar'].set_alpha(0.5)
        
        # Customize legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=10, labelcolor='white', 
                  edgecolor='black', facecolor=(0, 0, 0, 0.1), framealpha=0.8)

        plt.tight_layout()
        st.pyplot(fig)

else:
    st.write("Please upload the sprint_data.csv file to view the analysis.")
    st.write("Unable to load data from either Google Sheets or CSV. Please ensure:")
    st.write("- Google Sheet is shared with 'Anyone with the link' can view, or")
    st.write("- sprint_data.csv file exists in the same directory as this script")
