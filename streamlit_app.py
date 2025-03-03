import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page title and description
st.set_page_config(page_title="Sprint Analysis", layout="wide")
st.title("Sprint Performance Dashboard")
st.write("Visualization of athlete performance based on Average Percentrank and individual tests")

# Function to load data from Google Sheets
def load_data_from_sheets():
    try:
        SHEET_ID = '1fjqPnrhIRGeYNxkRmWn4igb81SJ_gxfLCSVb3KZyju4'
        SHEET_NAME = 'Sheet1'
        url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
        df = pd.read_csv(url)
        st.success("Successfully loaded data from Google Sheets")
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
            st.success("Successfully loaded data from local CSV file")
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
    # Add a chart for difference from average per athlete per test
    st.subheader("Difference from Average Performance per Test")

    # Prepare data for difference from average
    df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
    df = df.dropna(subset=['Result'])

    test_averages = df.groupby('Test')['Result'].mean().to_dict()
    df['Diff_From_Avg'] = df.apply(lambda row: row['Result'] - test_averages[row['Test']], axis=1)
    df_pivot = df.pivot_table(index='Name', columns='Test', values='Diff_From_Avg', fill_value=None)

    fig_diff = plt.figure(figsize=(12, 8), facecolor='none')
    ax_diff = fig_diff.add_axes([0.1, 0.1, 0.8, 0.8], facecolor='none')
    df_pivot.plot(kind='bar', ax=ax_diff, width=0.8, colormap='tab20', edgecolor='black')

    ax_diff.set_ylabel('Difference from Test Average (seconds)', color='white', fontsize=12, weight='bold')
    ax_diff.set_title('Difference from Average Performance by Test', color='white', fontsize=14, weight='bold')
    ax_diff.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_diff.legend(title='Test Types', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, labelcolor='white')
    ax_diff.set_xlabel('Athlete', color='white', fontsize=12, weight='bold')
    ax_diff.tick_params(axis='x', colors='white', rotation=45)
    ax_diff.tick_params(axis='y', colors='white')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig_diff)

    # Display raw data in expandable section
    with st.expander("View Raw Data"):
        st.dataframe(df)

    # Create a dataframe with unique athletes and their average percentrank
    df_athletes = df.groupby('Name')['Average_Percentrank'].first().reset_index()
    df_athletes_sorted = df_athletes.sort_values('Average_Percentrank', ascending=False)

    # Create horizontal bar chart for Average Percentrank
    st.subheader("Athlete Performance by Average Percentrank")
    st.image("https://i.imgur.com/RpkL2e0.png", use_container_width=True)

    fig = plt.figure(figsize=(10, 6), facecolor='none')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], facecolor='none')
    bars = ax.barh(df_athletes_sorted['Name'], 
                   df_athletes_sorted['Average_Percentrank'], 
                   color='skyblue')
    ax.set_xlabel('Average Percentrank', color='white', fontsize=12, weight='bold')
    ax.set_title('Athlete Performance Ranking', color='white', fontsize=14, weight='bold')
    ax.invert_yaxis()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                va='center', color='white', fontsize=10, weight='bold')

    if not df_athletes_sorted.empty:
        top_performer_idx = df_athletes_sorted['Average_Percentrank'].idxmax()
        top_performer_name = df_athletes_sorted.loc[top_performer_idx, 'Name']
        athlete_names = df_athletes_sorted['Name'].tolist()
        try:
            bar_idx = athlete_names.index(top_performer_name)
            bars[bar_idx].set_color('gold')
        except ValueError:
            st.warning("Could not highlight top performer")

    st.pyplot(fig)

    # Display statistics for Average Percentrank
    st.subheader("Performance Statistics (Average Percentrank)")
    col1, col2, col3 = st.columns(3)

    with col1:
        top_rank = df_athletes_sorted['Average_Percentrank'].max()
        top_athlete = df_athletes_sorted.loc[df_athletes_sorted['Average_Percentrank'] == top_rank, 'Name'].values[0]
        st.metric("Top Rank", f"{top_rank:.1f}", f"by {top_athlete}")

    with col2:
        avg_rank = df_athletes_sorted['Average_Percentrank'].mean()
        st.metric("Average Rank", f"{avg_rank:.1f}")

    with col3:
        bottom_rank = df_athletes_sorted['Average_Percentrank'].min()
        bottom_athlete = df_athletes_sorted.loc[df_athletes_sorted['Average_Percentrank'] == bottom_rank, 'Name'].values[0]
        st.metric("Bottom Rank", f"{bottom_rank:.1f}", f"by {bottom_athlete}")

    # Get unique test types
    test_types = df['Test'].unique()

    # Create compact charts for each test type
    st.subheader("Individual Test Performance")
    
    for test in test_types:
        df_test = df[df['Test'] == test].copy()
        df_test['Result'] = pd.to_numeric(df_test['Result'], errors='coerce')
        df_test = df_test.dropna(subset=['Result'])
        df_test_sorted = df_test.sort_values('Result', ascending=True)

        if not df_test_sorted.empty:
            st.markdown(f"### {test} Results")
            fig_test = plt.figure(figsize=(8, 4), facecolor='none')  # Smaller size for compactness
            ax_test = fig_test.add_axes([0.1, 0.1, 0.8, 0.8], facecolor='none')
            bars_test = ax_test.barh(df_test_sorted['Name'], 
                                     df_test_sorted['Result'], 
                                     color='lightgreen')
            ax_test.set_xlabel('Time (seconds)', color='white', fontsize=10, weight='bold')  # Smaller font
            ax_test.set_title(f'{test} Performance', color='white', fontsize=12, weight='bold')  # Smaller font
            ax_test.invert_yaxis()
            ax_test.tick_params(axis='x', colors='white', labelsize=8)  # Smaller tick labels
            ax_test.tick_params(axis='y', colors='white', labelsize=8)

            for bar in bars_test:
                width = bar.get_width()
                ax_test.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.2f}s', 
                             va='center', color='white', fontsize=8, weight='bold')  # Smaller font

            if not df_test_sorted.empty:
                best_time = df_test_sorted['Result'].min()
                best_athlete = df_test_sorted.loc[df_test_sorted['Result'] == best_time, 'Name'].values[0]
                athlete_names = df_test_sorted['Name'].tolist()
                try:
                    bar_idx = athlete_names.index(best_athlete)
                    bars_test[bar_idx].set_color('gold')
                except ValueError:
                    st.warning(f"Could not highlight best performer for {test}")

            plt.tight_layout(pad=0.5)  # Tighten layout with less padding
            st.pyplot(fig_test)

            col1, col2, col3 = st.columns(3)
            with col1:
                best_time = df_test_sorted['Result'].min()
                best_athlete = df_test_sorted.loc[df_test_sorted['Result'] == best_time, 'Name'].values[0]
                st.metric("Best Time", f"{best_time:.2f}s", f"by {best_athlete}")

            with col2:
                avg_time = df_test_sorted['Result'].mean()
                st.metric("Average Time", f"{avg_time:.2f}s")

            with col3:
                worst_time = df_test_sorted['Result'].max()
                worst_athlete = df_test_sorted.loc[df_test_sorted['Result'] == worst_time, 'Name'].values[0]
                st.metric("Worst Time", f"{worst_time:.2f}s", f"by {worst_athlete}")

    # Add sidebar filters
    st.sidebar.header("Filters")
    
    # Existing Percentrank filter
    rank_threshold = st.sidebar.slider(
        "Filter by minimum Average Percentrank",
        min_value=float(df_athletes_sorted['Average_Percentrank'].min()),
        max_value=float(df_athletes_sorted['Average_Percentrank'].max()),
        value=float(df_athletes_sorted['Average_Percentrank'].max())
    )

    # New interactive athlete filter
    athlete_names = df['Name'].unique().tolist()
    selected_athlete = st.sidebar.selectbox("Select an Athlete", options=athlete_names)

    # Filter data for the selected athlete
    df_selected_athlete = df[df['Name'] == selected_athlete].copy()

    # Create a chart for the selected athlete's test results (unchanged size)
    if not df_selected_athlete.empty:
        st.subheader(f"Test Results for {selected_athlete}")
        fig_athlete = plt.figure(figsize=(10, 6), facecolor='none')  # Keep original size
        ax_athlete = fig_athlete.add_axes([0.1, 0.1, 0.8, 0.8], facecolor='none')
        
        bars_athlete = ax_athlete.bar(df_selected_athlete['Test'], 
                                      df_selected_athlete['Result'], 
                                      color='teal', edgecolor='black')
        
        ax_athlete.set_xlabel('Test', color='white', fontsize=12, weight='bold')
        ax_athlete.set_ylabel('Result (seconds)', color='white', fontsize=12, weight='bold')
        ax_athlete.set_title(f'{selected_athlete} Test Performance', color='white', fontsize=14, weight='bold')
        ax_athlete.tick_params(axis='x', colors='white', rotation=45)
        ax_athlete.tick_params(axis='y', colors='white')
        
        for bar in bars_athlete:
            height = bar.get_height()
            ax_athlete.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}s', 
                            ha='center', color='white', fontsize=10, weight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_athlete)

    # Existing filtered dataframe based on Percentrank
    filtered_df = df_athletes_sorted[df_athletes_sorted['Average_Percentrank'] >= rank_threshold]

    if len(filtered_df) < len(df_athletes_sorted):
        st.subheader(f"Athletes with Average Percentrank above {rank_threshold:.1f}")
        st.dataframe(filtered_df)

    # Add download button for the data
    st.sidebar.download_button(
        label="Download Sprint Data",
        data=csv,
        file_name="sprint_data.csv",
        mime="text/csv"
    )

else:
    st.write("Please upload the sprint_data.csv file to view the analysis.")
    st.write("Unable to load data from either Google Sheets or CSV. Please ensure:")
    st.write("- Google Sheet is shared with 'Anyone with the link' can view, or")
    st.write("- sprint_data.csv file exists in the same directory as this script")
