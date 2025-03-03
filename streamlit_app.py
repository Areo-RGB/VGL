import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page title and description
st.set_page_config(page_title="Sprint Analysis", layout="wide")
st.title("Sprint Performance Dashboard")
st.write("Visualization of athlete performance based on Average Percentrank and individual tests")

# Function to load data from CSV
def load_data():
    csv_path = "sprint_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    else:
        st.error("sprint_data.csv not found. Please upload the file.")
        return None

# Load data
df = load_data()

if df is not None:
    # Display raw data in expandable section
    with st.expander("View Raw Data"):
        st.dataframe(df)

    # Create a dataframe with unique athletes and their average percentrank
    df_athletes = df.groupby('Name')['Average_Percentrank'].first().reset_index()
    
    # Sort by Average_Percentrank in descending order (highest rank at top)
    df_athletes_sorted = df_athletes.sort_values('Average_Percentrank', ascending=False)

    # Create horizontal bar chart for Average Percentrank
    st.subheader("Athlete Performance by Average Percentrank")

    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    bars = ax.barh(df_athletes_sorted['Name'], 
                   df_athletes_sorted['Average_Percentrank'], 
                   color='skyblue')
    
    ax.set_xlabel('Average Percentrank')
    ax.set_title('Athlete Performance Ranking')
    ax.invert_yaxis()  # Highest rank at the top

    # Add percentrank labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}',
                va='center')

    # Highlight the top performer
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
        top_athlete = df_athletes_sorted.loc[
            df_athletes_sorted['Average_Percentrank'] == top_rank, 'Name'
        ].values[0]
        st.metric("Top Rank", f"{top_rank:.1f}", f"by {top_athlete}")

    with col2:
        avg_rank = df_athletes_sorted['Average_Percentrank'].mean()
        st.metric("Average Rank", f"{avg_rank:.1f}")

    with col3:
        bottom_rank = df_athletes_sorted['Average_Percentrank'].min()
        bottom_athlete = df_athletes_sorted.loc[
            df_athletes_sorted['Average_Percentrank'] == bottom_rank, 'Name'
        ].values[0]
        st.metric("Bottom Rank", f"{bottom_rank:.1f}", f"by {bottom_athlete}")

    # Get unique test types
    test_types = df['Test'].unique()

    # Create charts for each test type
    st.subheader("Individual Test Performance")
    
    for test in test_types:
        # Filter data for current test
        df_test = df[df['Test'] == test].copy()
        df_test['Result'] = pd.to_numeric(df_test['Result'], errors='coerce')
        df_test = df_test.dropna(subset=['Result'])
        
        # Sort by Result (ascending, as lower times are better)
        df_test_sorted = df_test.sort_values('Result', ascending=True)

        if not df_test_sorted.empty:
            st.markdown(f"### {test} Results")
            
            fig_test, ax_test = plt.figure(figsize=(10, 6)), plt.axes()
            bars_test = ax_test.barh(df_test_sorted['Name'], 
                                  df_test_sorted['Result'], 
                                  color='lightgreen')
            
            ax_test.set_xlabel('Time (seconds)')
            ax_test.set_title(f'{test} Performance')
            ax_test.invert_yaxis()  # Fastest time at the top

            # Add time labels to the end of each bar
            for bar in bars_test:
                width = bar.get_width()
                ax_test.text(width + 0.02, 
                            bar.get_y() + bar.get_height()/2,
                            f'{width:.2f}s',
                            va='center')

            # Highlight the best performer
            if not df_test_sorted.empty:
                best_time = df_test_sorted['Result'].min()
                best_athlete = df_test_sorted.loc[
                    df_test_sorted['Result'] == best_time, 'Name'
                ].values[0]
                
                athlete_names = df_test_sorted['Name'].tolist()
                try:
                    bar_idx = athlete_names.index(best_athlete)
                    bars_test[bar_idx].set_color('gold')
                except ValueError:
                    st.warning(f"Could not highlight best performer for {test}")

            st.pyplot(fig_test)

            # Display statistics for this test
            col1, col2, col3 = st.columns(3)

            with col1:
                best_time = df_test_sorted['Result'].min()
                best_athlete = df_test_sorted.loc[
                    df_test_sorted['Result'] == best_time, 'Name'
                ].values[0]
                st.metric("Best Time", f"{best_time:.2f}s", f"by {best_athlete}")

            with col2:
                avg_time = df_test_sorted['Result'].mean()
                st.metric("Average Time", f"{avg_time:.2f}s")

            with col3:
                worst_time = df_test_sorted['Result'].max()
                worst_athlete = df_test_sorted.loc[
                    df_test_sorted['Result'] == worst_time, 'Name'
                ].values[0]
                st.metric("Worst Time", f"{worst_time:.2f}s", f"by {worst_athlete}")

    # Add sidebar filters
    st.sidebar.header("Filters")
    rank_threshold = st.sidebar.slider(
        "Filter by minimum Average Percentrank",
        min_value=float(df_athletes_sorted['Average_Percentrank'].min()),
        max_value=float(df_athletes_sorted['Average_Percentrank'].max()),
        value=float(df_athletes_sorted['Average_Percentrank'].max())
    )

    filtered_df = df_athletes_sorted[
        df_athletes_sorted['Average_Percentrank'] >= rank_threshold
    ]

    if len(filtered_df) < len(df_athletes_sorted):
        st.subheader(f"Athletes with Average Percentrank above {rank_threshold:.1f}")
        st.dataframe(filtered_df)

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