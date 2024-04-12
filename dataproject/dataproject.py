# Path: projects-2024-econtechvets/dataproject/dataproject.py
# Description: This file contains the functions used in the data project.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Making the stacked bar chart for total players and calling it in the notebook when wanting to display it
def total_players_dk(df, filename='stacked_bar_chart.png'):
    denmark_data = df[(df['region'] == 'All Denmark') & (df['sex'] != 'Sex, total')]
    pivot_data = denmark_data.pivot(index='year', columns='sex', values='players')
    ax = pivot_data.plot(kind='bar', stacked=True, figsize=(10, 7))
    plt.title('Total Number of Players by Sex in All Denmark')
    plt.xlabel('Year')
    plt.ylabel('Number of Players')
    ax.legend(['Men', 'Women'], title='')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename)
    plt.close()

# Making the graph for indexed football players and calling it in the notebook when wanting to display it
def index_players_dk(df, base_year=2014, filename='indexed_players_chart.png'):
    # Ensure 'year' is of type int
    df['year'] = df['year'].astype(int)

    # Filter data for 'All Denmark'
    denmark_data = df[df['region'] == 'All Denmark']
    
    # Pivot the data
    pivot_data = denmark_data.pivot(index='year', columns='sex', values='players')
    
    # Check if the base year exists in the index
    if base_year in pivot_data.index:
        # Calculate the index (base_year = 100)
        for column in pivot_data.columns:
            pivot_data[column] = pivot_data[column] / pivot_data.at[base_year, column] * 100
    else:
        print(f"Base year {base_year} not found in data.")
        return  # Exit the function if base year is not found

    # Create the plot
    ax = pivot_data.plot(kind='line', figsize=(10, 7))
    
    # Set the title and labels
    plt.title(f'Indexed Development of Players by Sex in All Denmark')
    plt.xlabel('Year')
    plt.ylabel('Index (Base Year = 100)')
    
    # Adjust the legend labels
    ax.legend(title='')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename)
    plt.close()

# Making the growth contribution graph and calling it in the notebook when wanting to display it
def plot_growth_contributions_all_denmark(df, filename='growth_contributions_all_denmark.png'):
    # Filter for 'All Denmark'
    df_dk = df[df['region'] == 'All Denmark']

    # Calculate year-over-year change for each category
    df_dk['year_over_year_change'] = df_dk.groupby('sex')['players'].diff()

    # Calculate the contribution for each category
    total_values_previous_year = df_dk[df_dk['sex'] == 'Sex, total'][['year', 'players']].rename(columns={'players': 'total_previous_year'})
    total_values_previous_year['year'] += 1
    df_dk = pd.merge(df_dk, total_values_previous_year, on='year', how='left')
    df_dk['contribution'] = df_dk.apply(lambda x: x['year_over_year_change'] / x['total_previous_year'] if x['sex'] != 'Sex, total' else None, axis=1)

    # Pivot the data for plotting
    pivot_contribution = df_dk.pivot(index='year', columns='sex', values='contribution').fillna(0)

    # Visualize the contributions
    ax = pivot_contribution[['Men', 'Women']].plot(kind='bar', stacked=True, figsize=(10, 6))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # Format y-axis as percentage
    plt.title('Growth Contributions from Men and Women in All Denmark')
    plt.xlabel('Year')
    plt.ylabel('Growth contribution')
    plt.legend(['Men', 'Women'])
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename)
    plt.close()