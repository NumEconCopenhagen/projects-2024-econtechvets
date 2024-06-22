# Path: projects-2024-econtechvets/dataproject/dataproject.py
# Description: This file contains the functions used in the data project.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
from plotly.subplots import make_subplots


"""
The function total_players_dk saves a figure of a stacked bar chart 
showing the total number of football players in Denmark by sex.
"""
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


"""
The function index_players_dk saves the figure indexed_players_chart.png in a line chart
showing the indexed values of football players across sex in Denmark from 2014-2022.
It also contains a dashed line of the indexed population.
"""
def index_players_dk(df, base_year=2014, filename='indexed_players_chart.png'):
    # Ensure 'year' is of type int
    df['year'] = df['year'].astype(int)

    # Filter data for 'All Denmark'
    denmark_data = df[df['region'] == 'All Denmark']

    # Pivot the player data
    pivot_data = denmark_data.pivot(index='year', columns='sex', values='players')

    # Calculate the index (base_year = 100) for players
    base_values = pivot_data.loc[base_year]
    pivot_data = pivot_data / base_values * 100

    # Extract and index the population data for 'Sex, total'
    population_data = denmark_data[denmark_data['sex'] == 'Sex, total'].set_index('year')['population']
    population_index = population_data / population_data.loc[base_year] * 100

    # Create the plot for indexed values
    ax = pivot_data[['Men', 'Women', 'Sex, total']].plot(kind='line', figsize=(10, 7))

    # Plot the indexed population with a dashed line
    population_index.plot(ax=ax, linestyle='--', color='k', label='Population')

    # Set the title and labels
    ax.set_title('Indexed Development of Players and Population in All Denmark')
    ax.set_xlabel('Year')
    ax.set_ylabel('Index (2014 = 100)')

    # Adjust the legend
    ax.legend(title='Category')

    plt.tight_layout()

    # Save the figure
    plt.savefig(filename)
    plt.close()


"""
The function plot_growth_contributions_all_denmark saves the figure growth_contribution_all_denmark in a stacked bar chart 
showing the year-over-year change in the number of football players by sex in Denmark.
"""
def plot_growth_contributions_all_denmark(df, filename='growth_contributions_all_denmark.png'):
    # Filter for 'All Denmark' and make a copy to avoid SettingWithCopyWarning
    df_dk = df[df['region'] == 'All Denmark'].copy()

    # Calculate year-over-year change for each category
    df_dk.loc[:, 'year_over_year_change'] = df_dk.groupby('sex', observed=False)['players'].diff()

    # Calculate the contribution for each category
    total_values_previous_year = df_dk[df_dk['sex'] == 'Sex, total'][['year', 'players']].rename(columns={'players': 'total_previous_year'})
    total_values_previous_year['year'] += 1
    df_dk = pd.merge(df_dk, total_values_previous_year, on='year', how='left')
    df_dk.loc[:, 'contribution'] = df_dk.apply(lambda x: x['year_over_year_change'] / x['total_previous_year'] if x['sex'] != 'Sex, total' else None, axis=1)

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


"""
The function process_data takes a DataFrame df as input and returns a new DataFrame 
with the indexed number of players per region across years.
It is used as a funciton with our merged_df dataset in the notebook to calculate the indexed number of players across regions.
"""
def process_data(df):
    filtered_df = df[df['sex'] == 'Sex, total']
    grouped_df = filtered_df.groupby(['region', 'year'])['players'].sum().reset_index()
    
    # Normalize the data to index it to the year 2014 for each region
    for region in grouped_df['region'].unique():
        base_value = grouped_df[(grouped_df['region'] == region) & (grouped_df['year'] == 2014)]['players'].values[0]
        grouped_df.loc[grouped_df['region'] == region, 'indexed_players'] = grouped_df[grouped_df['region'] == region]['players'] / base_value * 100
    
    return grouped_df


"""
The function plot_data creates a line chart showing the indexed number of players per region across years.
It is used as a function in the notebook to plot the indexed number of players across regions.
"""
def plot_data(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        fig.add_trace(
            go.Scatter(
                x=region_data['year'], 
                y=region_data['indexed_players'], 
                name=region,
                mode='lines+markers',
            )
        )

    fig.update_layout(
        title='Indexed Number of Players per Region Across Years',
        xaxis_title='Year',
        yaxis_title='Indexed Number of Players (2014 = 100)',
        legend_title='Region',
    )

    fig.show()


"""
code for making a graph of share of players in population across regions
First we need to calculate the share of players in the population and the we create the plots.
"""
def plot_share_data(df):
    #calculating the shares
    df['share'] = (df['players'] / df['population']) * 100

    # Create subplots: one row, three columns
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Men", "Women", "Sex, total"))

    # Define a color map for each unique region
    colors = {region: f'rgb({hash(region) % 256}, {hash(region * 2) % 256}, {hash(region * 3) % 256})' for region in df['region'].unique()}

    # Filter and plot data for Men
    men_df = df[df['sex'] == 'Men']
    for region, group in men_df.groupby('region'):
        fig.add_trace(go.Scatter(x=group['year'], y=group['share'], mode='lines+markers', name=region, marker=dict(color=colors[region])), row=1, col=1)

    # Filter and plot data for Women
    women_df = df[df['sex'] == 'Women']
    for region, group in women_df.groupby('region'):
        fig.add_trace(go.Scatter(x=group['year'], y=group['share'], mode='lines+markers', name=region, showlegend=False, marker=dict(color=colors[region])), row=1, col=2)

    # Filter and plot data for Sex, total
    total_df = df[df['sex'] == 'Sex, total']
    for region, group in total_df.groupby('region'):
        fig.add_trace(go.Scatter(x=group['year'], y=group['share'], mode='lines+markers', name=region, showlegend=False, marker=dict(color=colors[region])), row=1, col=3)

    # Update layout
    fig.update_layout(
        title='Share of Players in Population by Region and Sex',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            itemclick="toggleothers"
        )
    )
    fig.show()