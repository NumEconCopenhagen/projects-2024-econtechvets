# Path: projects-2024-econtechvets/dataproject/dataproject.py
# Description: This file contains the functions used in the data project.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    # Plot the indexed population with a stippled (dashed) line
    population_index.plot(ax=ax, linestyle='--', color='k', label='Population')

    # Set the title and labels
    ax.set_title('Indexed Development of Players and Population in All Denmark (Base Year: 2014)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Index (2014 = 100)')

    # Adjust the legend
    ax.legend(title='Category')

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

def process_data(df):
    filtered_df = df[df['sex'] == 'Sex, total']
    grouped_df = filtered_df.groupby(['region', 'year'])['players'].sum().reset_index()
    
    # Normalize the data to index it to the year 2014 for each region
    for region in grouped_df['region'].unique():
        base_value = grouped_df[(grouped_df['region'] == region) & (grouped_df['year'] == 2014)]['players'].values[0]
        grouped_df.loc[grouped_df['region'] == region, 'indexed_players'] = grouped_df[grouped_df['region'] == region]['players'] / base_value * 100
    
    return grouped_df

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

def process_data_for_shares(df):
    # Calculate the share of players for each region's population
    df['player_share'] = df['players'] / df['population'] * 100
    return df

#Making a graph of shared data
def calculate_player_share(df):
    # Calculate the share of players for each row
    df['player_share'] = df['players'] / df['population'] * 100
    return df

def plot_share_data(df):
    # Create a figure using make_subplots to enable multiple lines (one for each region)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add a line for each region
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        fig.add_trace(
            go.Scatter(
                x=region_data['year'], 
                y=region_data['player_share'], 
                name=region,
                mode='lines+markers',
            )
        )

    # Update the layout
    fig.update_layout(
        title='Share of Players in Population per Region',
        xaxis_title='Year',
        yaxis_title='Share of Players (%)',
        legend_title='Region',
    )

    fig.show()


#code for making a graph of share of players in population across regions
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_share_data(df):
    # Creating subplots: one row, three columns
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

    # Remove the x-axis title
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_xaxes(title_text="", row=1, col=3)

    # Update yaxis properties
    fig.update_yaxes(title_text="Share of Players (%)", row=1, col=1)
    fig.update_yaxes(title_text="Share of Players (%)", row=1, col=2)
    fig.update_yaxes(title_text="Share of Players (%)", row=1, col=3)

    # Update layout, position the legend below the graph, and disable interactivity
    fig.update_layout(title_text='Share of Players in Population by Region and Sex',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, traceorder="normal", tracegroupgap=0, itemclick=False, itemdoubleclick=False))

    fig.show()