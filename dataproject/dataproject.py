# Path: projects-2024-econtechvets/dataproject/dataproject.py
# Description: This file contains the functions used in the data project.
import pandas as pd
import matplotlib.pyplot as plt

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

#Function to calculate the share of players in each region/county. 
def calculate_geographic_shares(df_merged):

    idrakt_filtered = idrakt[idrakt['county'] != 'All Denmark']

    # Group by geography and sum the player counts
    total_players_by_region = idrakt_filtered.groupby('county')['value'].sum().reset_index()

    # Calculate the share for each region
    total_players = total_players_by_region['value'].sum()
    total_players_by_region['share'] = total_players_by_region['value'] / total_players

    return total_players_by_region
