# Path: projects-2024-econtechvets/dataproject/dataproject.py
# Description: This file contains the functions used in the data project.


#Function to calculate the share of players in each region/county. 
def calculate_geographic_shares(idrakt):

    idrakt_filtered = idrakt[idrakt['county'] != 'All Denmark']

    # Group by geography and sum the player counts
    total_players_by_region = idrakt_filtered.groupby('county')['value'].sum().reset_index()

    # Calculate the share for each region
    total_players = total_players_by_region['value'].sum()
    total_players_by_region['share'] = total_players_by_region['value'] / total_players

    return total_players_by_region
