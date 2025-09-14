# process_temperature_data.py

import netCDF4 as nc
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from global_land_mask import globe
# Assuming aux_ncdf.py is in the same directory or accessible via PYTHONPATH
from aux_ncdf import preprocess_data, add_max_increasing_chain, add_max_high_temp_chain, add_large_jump, calculate_increasing_temp_features, calculate_high_temp_features, calculate_temp_jump_features
import argparse
import os

def main():
    """
    Processes temperature and precipitation data from NetCDF files,
    calculates various features, and saves the results to a CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Process temperature and precipitation data from NetCDF files."
    )
    parser.add_argument(
        '--file_name_temp',
        type=str,
        required=True,
        help="Path to the temperature NetCDF file (e.g., 'data/tasmax_CanESM5_historical_Arg.nc')"
    )
    parser.add_argument(
        '--file_name_pp',
        type=str,
        required=True,
        help="Path to the precipitation NetCDF file (e.g., 'data/pr_CanESM5_historical_Arg.nc')"
    )
    parser.add_argument(
        '--results_folder',
        type=str,
        default='results',
        help="Optional: Folder to save the results. Defaults to 'results'."
    )

    parser.add_argument(
        '--era_model',
        type=str,
        default='no',
        help="Is this ERA model?"
    )

    args = parser.parse_args()

    file_name_temp = args.file_name_temp
    file_name_pp = args.file_name_pp
    results_folder = args.results_folder
    era_model = args.era_model

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"Created results folder: {results_folder}")

    if era_model == 'no':
        print(f"Loading temperature data from: {file_name_temp}")
        dset_temp = xr.open_dataset(f'./{file_name_temp}')
        print(f"Loading precipitation data from: {file_name_pp}")
        dset_pp = xr.open_dataset(f'./{file_name_pp}')
    
        df_temp = dset_temp.to_dataframe().reset_index()
        df_pp = dset_pp.to_dataframe().reset_index()
    
        seconds_per_day = 86400
        df_raw = pd.merge(df_temp, df_pp, on=['lat', 'lon', 'time'])
    
        # Adjust longitude from 0-360 to -180-180 if necessary
        # This is often needed for global_land_mask or other geospatial operations
        df_raw['lon'] = df_raw['lon'].apply(lambda x: x - 360 if x > 180 else x)
    
        # Convert temperature from Kelvin to Celsius
        df_raw['t2m'] = df_raw['tasmax'] - 273.15
        # Convert precipitation flux to mm/day
        df_raw['tp'] = df_raw['pr'] * seconds_per_day
    
        # Convert time column to datetime objects and extract components
        df_raw['time'] = pd.to_datetime(df_raw.time.astype(str))
        df_raw['year'] = df_raw.time.dt.year
        df_raw['month'] = df_raw.time.dt.month
        df_raw['day'] = df_raw.time.dt.day
        df_raw['hour'] = df_raw.time.dt.hour
    
        # Create a unique station identifier
        df_raw['station'] = df_raw['lat'].astype(str) + '_' + df_raw['lon'].astype(str)
    
        # Select and rename columns for further processing
        df = df_raw[['year', 'month', 'day', 'station', 't2m', 'tp', 'lat', 'lon']].rename(columns={
            'lat': 'latitude',
            'lon': 'longitude',
            't2m': 'TMAX',
            'tp': 'PP'
        })
        
    elif era_model == 'yes':
        df = pd.read_csv('era/daily_era_data.csv')

    # Loop through different processing scenarios
    for season in ['winter', 'summer']:
        for pp_lower_bound in [1]:  # [1, 5]
            for grouping_strategy in [0, 1, 2, 3]:
                print(f'\n--- Processing Scenario ---')
                print(f'Season: {season}')
                print(f'PP lower bound: {pp_lower_bound}')
                print(f'Grouping strategy: {grouping_strategy}')

                extreme_value_strategy = 'local'

                # Preprocess data based on current scenario
                df_prep = preprocess_data(
                    df,
                    season=season,
                    pp_lower_bound=pp_lower_bound,
                    grouping_strategy=grouping_strategy,
                    extreme_value_strategy=extreme_value_strategy
                )

                # Add various chain and jump features
                df_prep = add_max_increasing_chain(df_prep, eps=0.5)
                df_prep = add_max_high_temp_chain(df_prep, perc=0.75)
                df_prep = add_large_jump(df_prep, perc=0.9, rain_delay=1)

                # Calculate extreme precipitation rates and values
                extreme_pp_rate = df_prep.groupby('station').PP_extreme.agg(['mean', 'count']).reset_index()
                extreme_pp_rate.columns = ['station', 'extreme_pp_rate', 'days_total']
                
                days_with_rain = df_prep.groupby('station').rain.sum().to_frame('days_with_pp').reset_index()
                extreme_pp_rate = pd.merge(extreme_pp_rate, days_with_rain, on='station', how='left')

                # Note: PP.min() here seems to imply the minimum PP value when PP_extreme is 1.
                # If it's meant to be the *maximum* extreme PP value, you might want .max()
                extreme_pp_value = df_prep[df_prep.PP_extreme == 1].groupby('station').PP.min().reset_index().rename(columns={'PP': 'extreme_pp_value'})
                extreme_pp_rate = pd.merge(extreme_pp_rate, extreme_pp_value, on='station', how='left')

                # Calculate temperature features
                increasing_chain_stats = calculate_increasing_temp_features(df_prep)
                high_t_chain_stats = calculate_high_temp_features(df_prep)
                freq_large_jump_pos, pp_extreme_large_jump_pos, freq_large_jump_neg, pp_extreme_large_jump_neg = calculate_temp_jump_features(df_prep)

                pp_extreme_large_jump_pos = pp_extreme_large_jump_pos.rename(columns={'PP_extreme': 'large_jump_pos_extreme_pp_rate'})
                pp_extreme_large_jump_neg = pp_extreme_large_jump_neg.rename(columns={'PP_extreme': 'large_jump_neg_extreme_pp_rate'})

                # Merge all calculated features into a single results DataFrame
                results = pd.merge(extreme_pp_rate, increasing_chain_stats, on='station', how='left')
                results = pd.merge(results, high_t_chain_stats, on='station', how='left')
                results = pd.merge(results, freq_large_jump_pos, on='station', how='left')
                results = pd.merge(results, pp_extreme_large_jump_pos, on='station', how='left')
                results = pd.merge(results, freq_large_jump_neg, on='station', how='left')
                results = pd.merge(results, pp_extreme_large_jump_neg, on='station', how='left')

                # Add latitude, longitude, and land mask information
                results['lat'] = results.station.apply(lambda x: float(x.split('_')[0]))
                results['lon'] = results.station.apply(lambda x: float(x.split('_')[1]))
                results['is_land'] = results[['lat', 'lon']].apply(lambda r: globe.is_land(lat=r['lat'], lon=r['lon']), axis=1)

                # Calculate normalized extreme precipitation metrics
                extreme_pp_metrics = [
                    'large_jump_pos_extreme_pp_rate',
                    'large_jump_neg_extreme_pp_rate'
                ]

                for metric in extreme_pp_metrics:
                    results[f'{metric}_norm'] = results[metric] / results.extreme_pp_rate

                # Save results to CSV
                output_filename = f'{results_folder}/map_{season}_pp_{pp_lower_bound}_strat_{grouping_strategy}_extremes_local.csv'
                results.to_csv(output_filename, index=False)
                print(f"Results saved to: {output_filename}")
                print(f"Number of unique stations processed: {results.station.nunique()}")
                print(' ')

if __name__ == "__main__":
    main()