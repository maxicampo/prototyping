import netCDF4 as nc  
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

def generate_single_year_csv_era_data(year, raw_data_path):
    file_name_temp = f'{raw_data_path}/{year}_temp.nc' # file_name_temp = f'download_{year}.nc'
    file_name_pp = f'{raw_data_path}/{year}_pp.nc' # file_name_pp = f'download_pp_{year}.nc'

    dset_temp = xr.open_dataset(f'./{file_name_temp}')
    dset_pp = xr.open_dataset(f'./{file_name_pp}')

    df_temp = dset_temp.to_dataframe().reset_index().rename(columns={'valid_time': 'time'})
    df_pp = dset_pp.to_dataframe().reset_index().rename(columns={'valid_time': 'time'})
    
    df_daily = pd.merge(df_temp, df_pp, on=['latitude', 'longitude', 'time'])
    df_daily['t2m'] = df_daily['t2m'] - 273.15
    df_daily['tp'] = df_daily['tp']*1000
    
    df_daily['year'] = df_daily.time.dt.year
    df_daily['month'] = df_daily.time.dt.month
    df_daily['day'] = df_daily.time.dt.day
    df_daily['hour'] = df_daily.time.dt.hour

    df_daily = df_daily.drop(columns='time')
    df_daily['station'] = df_daily['latitude'].astype(str)+'_'+df_daily['longitude'].astype(str)
    
    df = df_daily.groupby(['year', 'month', 'day', 'station']).agg({'t2m': 'max', 
                                                                    'tp': 'sum',
                                                                    'latitude': 'max',
                                                                    'longitude': 'max'})

    df = df.rename(columns={'t2m': 'TMAX', 'tp': 'PP'}).reset_index()

    return df

def generate_csv_era_data(raw_data_path, output_path):
    print(1980)
    df_era = generate_single_year_csv_era_data(1980, raw_data_path)
    df_era.to_csv(f'{output_path}/daily_era_data.csv', index=False)
    
    for year in range(1981, 2025):
        print(year)
        df = generate_single_year_csv_era_data(year, raw_data_path)
        df_era = pd.concat([df, df_era], axis=0)
        df_era.to_csv(f'{output_path}/daily_era_data.csv', index=False)

    df_era.to_csv(f'{output_path}/daily_era_data.csv', index=False)

if __name__ == '__main__':
    generate_csv_era_data(raw_data_path='era_data', output_path='era')