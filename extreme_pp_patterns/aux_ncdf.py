import pandas as pd
import numpy as np

agg_map = {
    1: 'sum',
    2: 'first',
    3: 'max'
}

def get_extreme_pp(df, strategy='global', perc=0.95, n_neighbors=8, pp_lower_bound=0):
    # Calculate extreme values according to strategy. 
    # Strategies:
    # 'global': Use the full dataset to calculate percentiles
    # 'local': Calculate percentiles on each lat/lon point
    # 'neighborhood': Calculate percentiles based on the nearest `n_neighbors`

    df_result = df.copy()
    
    if strategy == 'global':
        df_result['PP_extreme'] = np.where(df_result.PP >= df_result[df_result.PP >= pp_lower_bound].PP.quantile(perc), 1, 0)
    if strategy == 'local':
        df_percentiles = df_result[df_result.PP >= pp_lower_bound].groupby('station').PP.quantile(perc).reset_index().rename(columns={'PP': 'PP_perc_local'})
        df_result = pd.merge(df_result, df_percentiles, on='station')
        df_result['PP_extreme'] = np.where(df_result.PP >= df_result.PP_perc_local, 1, 0)
        df_result = df_result.drop(columns=['PP_perc_local'])
    if strategy == 'neighborhood':
        pass

    return df_result

# Preprocesses data
def preprocess_data(data, season='summer', perc=0.95, pp_lower_bound=0, grouping_strategy=0, extreme_value_strategy='global'):
    
    df = data.copy()
    
    df['ymd'] = df.year*10000 + df.month*100 + df.day
    df = df.drop_duplicates(subset=['station', 'ymd'])
    df = df.sort_values(by=['station', 'ymd'])
    df['PP_prev'] = df.groupby('station')['PP'].shift(1)
    
    for idx in range(1,8):
        df[f'TMAX_{idx}'] = df.groupby('station')['TMAX'].shift(idx)

    if season == 'summer':
        df = df[df.month.isin([1,2,3])]
    elif season == 'winter':
        df = df[df.month.isin([6,7,8])]
    else:
        df = df

    df['rain'] = np.where(df.PP>=pp_lower_bound, 1, 0)
    
    # No grouping
    if grouping_strategy == 0:
        df_final = get_extreme_pp(df, strategy=extreme_value_strategy, perc=perc, pp_lower_bound=pp_lower_bound)
    # Drop rain happening in consecutive days
    elif grouping_strategy in [1, 2, 3]:
        df['consecutive_day'] = np.where((df.PP_prev >= pp_lower_bound)&(df.rain == 1), 0, 1)
        df['group'] = df.groupby('station').consecutive_day.cumsum()
    
        group_pp = df.groupby(['station', 'group']).PP.agg(agg_map[grouping_strategy]).reset_index().rename(columns={'PP': 'PP_group'})
        df_final = pd.merge(df, group_pp, on=['station', 'group'], how='left')
        df_final = df_final[df_final.consecutive_day==1]
        df_final = df_final.drop(columns=['consecutive_day', 'group', 'PP_prev', 'PP']).rename(columns={'PP_group': 'PP'})
        df_final = get_extreme_pp(df_final, strategy=extreme_value_strategy, perc=perc, pp_lower_bound=pp_lower_bound)
    
    df_final = df_final.dropna()
    
    return df_final

# Calculates the size of the increasing temperature chain for each precipitation day
def add_max_increasing_chain(df, eps=0):
    for i in range(1, 7):
        df[f'increase_{i}'] = np.where(df[f'TMAX_{i}'] + eps >= df[f'TMAX_{i+1}'], 1, 0)
    
    for i in range(1, 7):
        increase_cols = [f'increase_{j}' for j in range(1,i+1)]        
        df[f'increasing_chain_{i}'] = df[increase_cols].sum(axis=1)*df[increase_cols].min(axis=1)

    chain_cols = [f'increasing_chain_{i}' for i in range(1, 7)]
    df['max_increasing_chain'] = df[chain_cols].max(axis=1)+1

    return df

# Calculates the size of the high temperature chain for each precipitation day
def add_max_high_temp_chain(df, perc=0.75):
    thresh_t = df.groupby('station').TMAX.quantile(perc).to_frame('thresh_t').reset_index()
    
    df = pd.merge(df, thresh_t, on='station')
    
    for i in range(1,8):
        df[f'high_t_{i}'] = np.where(df[f'TMAX_{i}'] > df['thresh_t'], 1, 0)

    for i in range(1, 7):
        high_t_cols = [f'high_t_{j}' for j in range(1, i+1)]
        df[f'high_t_chain_{i}'] = df[high_t_cols].sum(axis=1)*df[high_t_cols].min(axis=1)

    high_t_chain_cols = [f'high_t_chain_{i}' for i in range(1, 7)]
    df['max_high_t_chain'] = df[high_t_chain_cols].max(axis=1)+1
    
    return df

# Adds flags for large jumps in temperature
def add_large_jump(df, perc=0.9, rain_delay=1):
    
    df['TGAP'] = (df['TMAX'] - df['TMAX_1']).apply(abs)
    large_gap = df.groupby('station').TGAP.quantile(perc).to_frame('large_gap').reset_index()
    df = pd.merge(df, large_gap, on='station')
    
    df['large_jump_pos'] = np.where((df[f'TMAX_{rain_delay}'] > df[f'TMAX_{rain_delay+1}'] + df['large_gap']), 1, 0)
    df['large_jump_neg'] = np.where((df[f'TMAX_{rain_delay}'] < df[f'TMAX_{rain_delay+1}'] - df['large_gap']), 1, 0)
    
    return df


# Computes features for increasing temperature chains
def calculate_increasing_temp_features(df_prep):
    
    # Chain length frequencies
    # total_days_per_chain = df_prep.groupby(['station', 'max_increasing_chain']).size().to_frame('increasing_chain_count').reset_index()
    # total_days = df_prep.groupby('station').size().to_frame('total_days').reset_index()

    # increasing_chain_freqs = pd.merge(total_days_per_chain, total_days, on='station')
    # increasing_chain_freqs['increasing_chain_freqs'] = increasing_chain_freqs['increasing_chain_count']/increasing_chain_freqs['total_days']
    # increasing_chain_freqs = increasing_chain_freqs.drop(columns=['total_days'])
    
    # Chain length PP extremes %
    # increasing_chain_pp_extreme = df_prep.groupby(['station', 'max_increasing_chain']).PP_extreme.mean().reset_index()
    
    # Average chain length
    avg_chain_length = df_prep.groupby('station').max_increasing_chain.mean().reset_index()
    avg_chain_length.columns = ('station', 'avg_increasing_chain_length')

    # Average chain length for extreme pp
    avg_chain_length_for_extremes = df_prep[df_prep.PP_extreme == 1].groupby('station').max_increasing_chain.mean().reset_index()
    avg_chain_length_for_extremes.columns = ('station', 'avg_increasing_chain_length_for_extremes')
    
    # Average chain length for extreme pp normalized by average chain length
    avg_chain_stats = pd.merge(avg_chain_length_for_extremes, avg_chain_length, on='station')
    avg_chain_stats['avg_increasing_chain_length_for_extremes_norm'] = avg_chain_stats['avg_increasing_chain_length_for_extremes']/avg_chain_stats['avg_increasing_chain_length']

    # Optimal chain length -> idea: compute weighted average of chain length by % of PP_extreme
    #optimal_length = increasing_chain_pp_extreme.copy()

    # max_increasing_chain_pp = increasing_chain_pp_extreme.groupby('station').PP_extreme.max()
    # optimal_length = pd.merge(optimal_length, max_increasing_chain_pp, on=['station', 'PP_extreme'])

    # optimal_length = optimal_length.sort_values(by=['station']).drop_duplicates(subset=['station'], keep='first')
    # optimal_length = optimal_length.rename(columns={'max_increasing_chain': 'optimal_length',
    #                                                 'PP_extreme': 'optimal_length_pp_extreme'})
        
    return avg_chain_stats # increasing_chain_freqs, increasing_chain_pp_extreme, avg_chain_length, avg_chain_length_for_extremes, optimal_length
    

# Computes features for high temperature chains
def calculate_high_temp_features(df_prep):
    
    # Chain length frequencies
    # total_days_per_chain = df_prep.groupby(['station', 'max_high_t_chain']).size().to_frame('high_t_chain_count').reset_index()
    # total_days = df_prep.groupby('station').size().to_frame('total_days').reset_index()

    # high_t_chain_freqs = pd.merge(total_days_per_chain, total_days, on='station')
    # high_t_chain_freqs['high_t_chain_freqs'] = high_t_chain_freqs['high_t_chain_count']/high_t_chain_freqs['total_days']
    # high_t_chain_freqs = high_t_chain_freqs.drop(columns=['total_days'])
    
    # Chain length PP extremes %
    # high_t_chain_pp_extreme = df_prep.groupby(['station', 'max_high_t_chain']).PP_extreme.mean().reset_index()
        
    # Average chain length
    avg_chain_length = df_prep.groupby('station').max_high_t_chain.mean().reset_index()
    avg_chain_length.columns = ('station', 'avg_high_t_chain_length')

    # Average chain length for extreme pp
    avg_chain_length_for_extremes = df_prep[df_prep.PP_extreme == 1].groupby('station').max_high_t_chain.mean().reset_index()
    avg_chain_length_for_extremes.columns = ('station', 'avg_high_t_chain_length_for_extremes')
    
    # Average chain length for extreme pp normalized by average chain length
    avg_chain_stats = pd.merge(avg_chain_length_for_extremes, avg_chain_length, on='station')
    avg_chain_stats['avg_high_t_chain_length_for_extremes_norm'] = avg_chain_stats['avg_high_t_chain_length_for_extremes']/avg_chain_stats['avg_high_t_chain_length']

    # Optimal chain length
    # optimal_length = high_t_chain_pp_extreme.copy()

    # max_high_t_chain_pp = high_t_chain_pp_extreme.groupby('station').PP_extreme.max()
    # optimal_length = pd.merge(optimal_length, max_high_t_chain_pp, on=['station', 'PP_extreme'])

    # optimal_length = optimal_length.sort_values(by=['station']).drop_duplicates(subset=['station'], keep='first')
    # optimal_length = optimal_length.rename(columns={'max_high_t_chain': 'optimal_length',
    #                                                 'PP_extreme': 'optimal_length_pp_extreme'})
    
    return avg_chain_stats # high_t_chain_freqs, high_t_chain_pp_extreme, avg_chain_length, avg_chain_length_for_extremes, optimal_length
    
# HERE
   
# Computes features for large temperature jumps
def calculate_temp_jump_features(df_prep):
    
    # % of large temperature upwards jumps
    freq_large_jump_pos = df_prep.groupby('station').large_jump_pos.mean().reset_index()

    # % of extreme PP for large upwards jumps
    pp_extreme_large_jump_pos = df_prep[df_prep.large_jump_pos == 1].groupby('station').PP_extreme.mean().reset_index()

    # % of large temperature downwards jumps
    freq_large_jump_neg = df_prep.groupby('station').large_jump_neg.mean().reset_index()

    # % of extreme PP for large downwards jumps
    pp_extreme_large_jump_neg = df_prep[df_prep.large_jump_neg == 1].groupby('station').PP_extreme.mean().reset_index()
    
    return freq_large_jump_pos, pp_extreme_large_jump_pos, freq_large_jump_neg, pp_extreme_large_jump_neg


# Run all metrics and add to data_dic for a single path
def run_single_path_metrics(path, data_dic):
    file = path.split('/')[2]
    feats = file.split('_')

    station = feats[0]
    temp = feats[1]
    season = feats[2]
    scenario = feats[3]
    model = feats[4]
    
    if season != 'summer' or temp != 'tmax':
        return None
    
    data_dic['path'].append(path)
    data_dic['station'].append(station)
    data_dic['scenario'].append(scenario)
    data_dic['model'].append(model)

    data = pd.read_csv(path)
        
    df = preprocess_data(data, season='summer', pp_only=True)
    df = add_max_increasing_chain(df, eps=0.5)
    df = add_max_high_temp_chain(df, perc=0.75)
    df = add_large_jump(df, perc=0.9, rain_delay=1)
    
    # increasing_chain_freqs, \
    # increasing_chain_pp_extreme, \
    # increasing_chain_avg_length, \
    # increasing_chain_optimal_length, \
    # increasing_chain_pp_extreme_optimal_length 
    increasing_chain_stats = calculate_increasing_temp_features(df)

    data_dic['increasing_chain_stats'].append(increasing_chain_stats)
    # data_dic['increasing_chain_optimal_length'].append(increasing_chain_optimal_length)
    # data_dic['increasing_chain_pp_extreme_optimal_length'].append(increasing_chain_pp_extreme_optimal_length)

    # for i in range(1,8):
    #    data_dic[f'increasing_chain_freq_{i}'].append(increasing_chain_freqs[i])
    #    data_dic[f'increasing_chain_pp_ext_{i}'].append(increasing_chain_pp_extreme[i])

    # high_t_chain_freqs,\
    # high_t_chain_pp_extreme,\
    # high_t_avg_chain_length,\
    # high_t_chain_optimal_length,\
    # high_t_chain_pp_extreme_optimal_length = 
    high_t_chain_stats = calculate_high_temp_features(df)

    data_dic['high_t_chain_stats'].append(high_t_chain_stats)
    # data_dic['high_t_chain_optimal_length'].append(high_t_chain_optimal_length)
    # data_dic['high_t_chain_pp_extreme_optimal_length'].append(high_t_chain_pp_extreme_optimal_length)

    # for i in range(1,8):
        # data_dic[f'high_t_chain_freq_{i}'].append(high_t_chain_freqs[i])
        # data_dic[f'high_t_chain_pp_ext_{i}'].append(high_t_chain_pp_extreme[i])

    large_jump_pos_freq,\
    large_jump_pos_pp_ext,\
    large_jump_neg_freq,\
    large_jump_neg_pp_ext = calculate_temp_jump_features(df)

    data_dic['large_jump_pos_freq'].append(large_jump_pos_freq)
    data_dic['large_jump_pos_pp_ext'].append(large_jump_pos_pp_ext)
    data_dic['large_jump_neg_freq'].append(large_jump_neg_freq)
    data_dic['large_jump_neg_pp_ext'].append(large_jump_neg_pp_ext)