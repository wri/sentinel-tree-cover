import pandas as pd
import numpy as np

def check_stats(country, extent):
    '''
    Checks a country's statistics spreadsheet for erroneous values.
    '''

    df = pd.read_csv(f'statistics/{country}_statistics_{extent}.csv')
    processing_area = pd.read_csv(f'statistics/full_processing_area.csv')
    actual_sampled = processing_area[processing_area.country == country].iloc[0][1]

    sampled = df[['admin', 'esa_class', 'esa_id', 'esa_sampled_ha', 'esa_total_ha']]
    sampled = sampled.drop_duplicates(keep='first', ignore_index=True)
    sampled = sampled.groupby(by=['esa_class']).sum().reset_index()
    est_sampled = round(sampled.esa_sampled_ha.sum())
    perc_diff = round(((actual_sampled - est_sampled) / est_sampled)*100, 2)

    print(f'SUMMARY FOR {country}:')
    print(' ')
    print(f'Processing extent (ha)')
    print(f'Estimated: {est_sampled}')
    print(f'Actual: {actual_sampled}')
    print(f'Diff: %{perc_diff}')
    print(' ')
    print(f'{df.esa_class.nunique()} of 24 ESA land cover classes represented')
    print('Frequency (% of total land cover classes)')
    print(df.esa_class.value_counts(normalize=True)*100)
    print(' ')
    print('Check for erroneous values (TOF mean is NaN if ha == 0)')
    if len(df[df.tof_mean >= 100]) > 0:
        print(f'Warning: {len(df[df.tof_mean >= 100])} TML means >= 100%')
    print(f'TOF NaN means: {len(df[df.tof_mean.isnull()])}')

    # if the hansen analysis is included
    if extent == 'full':
        if len(df[df.hans_mean >= 100]) > 0:
            print(f'Warning: {len(df[df.hans_mean >= 100])} Hans means >= 100%')
        print(f'Hans NaN means: {len(df[df.hans_mean.isnull()])}')

    if len(df[(df.tof_mean == 0) & (df.tof_ha > 0)]):
        print('Warning: Hectares counted but mean tree cover = 0%')
    print(' ')
    return None


def create_regional_csv(list_of_countries, region, processing_extent='full'):

    regional_df = pd.DataFrame()
    dfs_to_concat = []

    for country in list_of_countries:
        try:
            country_df = pd.read_csv(f'statistics/{country}_statistics_{processing_extent}.csv')
        except OSError:
            country_df = pd.read_csv(f'statistics/{country}_statistics_{processing_extent}_tmlonly.csv')

        dfs_to_concat.append(country_df)

    regional_df = pd.concat(dfs_to_concat, ignore_index=True)
    regional_df.to_csv(f'statistics/{region}.csv', index=False)

    return None
