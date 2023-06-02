import pandas as pd
import numpy as np
import confuse
import boto3
import os

def check_stats(country, extent, adm2):
    '''
    Checks a country's statistics spreadsheet for erroneous values.
    Including
        calculates difference b/w the estimated area sampled and actual area sampled (from csv)
        prints total land cover classes in the country and frequency
        confirms there is no mean tree cover if there are no ha tree cover
        confirms the count of NaNs
        confirms there are no means above 100%
        confirms total TML ha is equivalent to total ha sampled


    '''
    # download stats csv
    config = confuse.Configuration('sentinel-tree-cover')
    config.set_file('/Users/jessica.ertel/sentinel-tree-cover/jessica-config.yaml')
    aws_access_key = config['aws']['aws_access_key_id']
    aws_secret_key = config['aws']['aws_secret_access_key']
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key.as_str(), aws_secret_access_key=aws_secret_key.as_str())

    # download stats if they don't exist locally
    if not os.path.exists(f'statistics/{country}_statistics_{extent}.csv'):
        s3.download_file('tof-output',
                        f'2020/analysis/2020-full/{country}/stats/{country}_statistics_{extent}.csv',
                        f'/Users/jessica.ertel/sentinel-tree-cover/notebooks/analysis/statistics/{country}_statistics_{extent}.csv')


    df = pd.read_csv(f'statistics/{country}_statistics_{extent}.csv')

    if adm2:
        df = pd.read_csv(f'statistics/{country}_statistics2_{extent}.csv')

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
    print('Check for erroneous values')
    nan_means = len(df[df.tof_mean.isnull()])
    if nan_means > 0:
        print(f'TOF NaN means: {nan_means} (TOF mean is NaN if esa sampled == 0)')
    else:
        print('No NaN means')
    if len(df[df.tof_mean >= 100]) > 0:
        print(f'Warning: {len(df[df.tof_mean >= 100])} TML mean values >= 100%')
    else:
        print('No mean values above 100%')


    # if the hansen analysis is included
    if extent == 'full':
        if len(df[df.hans_mean >= 100]) > 0:
            print(f'Warning: {len(df[df.hans_mean >= 100])} Hans means >= 100%')
        print(f'Hans NaN means: {len(df[df.hans_mean.isnull()])}')

    if len(df[(df.tof_mean == 0) & (df.tof_ha > 0)]):
        print('Warning: Hectares counted but mean tree cover = 0%')
    print(' ')

    # get sampled ha df
    admin_sampled = df[['admin', 'esa_id', 'esa_sampled_ha']]
    admin_sampled = admin_sampled.drop_duplicates()
    admin_sampled = admin_sampled.groupby('admin').sum()
    admin_sampled = admin_sampled[['esa_sampled_ha']]

    # get total ha df
    admin_tof = df[['admin', 'esa_id', 'tree_cover_class', 'tof_ha']]
    admin_tof = admin_tof.groupby('admin').sum()
    admin_tof = admin_tof[['tof_ha']]

    # confirm total tof ha == total sampled ha
    print('Check total TML ha = total area sampled:')
    print(admin_sampled.esa_sampled_ha == admin_tof.tof_ha)

    # confirm esa sampled is never greater than esa total
    if df.esa_sampled_ha.any() > df.esa_total_ha.any():
        print(f'Warning: ESA sampled is greater than ESA total')

    return None


def create_regional_csv(list_of_countries, region, processing_extent='full_tmlonly'):

    regional_df = pd.DataFrame()
    dfs_to_concat = []

    for country in list_of_countries:
        try:
            country_df = pd.read_csv(f'statistics/{country}_statistics_{processing_extent}.csv')
        except OSError as e:
            print(f'Trying alternative processing extent for {country}')
            return e

        dfs_to_concat.append(country_df)

    regional_df = pd.concat(dfs_to_concat, ignore_index=True)

    # for Brazil, China, Indonesia, Australia, combine admins and save as country spreadsheet
    if region == 'China':
        regional_filename = f'statistics/{region}_statistics_{processing_extent}.csv'
        regional_df.country = region

    else:
        regional_filename = f'statistics/{region}.csv'

    regional_df.to_csv(regional_filename, index=False)

    return None
