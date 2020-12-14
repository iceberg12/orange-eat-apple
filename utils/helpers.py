import numpy as np
import pandas as pd
import calendar
from dateutil.relativedelta import relativedelta


def compute_df_freq_m(df):
    # Frequency at month level
    last_month = df.year_month.max()
    month_dict = {str(pd.to_datetime(last_month)+relativedelta(months=-j))[0:7]:'M-'+str(j+1) for j in range(0, 12)}
    df_freq_m = df.groupby(['customer_id', 'year_month'])['is_succeeded']\
        .agg('sum').reset_index(name='numOfOrders')
    df_freq_m['year_month'] = df_freq_m.year_month.apply(lambda s: month_dict[s]) 
    df_freq_m = pd.pivot_table(df_freq_m, index=['customer_id'], columns=['year_month'], 
            values=['numOfOrders'], aggfunc=np.mean)
    df_freq_m.columns = ['_'.join(col) for col in df_freq_m.columns.values]
    df_freq_m = df_freq_m.reset_index().fillna(0)
    return df_freq_m

def compute_df_rcc(df):
    last_month = df.year_month.max()
    current_date = pd.to_datetime(last_month) + relativedelta(months=1)
    maxday_gap = 365

    df_rcc = (df[['customer_id', 'is_failed', 'order_date']]
        .sort_values(['customer_id', 'is_failed', 'order_date'], ascending=True)
        .drop_duplicates(['customer_id', 'is_failed'], keep='last'))
    df_rcc['is_failed'] = df_rcc.is_failed.apply(lambda x: 'failed' if x==1 else 'succeeded')
    df_rcc['last_order_in_days'] = df_rcc.order_date.apply(lambda d:
        (current_date - pd.to_datetime(d)).days)
    df_rcc = pd.pivot_table(df_rcc, index=['customer_id'], columns=['is_failed'], 
            values=['last_order_in_days'], aggfunc=np.mean)
    df_rcc.columns = ['_'.join(col) for col in df_rcc.columns.values]
    df_rcc = df_rcc.reset_index()
    df_rcc.fillna(value=maxday_gap, inplace=True)
    return df_rcc

def compute_df_money(df):
    df_money = df.groupby('customer_id')[['voucher_amount', 'delivery_fee', 'amount_paid']]\
        .agg(['sum','median']).reset_index()
    df_money.columns = ['_'.join(col).strip('_') for col in df_money.columns.values]
    return df_money

def compute_df_misc(df):
    df_misc = df.groupby('customer_id').agg({
        'customer_order_rank': 'max',
        'is_failed': 'sum',
        'is_succeeded': 'sum'
    })
    df_misc.rename(columns={'is_failed':'is_failed_sum',
                           'is_succeeded':'is_succeeded_sum'}, inplace=True)
    df_misc.reset_index(inplace=True)
    return df_misc

def compute_df_hour(df):
    df_hour = df.groupby('customer_id')['order_hour'].value_counts(normalize=True)
    
    df_hour = df_hour.reset_index(name='hour')
    df_hour['order_hour'] = df_hour.order_hour.apply(str)
    df_hour = pd.pivot_table(df_hour, index=['customer_id'], columns=['order_hour'], 
            values=['hour'], aggfunc=np.mean)
    df_hour.columns = ['_'.join(col).strip('_') for col in df_hour.columns.values]
    df_hour.fillna(0, inplace=True)
    df_hour.reset_index(inplace=True)
    return df_hour

def compute_df_weekday(df):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    df_weekday = df.groupby('customer_id')['weekday'].value_counts(normalize=True)
    df_weekday = df_weekday.reset_index(name='weekdays')
    df_weekday = pd.pivot_table(df_weekday, index=['customer_id'], columns=['weekday'], 
            values=['weekdays'], aggfunc=np.mean)
    df_weekday.columns = ['_'.join(col).strip('_') for col in df_weekday.columns.values]
    df_weekday.fillna(0, inplace=True)
    df_weekday.reset_index(inplace=True)
    return df_weekday

def compute_df_payment(df):
    df_payment = df.groupby('customer_id')['payment_id'].value_counts(normalize=True)
    df_payment = df_payment.reset_index(name='payment')
    df_payment = pd.pivot_table(df_payment, index=['customer_id'], columns=['payment_id'], 
            values=['payment'], aggfunc='first')
    df_payment.columns = ['_'.join(col).strip('_') for col in df_payment.columns.values]
    df_payment.fillna(0, inplace=True)
    df_payment.reset_index(inplace=True)
    return df_payment

def compute_df_platform(df):
    df_platform = df.groupby('customer_id')['platform_id'].value_counts(normalize=True)
    df_platform = df_platform.reset_index(name='platform')
    df_platform = pd.pivot_table(df_platform, index=['customer_id'], columns=['platform_id'], 
            values=['platform'], aggfunc=np.mean)
    df_platform.columns = ['_'.join(col).strip('_') for col in df_platform.columns.values]
    df_platform.fillna(0, inplace=True)
    df_platform.reset_index(inplace=True)
    return df_platform

def compute_df_transmission(df):
    df_transmission = df.groupby('customer_id')['transmission_id'].value_counts(normalize=True)
    df_transmission = df_transmission.reset_index(name='transmission')
    df_transmission = pd.pivot_table(df_transmission, index=['customer_id'], columns=['transmission_id'], 
            values=['transmission'], aggfunc=np.mean)
    df_transmission.columns = ['_'.join(col).strip('_') for col in df_transmission.columns.values]
    df_transmission.fillna(0, inplace=True)
    df_transmission.reset_index(inplace=True)
    return df_transmission

def compute_df_res(df):
    df_res = df.groupby('customer_id')['restaurant_id'].value_counts(normalize=True)
    df_res = df_res.reset_index(name='top_restaurant_perc').drop('restaurant_id', axis='columns')
    df_res = df_res.groupby('customer_id').first().reset_index()
    return df_res


