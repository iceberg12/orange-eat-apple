import numpy as np
import pandas as pd
import calendar
import sys

from utils.helpers import *
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
# from pycaret.classification import *
from catboost import CatBoostClassifier, Pool

from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load

def load_data():
    """Load data.

    Returns:
        tuple: order dataframe, label dataframe, one-hot features
    """
    df_order = pd.read_csv('data/machine_learning_challenge_order_data.csv')
    for c in df_order.columns:
        if '_id' in c:
            df_order[c] = df_order[c].apply(str)

    hour_feat = ['hour_' + str(s) for s in df_order.order_hour.unique().tolist()]
    payment_feat = ['payment_' + s for s in df_order.payment_id.unique().tolist()]
    platform_feat = ['platform_' + s for s in df_order.platform_id.unique().tolist()]
    transmission_feat = ['transmission_' + s for s in df_order.transmission_id.unique().tolist()]
    onehot_features = hour_feat + payment_feat + platform_feat + transmission_feat

    last_one_year = str(pd.to_datetime(df_order.order_date.max()) - relativedelta(months=11))[0:10]
    df_order = df_order[last_one_year <= df_order.order_date].reset_index(drop=True)
    
    df_label = pd.read_csv('data/machine_learning_challenge_labeled_data.csv')
    avail_customer_id = set(df_order.customer_id.unique())
    df_label = df_label[df_label.customer_id.isin(avail_customer_id)].reset_index(drop=True)
    return df_order, df_label, onehot_features

def preprocessing(df_order, features=[]):
    df_order['year_month'] = df_order.order_date.apply(lambda x: x[0:7])
    df_order['week_in_month'] = df_order.order_date.apply(lambda x: 
        'W1' if x[8:10] <= '07' 
        else('W2' if x[8:10] <= '14'
        else ('W3' if x[8:10] <= '21' 
        else 'W4')))
    df_order['weekday'] = pd.to_datetime(df_order.order_date).apply(lambda x: calendar.day_name[x.weekday()])
    df_order['is_succeeded'] = df_order.is_failed.apply(lambda x: 1-x)

    df_freq_m = compute_df_freq_m(df_order)
    df_rcc = compute_df_rcc(df_order)
    df_money = compute_df_money(df_order)
    df_misc = compute_df_misc(df_order)
    df_hour = compute_df_hour(df_order)
    df_weekday = compute_df_weekday(df_order)
    df_payment = compute_df_payment(df_order)
    df_platform = compute_df_platform(df_order)
    df_transmission = compute_df_transmission(df_order)
    df_res = compute_df_res(df_order)
    
    data = (df_freq_m
        .merge(df_rcc, on='customer_id')
        .merge(df_money, on='customer_id')
        .merge(df_misc, on='customer_id')
        .merge(df_hour, on='customer_id')
        .merge(df_weekday, on='customer_id')
        .merge(df_payment, on='customer_id')
        .merge(df_platform, on='customer_id')
        .merge(df_transmission, on='customer_id')
        .merge(df_res, on='customer_id'))
    
    for col in features:
        if col not in data.columns:
            data[col] = 0
    return data

def train_model(df, target='', save_path='models/catboost_manual_latest.json'):
    """[Train model]

    Args:
        filename (str): [training data]. Defaults to 'train_cases.csv'.

    Returns:
        [None]: [None]
    """
    features = [s for s in df.columns if s not in ['customer_id', target]]
    dump(features, open('models/featureList.pkl', 'wb'))
    
    # Try PyCaret
    # scaler = MinMaxScaler()
    # # fit and save scaler on the training dataset
    # df[features] = scaler.fit_transform(df[features])
    # dump(scaler, open('scaler.pkl', 'wb'))
    # clf_setup = setup(df, 
    #     target = target, 
    #     numeric_features=features,
    #     # fix_imbalance=True,
    #     data_split_stratify=True,
    #     fold_strategy='stratifiedkfold',
    #     html=False, silent=True, verbose=True)
    # add_metric('logloss', 'logloss', log_loss, greater_is_better = False)
    # best = compare_models(exclude=['xgboost'], fold=5, errors='ignore', sort='Log Loss')
    # # Best: ?

    seed = 12
    X = df[['customer_id'] + features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        stratify=y, 
        test_size=0.05, 
        random_state=seed)
    test_customer_id = set(X_test.customer_id)
    train_pool = Pool(X_train[features], y_train)
    val_pool = Pool(X_test[features], y_test) 
    # train model
    model = CatBoostClassifier(iterations=5000,
        learning_rate=0.01,
        loss_function='Logloss',
        # auto_class_weights='Balanced',
        verbose=100)
    model.fit(train_pool,
        eval_set=val_pool,
        early_stopping_rounds=500,
        use_best_model=False)

    if save_path != '':
        model.save_model(save_path,
            format='json',
            export_parameters=None,
            pool=None)
    return model, test_customer_id

def predict(df_order_test):
    """[Predict on test data and save to a file named result.csv.]

    Args:
        filename (str): [test data path]. Defaults to 'sample_test.csv'.
    """
    features = load(open('models/featureList.pkl', 'rb'))

    X_test = df_order_test[features]

    model_saved = CatBoostClassifier()
    model_saved.load_model('models/catboost_manual.json', format='json')
    
    preds_probs = model_saved.predict_proba(X_test)
    preds_probs_1 = [x[1] for x in preds_probs]
    preds_class = model_saved.predict(X_test)
    df_order_test['is_returning_customer'] = preds_class
    df_order_test['is_returning_customer_probs'] = preds_probs_1
    result = df_order_test[['customer_id', 'is_returning_customer', 'is_returning_customer_probs']]

    return result

def create_unittest_data(df_order, df_label, test_customer_id):

    df_order_test = df_order[df_order.customer_id.isin(test_customer_id)]
    df_label_test = df_label[df_label.customer_id.isin(test_customer_id)]
    
    df_order_test.to_csv('data/df_order_test.csv', index=False)
    df_label_test.to_csv('data/df_label_test.csv', index=False)
    return

if __name__ == '__main__':
    mode = sys.argv[1]
    
    if mode == 'train':
        df_order, df_label, onehot_features = load_data()
        data = preprocessing(df_order, onehot_features)
        data = data.merge(df_label, on='customer_id')

        print('Training model and creating test cases ...')
        _, test_customer_id = train_model(data, target='is_returning_customer')
        create_unittest_data(df_order, df_label, test_customer_id)
        print('Done')

    if mode == 'predict':
        filename = sys.argv[2]
        # filename = 'data/df_order_test.csv'
        df_order_test = pd.read_csv(filename)
        for c in df_order_test.columns:
            if '_id' in c:
                df_order_test[c] = df_order_test[c].apply(str)
        features = load(open('models/featureList.pkl', 'rb'))
        df_order_test = preprocessing(df_order_test, features)

        print('Performing prediction...')
        result = predict(df_order_test)
        print('Done.')
        print('Saving prediction output to data/result.csv ...')
        result.to_csv('data/result.csv', index=False)
        print('Done.')
        
        




