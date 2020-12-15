import numpy as np
import pandas as pd
import calendar
import sys

from utils.helpers import *
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pickle import dump, load

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from keras.utils import np_utils

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

def train_model(df, target='is_returning_customer', save_path='models/nn_latest.json'):
    """[Train model]

    Args:
        filename (str): [training data]. Defaults to 'train_cases.csv'.

    Returns:
        [None]: [None]
    """
    features = [s for s in df.columns if s not in ['customer_id', target]]
    dump(features, open('models/featureList.pkl', 'wb'))

    seed = 12
    X = df[['customer_id'] + features].fillna(1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        stratify=y, 
        test_size=0.05, 
        random_state=seed)
    test_customer_id = set(X_test.customer_id)
    X_train = X_train[features].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    dump(scaler, open('models/scaler.pkl', 'wb'))

    # y_train = y_train.values
    # encode class values as integers
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(y_train)
    # convert integers to OneHot variables
    onehot_labels = np_utils.to_categorical(encoded_labels)

    def nn_model():

        # Create model here
        model = Sequential()
        model.add(Dense(128,# kernel_regularizer=regularizers.l2(0.0001),
                    activation='elu', input_shape=(len(features), ))) # Rectified Linear Unit Activation Function
        model.add(Dropout(0.2))
        model.add(Dense(128,# kernel_regularizer=regularizers.l2(0.0001),
                    activation='elu')) # Rectified Linear Unit Activation Function
        model.add(Dropout(0.2))
        model.add(Dense(128,# kernel_regularizer=regularizers.l2(0.0001),
                    activation='elu')) # Rectified Linear Unit Activation Function
        model.add(Dropout(0.2))
        model.add(Dense(128,# kernel_regularizer=regularizers.l2(0.0001),
                    activation='elu')) # Rectified Linear Unit Activation Function
        model.add(Dropout(0.2))
        model.add(Dense(2, activation = 'softmax')) # Softmax for multi-class classification

        # Compile model here
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        
        return model
    
    # For tuning NN architecture
    estimator = KerasClassifier(build_fn=nn_model, 
        epochs=50, 
        batch_size=128, 
        callbacks=[EarlyStopping(monitor='val_categorical_crossentropy', patience=5)],
        verbose=True
        )
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
    # fit on one fold
    for train, test in skfold.split(X_train, y_train):
        estimator.fit(X_train[train], onehot_labels[train], 
            validation_data=(X_train[test], onehot_labels[test]),
            )
        break
    # Got log loss 0.4749

    # estimator.model.summary()
    if save_path != '':
        estimator.model.save(save_path)
    return estimator.model, test_customer_id

def predict(df_order_test):
    """[Predict on test data and save to a file named result.csv.]

    Args:
        filename (str): [test data path]. Defaults to 'sample_test.csv'.
    """
    features = load(open('models/featureList.pkl', 'rb'))
    X_test = df_order_test[features].values
    scaler = load(open('models/scaler.pkl', 'rb'))
    X_test = scaler.transform(X_test)

    model_saved = load_model('models/nn_latest.json')
    
    preds_probs = model_saved.predict(X_test)
    preds_probs_1 = [x[1] if not pd.isna(x[1]) else 0 for x in preds_probs]
    preds_class = np.argmax(preds_probs, axis=1)
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
        
        




