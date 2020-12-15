import pytest
from main import *
import pandas as pd
from pickle import dump, load
from sklearn.metrics import accuracy_score, log_loss
import catboost

testdata = [('data/df_order_test.csv', 'data/df_label_test.csv')]

@pytest.mark.parametrize('filepaths', testdata)
def test_pipeline(filepaths):
    
    df_order_test = pd.read_csv(filepaths[0])
    for c in df_order_test.columns:
        if '_id' in c:
            df_order_test[c] = df_order_test[c].apply(str)

    print('Testing `processing` functions ...')
    features = load(open('models/featureList.pkl', 'rb'))
    df_order_test = preprocessing(df_order_test, features)
    assert set(df_order_test.columns.tolist()) == set(['customer_id'] + features),\
        'Processing func does not produce expected features.'
    print('Passed.')

    print('Testing `training` functions ...')
    df_label_test = pd.read_csv(filepaths[1])
    data = df_order_test.merge(df_label_test, on='customer_id')
    model, _ = train_model(data, target='is_returning_customer', save_path='')
    assert type(model) == catboost.core.CatBoostClassifier,\
        'Training func does not produce a catboost model.'
    print('Passed.')

    print('Testing `predict` functions ...')
    result = predict(df_order_test)
    logloss_result = log_loss(df_label_test.is_returning_customer, result.is_returning_customer_probs)
    acc_result = accuracy_score(df_label_test.is_returning_customer, result.is_returning_customer)
    assert logloss_result < 0.5, 'Predict func does not produce a expected log loss (less than 0.5).'
    assert acc_result > 0.7, 'Predict func does not produce a expected accuracy (more than 0.7).'
    print('Passed.')
