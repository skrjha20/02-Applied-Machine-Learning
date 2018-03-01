import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.neural_network import MLPClassifier

def time_gap(hearing_date_str, ticket_issued_date_str):

    if not hearing_date_str or type(hearing_date_str)!=str: 
        return 73
    hearing_date = datetime.strptime(hearing_date_str, "%Y-%m-%d %H:%M:%S")
    ticket_issued_date = datetime.strptime(ticket_issued_date_str, "%Y-%m-%d %H:%M:%S")
    gap = hearing_date - ticket_issued_date
    return gap.days

def blight_model():
    
    train = pd.read_csv('train.csv', encoding = 'ISO-8859-1')
    test = pd.read_csv('test.csv')
    
    train = train[(train['compliance'] == 0) | (train['compliance'] == 1)]
    address =  pd.read_csv('addresses.csv')
    latlons = pd.read_csv('latlons.csv')
    
    address = address.set_index('address').join(latlons.set_index('address'), how='left')
    
    train = train.set_index('ticket_id').join(address.set_index('ticket_id'))
    test = test.set_index('ticket_id').join(address.set_index('ticket_id'))
    
    train = train[~train['hearing_date'].isnull()]
    train['time_gap'] = train.apply(lambda row: time_gap(row['hearing_date'], row['ticket_issued_date']), axis=1)
    test['time_gap'] = test.apply(lambda row: time_gap(row['hearing_date'], row['ticket_issued_date']), axis=1)
    
    feature_to_be_splitted = ['agency_name', 'state', 'disposition']
    train.lat.fillna(method='pad', inplace=True)
    train.lon.fillna(method='pad', inplace=True)
    train.state.fillna(method='pad', inplace=True)

    test.lat.fillna(method='pad', inplace=True)
    test.lon.fillna(method='pad', inplace=True)
    test.state.fillna(method='pad', inplace=True)
    
    train = pd.get_dummies(train, columns=feature_to_be_splitted)
    test = pd.get_dummies(test, columns=feature_to_be_splitted)
    list_to_remove_train = ['balance_due', 'collection_status', 'compliance_detail',
                            'payment_amount', 'payment_date', 'payment_status']
    list_to_remove_all = ['fine_amount', 'violator_name', 'zip_code', 'country', 'city',
                          'inspector_name', 'violation_street_number', 'violation_street_name',
                          'violation_zip_code', 'violation_description', 
                          'mailing_address_str_number', 'mailing_address_str_name',
                          'non_us_str_code',  'ticket_issued_date', 'hearing_date', 
                          'grafitti_status', 'violation_code']
    
    train.drop(list_to_remove_train, axis=1, inplace=True)
    train.drop(list_to_remove_all, axis=1, inplace=True)
    test.drop(list_to_remove_all, axis=1, inplace=True)
    train_features = train.columns.drop('compliance')
    train_features_set = set(train_features)
    
    for feature in set(train_features):
        if feature not in test:
            train_features_set.remove(feature)
    train_features = list(train_features_set)
    
    X_train = train[train_features]
    y_train = train.compliance
    X_test = test[train_features]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = MLPClassifier(hidden_layer_sizes = [100, 10], alpha = 5, random_state = 0, solver='lbfgs', verbose=0)
    clf.fit(X_train_scaled, y_train)

    test_proba = clf.predict_proba(X_test_scaled)[:,1]
    
    test_df = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    test_df['compliance'] = test_proba
    test_df.set_index('ticket_id', inplace=True)
    
    return test_df.compliance


if __name__ == '__main__':   
    
    blight_model()