import pickle


def load_models():
    clfs = []
    for indx in range(2):
        clfs.append(pickle.load(open('xgb{}.pickle'.format(indx + 1), 'rb')))
    return clfs


def preprocess(df):
    df = df.drop(['registrationid', 'buildingid', 'housenumber', 'streetname', 'boro', 'zip', 'recordstatus',
                  'contactdescription', 'firstname', 'lastname', 'corporationname', 'churned'],
                 axis=1, errors='ignore')
    p_i_start, p_i_end = df.columns.get_loc('percent_condo_portfolio'), df.columns.get_loc('percent_hdfc_portfolio')
    df.iloc[:, p_i_start:p_i_end + 1] = df.iloc[:, p_i_start:p_i_end + 1].fillna(0, axis=1)
    df = df.fillna(df.median())
    df.registered = df.registered.apply(lambda s: 1 if s == 'YES' else 0)
    return df


def preprocess_part2_for_second_model(df):
    df = df.drop(['energy_efficiency', 'number_of_ecb_violations_last_year',
                  'hmcv_violations_past_year_class_b',
                  'hmcv_violations_past_year_class_c',
                  'hmcv_violations_2_years_prior_class_b',
                  'violations_2_years_prior', 'legalclassb',
                  'percent_genpart_portfolio', 'percent_condominium_portfolio',
                  'percent_hdfc_portfolio'],
                 axis=1, errors='ignore')
    return df


def predict(X):
    clf = load_models()
    X = preprocess(X)
    y = {'more_precision': clf[0].predict(X)}
    X2 = preprocess_part2_for_second_model(X)
    y['more_recall'] = clf[1].predict(X2)
    return y
