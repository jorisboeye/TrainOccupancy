import warnings

warnings.filterwarnings('ignore')

# Standard libraries
import pandas as pd
import numpy as np

# Data processing
import json
from dateutil.parser import parse
import re

# Machine learning
from sklearn.cross_validation import StratifiedKFold
from xgboost import XGBClassifier

# Plotting
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance

# Global variables
NR_FOLDS = 5
NR_FEATURES = 20

_columns = ['id', 'querytime', 'seconds_since_midnight', 'hour', 'weekday', 'month', 'connection',
            'from', 'from_string', 'from_lat', 'from_lng', 'morning_jam', 'evening_jam',
            'to', 'to_string', 'to_lat', 'to_lng', 'vehicle', 'vehicle_type', 'occupancy',
            'year', 'day', 'quarter', 'vehicle_nr', 'line_nr']

stations_df = pd.read_csv('data/stations.csv')
stations_df['URI'] = stations_df['URI'].apply(lambda x: x.split('/')[-1])
week_day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}


def get_line_number(vehicle):
    pattern = re.compile("^([A-Z]+)[0-9]+$")
    vehicle_type = pattern.match(vehicle).group(1)
    pattern = re.compile("^[A-Z]+([0-9]+)$")
    vehicle_nr = int(pattern.match(vehicle).group(1))
    line_nr = 0
    if vehicle_type == 'IC':
        line_nr = str(int(100 * np.floor(vehicle_nr / 100)))
    elif vehicle_type == 'L':
        line_nr = str(int(50 * np.floor(vehicle_nr / 50)))
    elif vehicle_type == 'S':
        line_nr = str(int(50 * np.floor(vehicle_nr / 50)))
    else:
        line_nr = 'P'

    return vehicle_nr, line_nr


def parse_file(path):
    parsed_logs = []
    faulty_logs = 0
    time_zones = []
    with open(path) as data_file:
        for line in data_file:
            occ_logline = json.loads(line)
            morning_commute = 0
            evening_commute = 0
            commute_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            # Do a check if the querytype is occupancy
            if occ_logline['querytype'] == 'occupancy' and 'error' not in occ_logline \
                    and 'querytime' in occ_logline:

                try:
                    query_time = occ_logline['querytime']
                    try:
                        parsed_query_time = parse(query_time)
                        week_day = week_day_mapping[parsed_query_time.weekday()]
                        weekday_nr = parsed_query_time.weekday()
                        midnight = parsed_query_time.replace(hour=0, minute=0, second=0, microsecond=0)
                        seconds_since_midnight = (parsed_query_time - midnight).seconds
                        day = parsed_query_time.day
                        year = parsed_query_time.year
                        month = parsed_query_time.month
                        hour = parsed_query_time.hour
                        quarter = int(parsed_query_time.minute / 15)
                        timezone_offset = parsed_query_time.tzinfo._offset
                        time_zones.append(timezone_offset)
                        hours_offset, remainder = divmod(timezone_offset.seconds, 3600)
                        # De ochtendspits valt doorgaans in de periode van 7.00 tot 9.00 uur.
                        # De avondspits valt in de regel tussen 16.30 en 18.30 uur.
                        if 6 < (hour - hours_offset + 1) < 10 and week_day in commute_days:
                            morning_commute = 1
                        if 15 < (hour - hours_offset + 1) < 19 and week_day in commute_days:
                            evening_commute = 1
                    except ValueError:
                        faulty_logs += 1
                        continue

                    vehicle_id = occ_logline['post']['vehicle'].split('/')[-1]
                    connection = occ_logline['post']['connection']
                    if 'occupancy' in occ_logline['post']:
                        occupancy = occ_logline['post']['occupancy'].split('/')[-1]
                    else:
                        occupancy = 'unknown'

                    if 'to' in occ_logline['post']:
                        to_id = occ_logline['post']['to'].split('/')[-1]
                        if to_id[:2] == '00' and to_id != '00':
                            to_string = stations_df[stations_df['URI'] == to_id]['name'].values[0]
                            to_lat = stations_df[stations_df['URI'] == to_id]['latitude'].values[0]
                            to_lng = stations_df[stations_df['URI'] == to_id]['longitude'].values[0]
                        else:
                            to_string = None
                            to_lat = None
                            to_lng = None
                    else:
                        to_id = None
                        to_string = None
                        to_lat = None
                        to_lng = None

                    if 'from' in occ_logline['post']:
                        from_id = occ_logline['post']['from'].split('/')[-1]
                        if from_id[:2] == '00' and from_id != '00':
                            from_string = stations_df[stations_df['URI'] == from_id]['name'].values[0]
                            from_lng = stations_df[stations_df['URI'] == from_id]['longitude'].values[0]
                            from_lat = stations_df[stations_df['URI'] == from_id]['latitude'].values[0]
                        else:
                            from_string = None
                            from_lat = None
                            from_lng = None
                    else:
                        from_id = None
                        from_string = None
                        from_lat = None
                        from_lng = None

                    pattern = re.compile("^([A-Z]+)[0-9]+$")
                    try:
                        vehicle_type = pattern.match(vehicle_id).group(1)
                        vehicle_nr, line_nr = get_line_number(vehicle_id)
                    except Exception as e:
                        vehicle_type = None
                        vehicle_nr, line_nr = None, None

                    if 'id' in occ_logline:
                        _id = occ_logline['id']
                    else:
                        _id = -1

                    parsed_logs.append([_id, parsed_query_time, seconds_since_midnight, hour, week_day, month,
                                        connection, from_id, from_string, from_lat, from_lng, morning_commute,
                                        evening_commute, to_id, to_string, to_lat, to_lng, vehicle_id,
                                        vehicle_type, occupancy, year, day, quarter, vehicle_nr, line_nr])

                except Exception as e:
                    faulty_logs += 1
                    raise
                    continue
        return parsed_logs, faulty_logs


parsed_file1, faulty1 = parse_file('data/train.nldjson')
parsed_file2, faulty2 = parse_file('data/test.nldjson')
logs_df = pd.DataFrame(parsed_file1 + parsed_file2)
logs_df.columns = _columns
logs_df = logs_df[['id', 'seconds_since_midnight', 'weekday', 'vehicle_type',
                   'month', 'from_lat', 'from_lng', 'to_lat', 'to_lng', 'line_nr',
                   'morning_jam', 'evening_jam', 'occupancy', 'hour']]
logs_df = pd.get_dummies(logs_df, columns=['weekday', 'vehicle_type', 'line_nr'])
train_df = logs_df[logs_df['occupancy'] != 'unknown']
test_df = logs_df[logs_df['occupancy'] == 'unknown']
print(faulty1 + faulty2, 'logs discarded ---', len(logs_df), 'parsed')
print(len(train_df), 'training samples')
print(len(test_df), 'test samples')

labels_df = train_df['occupancy'].map({'low': 0, 'medium': 1, 'high': 2})
features_df = train_df.drop(['occupancy', 'id'], axis=1)

skf = StratifiedKFold(labels_df.values, n_folds=NR_FOLDS, shuffle=True, random_state=1337)

accuracies = []
for fold, (train_idx, test_idx) in enumerate(skf):
    print('Fold', fold + 1, '/', NR_FOLDS)
    X = features_df.iloc[train_idx, :].reset_index(drop=True)
    y = labels_df.iloc[train_idx].reset_index(drop=True)

    msk = np.random.rand(len(X)) < 0.9
    X_train = X[msk]
    X_val = X[~msk]
    y_train = y[msk]
    y_val = y[~msk]

    X_test = features_df.iloc[test_idx, :].reset_index(drop=True)
    y_test = labels_df.iloc[test_idx].reset_index(drop=True)

    weights = np.zeros((len(y_train)))
    for i in range(len(y_train)):
        label = y_train.iloc[i]
        #         weights[i] = [1.25, 1, 1.5][int(label)]
        weights[i] = 1

    xgb = XGBClassifier(n_estimators=5000, max_depth=4, min_child_weight=6, learning_rate=0.01,
                        colsample_bytree=0.5, subsample=0.6, gamma=0., nthread=-1,
                        max_delta_step=1, objective='multi:softmax')
    xgb.fit(X_train, y_train, sample_weight=weights, verbose=10,
            eval_metric=['merror', 'mlogloss'], eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50)

    results = xgb.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='logloss Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='logloss Val')
    ax.plot(x_axis, results['validation_0']['merror'], label='error Train')
    ax.plot(x_axis, results['validation_1']['merror'], label='error Val')
    ax.legend(loc='center left')
    plt.ylabel('Error')
    plt.title('XGBoost Logloss/Classification Error')
    plt.show()

    selected_features_idx = xgb.feature_importances_.argsort()[-NR_FEATURES:][::-1]
    plt.bar(range(len(selected_features_idx)), [xgb.feature_importances_[i] for i in selected_features_idx])
    plt.xticks(range(len(selected_features_idx)), [features_df.columns[i] for i in selected_features_idx],
               rotation='vertical')
    plt.show()

    predictions = xgb.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(conf_matrix)
    accuracy = sum([conf_matrix[i][i] for i in range(len(conf_matrix))]) / np.sum(conf_matrix)
    print('accuracy:', accuracy)
    accuracies.append(accuracy)

print('Avg accuracy:', np.mean(accuracies), np.std(accuracies))

labels_df = train_df['occupancy'].map({'low': 0, 'medium': 1, 'high': 2})
features_df = train_df.drop(['occupancy', 'id'], axis=1)
test_features_df = test_df.drop(['occupancy', 'id'], axis=1)

xgb = XGBClassifier(n_estimators=5000, max_depth=4, min_child_weight=6, learning_rate=0.01,
                    colsample_bytree=0.5, subsample=0.6, gamma=0., nthread=-1,
                    max_delta_step=1, objective='multi:softmax')
xgb.fit(features_df, labels_df, sample_weight=[1] * len(labels_df))

prediction_vectors = []
for prediction, _id in zip(xgb.predict(test_features_df), test_df['id'].values):
    prediction_vectors.append([_id, prediction])
prediction_df = pd.DataFrame(prediction_vectors)
prediction_df.columns = ['id', 'occupancy']

prediction_df.to_csv('submission.csv', index=False)