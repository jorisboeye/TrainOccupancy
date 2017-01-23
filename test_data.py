import csv
import json
import re
import folium
import pandas as pd
from models import Train, Station, Trip

station_csv_path = '/home/joris/PycharmProjects/Kaggle/TrainOccupancy/data/stations.csv'
stations = []
with open(station_csv_path) as station_csv_file:
    station_reader = csv.DictReader(station_csv_file)
    for row in station_reader:
        stations.append(Station(row))


trips_csv_path = '/home/joris/PycharmProjects/Kaggle/TrainOccupancy/data/nmbs-2016-09/trips.txt'
trips_df = pd.read_csv(trips_csv_path)

stops_csv_path = '/home/joris/PycharmProjects/Kaggle/TrainOccupancy/data/nmbs-2016-09/stop_times.txt'
stops_df = pd.read_csv(stops_csv_path)

json_path = '/home/joris/PycharmProjects/Kaggle/TrainOccupancy/data/train.nldjson'
trains = []
load_failures = 0
with open(json_path) as json_file:
    for line in json_file:
        json_line = json.loads(line)
        train = Train(json_line=json_line, stations=stations)
        if train.load_success and train.origin is not None and train.destination is not None:
            trains.append(train)
        else:
            load_failures += 1

print(len(trains), load_failures)

# trips_df['route_id'] = re.sub(r"routes:", "", trips_df['route_id'])
# trips_df['route_id'].str.replace(to_replace='routes:', value='', inplace=True)
trips_df.replace({'routes:': ''}, regex=True, inplace=True)

print(trips_df.head(n=5))
print(stops_df.head(n=5))

for train in trains[100:110]:
    print('{:*>80s}'.format(''))
    print(train.vehicle)
    print(trips_df[trips_df['route_id'] == train.vehicle])
    trip_id = trips_df[trips_df['route_id'] == train.vehicle]
    print(trip_id.get('trip_id'))
    # print(stops_df[stops_df['route_id'] == trip_id])


# map_days = folium.Map(location=[50.85, 4.35], zoom_start=9)
#
# kw = dict(opacity=1.0, weight=4)
#
# for week_day in Train.week_day_mapping().values():
#     fg=folium.FeatureGroup(name=week_day)
#     for train in trains:
#         if train.week_day() == week_day:
#             origin = train.origin.coordinates()
#             destination = train.destination.coordinates()
#             line = folium.PolyLine(locations=[origin, destination], color=train.color(), **kw)
#             fg.add_child(line)
#
#     map_days.add_child(fg)
# map_days.add_children(folium.map.LayerControl())
# map_days.save('maps/occupancy_days.html')
#
#
# map_hours = folium.Map(location=[50.85, 4.35], zoom_start=9)
#
# kw = dict(opacity=1.0, weight=4)
#
# for x in range(3600, 25*3600, 3600):
#     time_string = '{:0>2d}h - {:0>2d}h'.format(int((x - 3600)/3600), int(x/3600))
#     fg=folium.FeatureGroup(name=time_string)
#     for train in trains:
#         if x - 3600 <= train.seconds_since_midnight < x:
#             origin = train.origin.coordinates()
#             destination = train.destination.coordinates()
#             line = folium.PolyLine(locations=[origin, destination], color=train.color(), **kw)
#             fg.add_child(line)
#
#     map_hours.add_child(fg)
# map_hours.add_children(folium.map.LayerControl())
# map_hours.save('maps/occupancy_hours.html')