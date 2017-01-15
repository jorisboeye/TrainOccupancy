import csv
import json
import folium
from models import Train, Station

csv_path = '/home/joris/PycharmProjects/Kaggle/TrainOccupancy/data/stations.csv'
stations = []
with open(csv_path) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        stations.append(Station(row))

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


map_days = folium.Map(location=[50.85, 4.35], zoom_start=9)

kw = dict(opacity=1.0, weight=4)

for week_day in Train.week_day_mapping().values():
    fg=folium.FeatureGroup(name=week_day)
    for train in trains:
        if train.week_day() == week_day:
            origin = train.origin.coordinates()
            destination = train.destination.coordinates()
            line = folium.PolyLine(locations=[origin, destination], color=train.color(), **kw)
            fg.add_child(line)

    map_days.add_child(fg)
map_days.add_children(folium.map.LayerControl())
map_days.save('maps/occupancy_days.html')