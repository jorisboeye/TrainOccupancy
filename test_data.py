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


map = folium.Map(location=[50.85, 4.35], zoom_start=9)

kw = dict(opacity=1.0, weight=4)

fg=folium.FeatureGroup(name="Train occupancies")

for train in trains:
    origin = train.origin.coordinates()
    destination = train.destination.coordinates()
    line = folium.PolyLine(locations=[origin, destination], color=train.color(), **kw)
    fg.add_child(line)
    # fg.add_child(folium.Marker(location=[lat,lon],
    #                            popup=(folium.Popup(occ)),
    #                            icon=folium.Icon(color=color(occ),icon_color='green')))

map.add_child(fg)

map.save('maps/occupancy.html')