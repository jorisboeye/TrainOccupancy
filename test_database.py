from models import Base, Route, Station, Stop
from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker
import os
import pandas as pd
import folium

# determine the db location
basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'trains.db')

# create the db engine
engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)

# create the db schema
Base.metadata.create_all(engine)

# create the db session
Session = sessionmaker(bind=engine)

# create the db session connection
db = Session()

map_route = folium.Map(location=[50.85, 4.35], zoom_start=9)
kw = dict(opacity=1.0, weight=4)
locations = []

routes = db.query(Route).filter_by(route_id='S11755').all()
for route in routes:
    print('*'*80)
    print(route.trip_id)
    fg=folium.FeatureGroup(name=route.trip_id)
    for stop in route.stops:
        print(stop.sequence, stop.station.name, stop.station.coordinates())
        locations.append(stop.station.coordinates())
        line = folium.PolyLine(locations=locations, color='blue', **kw)

fg.add_child(line)

map_route.add_child(fg)
map_route.add_children(folium.map.LayerControl())
map_route.save('maps/occupancy_route.html')
