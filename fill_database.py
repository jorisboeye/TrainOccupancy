from models import Base, Route, Station, Stop
from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker
import os
import re
import pandas as pd

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

# add trips
trips_csv_path = '/home/joris/PycharmProjects/Kaggle/TrainOccupancy/data/nmbs-2016-09/trips.txt'
trips_df = pd.read_csv(trips_csv_path)
for row in trips_df.itertuples():
    if not db.query(exists().where(Route.trip_id==row.trip_id)).scalar():
        route = Route()
        route.from_tuple(row)
        db.add(route)
db.commit()

# add stations
station_csv_path = '/home/joris/PycharmProjects/Kaggle/TrainOccupancy/data/stations.csv'
stations_df = pd.read_csv(station_csv_path)
for row in stations_df.itertuples():
    if not db.query(exists().where(Station.uri==row.URI)).scalar():
        station = Station()
        station.from_tuple(row)
        db.add(station)
db.commit()

# add stops
stop_csv_path = '/home/joris/PycharmProjects/Kaggle/TrainOccupancy/data/nmbs-2016-09/stop_times.txt'
stops_df = pd.read_csv(stop_csv_path)
for row in stops_df.itertuples():
    try:
        stop = Stop()
        route = db.query(Route).filter_by(trip_id=row.trip_id).first()
        station_name = int(re.sub(':0', '', row.stop_id))
        station = db.query(Station).filter_by(code=station_name).first()
        stop.route = route
        stop.station = station
        stop.sequence = int(row.stop_sequence)
        db.add(stop)
    except Exception:
        print('stop exception')
        continue
db.commit()
