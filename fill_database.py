from models import Base, Route, Station, Stop
from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker
import os
import re
import pandas as pd
from dateutil.parser import parse

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
errors = 0
for row in stops_df.itertuples():
    try:
        route = db.query(Route).filter_by(trip_id=row.trip_id).first()
        station_name = int(re.sub(r':\w+', '', row.stop_id))
        station = db.query(Station).filter_by(code=station_name).first()
    except Exception as err:
        print('*'*80)
        print('Handling run-time error:', err)
        errors += 1
        print('stop exception nr.:' + str(errors))
        print('trip_id:', row.stop_id)
        continue
    else:
        if not db.query(exists().where(Stop.route==route and Stop.station==station)).scalar():
            stop = Stop()
            stop.route = route
            stop.station = station
            stop.sequence = int(row.stop_sequence)
            arrival_parsed = parse(row.arrival_time)
            departure_parsed = parse(row.departure_time)
            midnight = arrival_parsed.replace(hour=0, minute=0, second=0, microsecond=0)
            stop.arrival = (arrival_parsed - midnight).seconds
            stop.departure = (departure_parsed - midnight).seconds
            db.add(stop)
db.commit()
