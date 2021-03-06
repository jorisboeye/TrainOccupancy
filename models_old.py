from dateutil.parser import parse


class Train:
    def __init__(self, json_line=None, stations=None):
        if json_line is not None and stations is not None:
            try:
                self.query_time = json_line['querytime']
                self.parsed_query_time = parse(self.query_time)
                midnight = self.parsed_query_time.replace(hour=0, minute=0, second=0, microsecond=0)
                self.seconds_since_midnight = (self.parsed_query_time - midnight).seconds
                f_uri = json_line['post']['from']
                self.origin = next((x for x in stations if x.uri == f_uri), None)
                d_uri = json_line['post']['to']
                self.destination = next((x for x in stations if x.uri == d_uri), None)
                self.occupancy = json_line['post']['occupancy'].split('/')[-1]
                self.vehicle = json_line['post']['vehicle'].split('/')[-1]
            except Exception as e:
                print(e)
                self.load_success = False
            else:
                self.load_success = True

    @staticmethod
    def week_day_mapping():
        return {0: 'Monday',
                1: 'Tuesday',
                2: 'Wednesday',
                3: 'Thursday',
                4: 'Friday',
                5: 'Saturday',
                6: 'Sunday'}

    def week_day(self):
        return self.week_day_mapping()[self.parsed_query_time.weekday()]

    def color(self):
        color_mapping = {'low': 'green', 'medium': 'orange', 'high': 'red'}
        if self.occupancy in color_mapping.keys():
            col = color_mapping[self.occupancy]
        else:
            col = 'blue'
        return col


class Station:
    def __init__(self, station_dict=None):
        if station_dict is not None:
            self.uri = station_dict['URI']
            self.name = station_dict['name']
            self.longitude = float(station_dict['longitude'])
            self.latitude = float(station_dict['latitude'])

    def coordinates(self):
        return self.latitude, self.longitude


class Trip:
    def __init__(self):
        self.stops = []

    def add_stop(self, stop):
        self.stops.append(stop)


class Stop:
    def __init__(self, station):
        self.station = station
