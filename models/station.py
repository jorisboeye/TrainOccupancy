from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import relationship
from .base import Base

class Station(Base):
    __tablename__ = 'stations'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    uri = Column(String)
    code = Column(Integer)
    longitude = Column(Float)
    latitude = Column(Float)
    stops = relationship("Stop", backref="station")

    def from_tuple(self, tuple):
        if tuple is not None:
            self.uri = tuple.URI
            self.name = tuple.name
            self.longitude = float(tuple.longitude)
            self.latitude = float(tuple.latitude)
            self.code = int(self.uri.split('/')[-1])

    def coordinates(self):
        return self.latitude, self.longitude