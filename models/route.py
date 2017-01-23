from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from .base import Base


class Route(Base):
    __tablename__ = 'routes'

    id = Column(Integer, primary_key=True)
    route_id = Column(String, unique=True)
    service_id = Column(String, unique=True)
    trip_id = Column(String, unique=True)
    stops = relationship("Stop", backref="route")

    def from_tuple(self, tuple):
        self.route_id = tuple.route_id.replace('routes:', '')
        self.service_id = str(tuple.service_id)
        self.trip_id = tuple.trip_id
