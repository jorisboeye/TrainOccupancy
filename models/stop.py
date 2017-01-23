from sqlalchemy import Column, Integer, String, ForeignKey
from .base import Base

class Stop(Base):
    __tablename__ = 'stops'

    id = Column(Integer, primary_key=True)
    station_id = Column(Integer, ForeignKey('stations.id'))
    route_id = Column(Integer, ForeignKey('routes.id'))
    sequence = Column(Integer)
