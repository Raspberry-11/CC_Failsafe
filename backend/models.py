from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(100))
    email         = Column(String(100), unique=True, index=True)
    password_hash = Column(String(255))
    role          = Column(String(20), default="faculty")
    created_at    = Column(DateTime, server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"))
    student_name  = Column(String(100))
    subject       = Column(String(20))
    features      = Column(Text)
    risk_prob     = Column(Float)
    risk_level    = Column(String(10))
    interventions = Column(Text)
    shap_top5     = Column(Text)
    created_at    = Column(DateTime, server_default=func.now())
