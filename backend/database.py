from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

# Default: SQLite — zero config, file-based, ships with Python.
# To use MySQL  (XAMPP): DATABASE_URL=mysql+pymysql://root:@127.0.0.1:3306/failsafe
# To use PostgreSQL:     DATABASE_URL=postgresql://user:pass@127.0.0.1:5432/failsafe
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./failsafe.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
