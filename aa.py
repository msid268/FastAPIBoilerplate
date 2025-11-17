from pathlib import Path
print(Path(__file__).resolve().parent)
from app.db.base import Base
target_metadata = Base.metadata
print(Base.metadata.tables.keys())