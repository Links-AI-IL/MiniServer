import datetime
import pytz
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ConvoChunk(Base):
    __tablename__ = "convo_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String, index=True)

    ts = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(pytz.timezone("Asia/Jerusalem")),
        index=True
    )

    role = Column(String)
    text = Column(Text)

# class ConvoChunk(Base):
#     __tablename__ = "convo_chunks"

#     id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
#     device_id: Mapped[str] = mapped_column(String, index=True)

#     ts: Mapped[datetime.datetime] = mapped_column(
#         DateTime(timezone=True),
#         default=lambda: datetime.datetime.now(pytz.timezone("Asia/Jerusalem")),
#         index=True
#     )

#     role: Mapped[str] = mapped_column(String)
#     text: Mapped[str] = mapped_column(Text)
