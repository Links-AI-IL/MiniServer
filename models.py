import datetime
import pytz
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, func, Integer

class Base(DeclarativeBase):
    """בסיס ORM של SQLAlchemy 2.x"""
    pass

class ConvoChunk(Base):
    __tablename__ = "convo_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String, index=True)

    ts: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(pytz.timezone("Asia/Jerusalem")),
        index=True
    )

    role: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
