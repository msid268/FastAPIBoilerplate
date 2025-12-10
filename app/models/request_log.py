from __future__ import annotations
from typing import List, Optional
from sqlalchemy import Integer, Text, Float, Index, ForeignKey,  Enum as SAEnum, String, DateTime
from sqlalchemy.dialects.mssql import NVARCHAR, DATETIME2
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from datetime import datetime

from app.db.base import Base


class ActionLog(Base):
    __tablename__ = "action"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # KEY PART: proper FK to request.id
    request_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("request.id", ondelete="CASCADE"),
        nullable=True,
    )
    job_log_id: Mapped[int] = mapped_column(Integer, ForeignKey("job_logs.id", ondelete="CASCADE"), nullable=True)
    
    action_type: Mapped[Optional[str]] = mapped_column(NVARCHAR(36), index=True)
    action_name: Mapped[str] = mapped_column(NVARCHAR(200), nullable=False)

    module_name: Mapped[Optional[str]] = mapped_column(NVARCHAR(200))
    function_name: Mapped[Optional[str]] = mapped_column(NVARCHAR(200))
    line_number: Mapped[Optional[int]] = mapped_column(Integer)

    input_params: Mapped[Optional[str]] = mapped_column(Text)
    output_results: Mapped[Optional[str]] = mapped_column(Text)

    start_time: Mapped[Optional[str]] = mapped_column(DATETIME2, nullable=False, server_default=func.sysdatetime())
    end_time: Mapped[Optional[str]] = mapped_column(DATETIME2)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float)

    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text)
    is_error: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)

    llm_provider: Mapped[Optional[str]] = mapped_column(NVARCHAR(50))
    llm_model: Mapped[Optional[str]] = mapped_column(NVARCHAR(100))
    llm_prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    llm_completion_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    llm_total_tokens: Mapped[Optional[int]] = mapped_column(Integer)

    # KEY PART: back_populates matches RequestLog.action_logs
    request: Mapped["RequestLog"] = relationship(back_populates="action_logs")

    __table_args__ = (
        Index("idx_action_logs_start_time", "start_time"),
        Index("idx_action_logs_composite", "request_id", "action_type", "is_error"),
    )

class RequestLog(Base):
    __tablename__ = "request"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(NVARCHAR(36), unique=True, nullable=False, index=True)

    method: Mapped[str] = mapped_column(NVARCHAR(10), nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    query_params: Mapped[Optional[str]] = mapped_column(Text)
    headers: Mapped[Optional[str]] = mapped_column(Text)
    body: Mapped[Optional[str]] = mapped_column(Text)

    server_name: Mapped[Optional[str]] = mapped_column(NVARCHAR(50))
    api_version: Mapped[Optional[str]] = mapped_column(NVARCHAR(10))

    status_code: Mapped[Optional[int]] = mapped_column(Integer)
    response_body: Mapped[Optional[str]] = mapped_column(Text)

    start_time: Mapped[Optional[str]] = mapped_column(DATETIME2, nullable=False, server_default=func.sysdatetime())
    end_time: Mapped[Optional[str]] = mapped_column(DATETIME2)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float)

    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text)
    is_error: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)

    # KEY PART: back_populates must match the child side name
    action_logs: Mapped[List["ActionLog"]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )
    jobs: Mapped[List["JobLog"]] = relationship(
        back_populates="request_log",
        cascade="all, delete-orphan",
    )
    __table_args__ = (
        Index("idx_request_logs_start_time", "start_time"),
        Index("idx_request_logs_composite", "method", "is_error"),
    )

class JobStatusEnum(str, PyEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class JobLog(Base):
    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # what the client sees
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    # which request created this job, if any
    # NOTE: adjust "request.id" to match RequestLog.__tablename__
    request_log_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("request.id"),  # or "request_logs.id" if that's your real table name
        nullable=True,
    )
    request_log: Mapped[Optional["RequestLog"]] = relationship(
        back_populates="jobs"
    )

    status: Mapped[JobStatusEnum] = mapped_column(
        SAEnum(JobStatusEnum),
        default=JobStatusEnum.queued,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # optional: payload + results
    input_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)