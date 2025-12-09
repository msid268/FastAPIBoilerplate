from contextvars import ContextVar


current_job_id: ContextVar[str | None ] = ContextVar("current_job_id", default=None)


def set_job_id(job_id: str):
    return current_job_id.set(job_id)

def get_job_id() -> str | None:
    return current_job_id.get()


def reset_job_id(token):
    current_job_id.reset(token)
    
    
    