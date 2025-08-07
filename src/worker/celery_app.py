from celery import Celery
from ..config import settings

# Create Celery instance
app = Celery(
    'hunyuan3d_worker',
    broker=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}',
    backend=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}'
)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes per task
    task_soft_time_limit=540,  # 9 minutes warning
    worker_max_tasks_per_child=10,  # Recycle workers after 10 tasks
)

# Import tasks after app is created to avoid circular imports
from . import tasks  # noqa
