from fastapi import FastAPI
from celery import Celery
from .worker import submit_job_task

app = FastAPI()


celery_app = Celery(
    'worker',
    broker='amqp://user:password@rabbitmq',
    backend='rpc://'
)


@app.post("/submit-job/")
def submit_job():
    result = submit_job_task.delay()
    result_sync = result.get(timeout=3)
    return {"message": result_sync}
