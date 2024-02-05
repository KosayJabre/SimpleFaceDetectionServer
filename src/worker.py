from celery import Celery


app = Celery(
    'worker',
    broker='amqp://user:password@rabbitmq',  
    backend='rpc://'
)


@app.task(name='app.worker.submit_job_task')
def submit_job_task():
    return "Job completed successfully"