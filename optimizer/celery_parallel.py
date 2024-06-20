# sudo apt update
# sudo apt install rabbitmq-server
# sudo systemctl enable rabbitmq-server
# sudo systemctl start rabbitmq-server
import os
import sys
import time

from celery import Celery
# celery -A celery_parallel worker --loglevel=INFO
# backend: RPC (RabbitMQ/AMQP),
# broker specifying the URL of the message broker you want to use. Here we are using RabbitMQ (also the default option).
app = Celery('celery_parallel', broker='pyamqp://guest@localhost//', backend='rpc://')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)


@app.task
def run_function(node_id):
    import torch

    return f"fuck, {node_id}"


def send_task(node_id):
    result = run_function.delay(node_id)
    return result


if __name__ == "__main__":
    nodes = ['192.168.0.66', '192.168.0.6']
    results = []

    for node in nodes:
        result = send_task(node)
        results.append(result)
        print(f'Task sent for node {node}. Task ID: {result.id}')

    # Wait for all tasks to complete and collect results
    for result in results:
        while not result.ready():
            time.sleep(1)
        print(f'Result for task {result.id}: {result.get()}')
