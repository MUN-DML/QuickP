# sudo apt update
# sudo apt install rabbitmq-server
# sudo systemctl enable rabbitmq-server
# sudo systemctl start rabbitmq-server
import os
import sys

from celery import Celery

app = Celery('tasks', broker='pyamqp://guest@192.168.0.66//', backend='rpc://')

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
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(script_dir)
    sys.path.append(project_root)

    from optimizer.device_topo.intra_node_topo_parallel import get_intra_node_topo

    result = get_intra_node_topo()
    return result


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
