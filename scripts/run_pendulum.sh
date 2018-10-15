# x2
#CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="worker" --task_index=0 --config="examples/pendulum.json" > actor0.stdout &
#
#CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="worker" --task_index=1 --config="examples/pendulum.json" > actor1.stdout &
#
#CUDA_VISIBLE_DEVICES="0" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="ps" --task_index=0 --config="examples/pendulum.json"


# x4
CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171,localhost:8173,localhost:8175" --job_name="worker" --task_index=0 --config="examples/pendulum.json" > actor0.stdout &

CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171,localhost:8173,localhost:8175" --job_name="worker" --task_index=1 --config="examples/pendulum.json" > actor1.stdout &

CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171,localhost:8173,localhost:8175" --job_name="worker" --task_index=2 --config="examples/pendulum.json" > actor2.stdout &

CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171,localhost:8173,localhost:8175" --job_name="worker" --task_index=3 --config="examples/pendulum.json" > actor3.stdout &

CUDA_VISIBLE_DEVICES="0" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171,localhost:8173,localhost:8175" --job_name="ps" --task_index=0 --config="examples/pendulum.json"
