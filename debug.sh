CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="worker" --task_index=0 --config="examples/pendulum.json" >> actor0.stdout &

CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="worker" --task_index=1 --config="examples/pendulum.json" >> actor1.stdout &

CUDA_VISIBLE_DEVICES="0" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="ps" --task_index=0 --config="examples/pendulum.json"


#CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="worker" --task_index=0 --config="examples/prosthetics.json" &
#
#CUDA_VISIBLE_DEVICES="" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="worker" --task_index=1 --config="examples/prosthetics.json" &
#
#CUDA_VISIBLE_DEVICES="0" python main.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="ps" --task_index=0 --config="examples/prosthetics.json"


#CUDA_VISIBLE_DEVICES="" python toy.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="worker" --task_index=0 &
#
#CUDA_VISIBLE_DEVICES="" python toy.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="worker" --task_index=1 &
#
#CUDA_VISIBLE_DEVICES="0" python toy.py --ps_hosts="localhost:8167" --worker_hosts="localhost:8169,localhost:8171" --job_name="ps" --task_index=0
