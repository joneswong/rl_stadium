CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 0 > contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 10 >> contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 200 >> contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 3000 >> contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 40000 >> contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 500000 >> contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 60000 >> contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 7000 >> contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 800 >> contd_v4dot1b_perf.txt
CUDA_VISIBLE_DEVICES="0" python eval.py --config="examples/contd_r2_v4dot1b.json" --checkpointDir contd_v4dot1b/ --seed 90 >> contd_v4dot1b_perf.txt
