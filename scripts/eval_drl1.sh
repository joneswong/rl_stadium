CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 0 > drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 10 >> drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 200 >> drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 3000 >> drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 40000 >> drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 500000 >> drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 60000 >> drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 7000 >> drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 800 >> drl1_perf.txt
CUDA_VISIBLE_DEVICES="" python eval.py --config="examples/drl1.json" --checkpointDir agents/drl1/ --seed 90 >> drl1_perf.txt
