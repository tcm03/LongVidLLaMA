# Evaluation

## MVBench

Download [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) to `./data/MVBench`

```
python -m torch.distributed.launch --nproc-per-node=8 eval/eval_mvbench.py --data_path ./data/MVBench --version qwen --model_path ./checkpoints/longvu_qwen
```

## EgoSchema

Download [EgoSchema](https://github.com/egoschema/EgoSchema) to `./data/EgoSchema`

```
python -m torch.distributed.launch --nproc-per-node=8 eval/eval_egoschema.py --data_path ./data/EgoSchema --version qwen --model_path ./checkpoints/longvu_qwen
```

Then submit the result file `.csv` to [Kaggle](https://www.kaggle.com/competitions/egoschema-public/submissions)


## MLVU

Download [MVLU](https://huggingface.co/datasets/MLVU/MVLU) to `./data/MVLU`

```
python -m torch.distributed.launch --nproc-per-node=8 eval/eval_mlvu.py --data_path ./data/MVLU --version qwen --model_path ./checkpoints/longvu_qwen
```

## VideoMME

Download [VideoMME](https://github.com/BradyFU/Video-MME?tab=readme-ov-file#-dataset) to `./data/VideoMME`

```
python -m torch.distributed.launch --nproc-per-node=8 eval/eval_videomme.py --data_path ./data/VideoMME --version qwen --model_path ./checkpoints/longvu_qwen
```