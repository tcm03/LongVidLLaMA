MODEL_PATH="./checkpoints/longvu_llama3_2"
MODEL_NAME="cambrian_llama"
DATA_PATH="/root/hcmus/EnTube"
JSON_PATH="/root/hcmus/EnTube_preprocessing/data/EnTube_50m_test.json"

python longvu/inference.py \
--model_path $MODEL_PATH \
--model_name $MODEL_NAME \
--data_path $DATA_PATH \
--json_path $JSON_PATH