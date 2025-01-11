import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from decord import VideoReader, cpu
from datasets import DatasetDict, Dataset
from transformers import Trainer, TrainingArguments
from longvu.builder import load_pretrained_model
from longvu.mm_datautils import process_images
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

parser = HfArgumentParser((TrainingArguments,))
parser.add_argument("--input_model_filename", type=str, required=True)
parser.add_argument("--output_model_filename", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--num_labels", type=int, default=3)
parser.add_argument("--model_max_length", type=int, default=8192)
parser.add_argument("--freeze_backbone", type=bool, default=False)  # New argument
args, training_args = parser.parse_args_into_dataclasses()

# Step 1: Load EnTube.csv and prepare video paths
def load_entube_csv(csv_path, video_root):
    """
    Load the EnTube.csv file and prepare paths to video files.

    Args:
        csv_path (str): Path to EnTube.csv.
        video_root (str): Root directory containing video subfolders (0/, 1/, 2/).

    Returns:
        pandas.DataFrame: DataFrame with video paths and engagement labels.
    """
    df = pd.read_csv(csv_path)
    df['video_path'] = df.apply(lambda row: os.path.join(video_root, str(row['engagement_rate_label']), f"{row['video_id']}.mp4"), axis=1)
    return df

# Step 2: Sample frames at 1 FPS and preprocess them
def sample_and_preprocess_frames(df, image_processor, model_config, fps=1):
    """
    Sample frames from videos at 1 FPS and preprocess them for the model.

    Args:
        df (pandas.DataFrame): DataFrame with video paths and labels.
        image_processor: Pre-trained image processor from LongVU.
        model_config: Model configuration for image preprocessing.
        fps (int): Frames per second to sample.

    Returns:
        list, list: Preprocessed frames and corresponding labels.
    """
    videos = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row['video_path']
        label = row['engagement_rate_label']

        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            video_fps = float(vr.get_avg_fps())
            frame_indices = np.array([i for i in range(0, len(vr), round(video_fps / fps))])
            video_frames = [vr[frame_index].asnumpy() for frame_index in frame_indices]

            # Preprocess frames using the LongVU image processor
            processed_frames = process_images(video_frames, image_processor, model_config)
            videos.append(processed_frames)
            labels.append(label)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue

    return videos, labels

# Step 3: Load dataset
csv_path = "/content/drive/MyDrive/Thesis/EnTube/EnTube.csv"
video_root = "/content/drive/MyDrive/Thesis/EnTube"

print(f'@tcm: In entube_finetune.py: load dataset')
# Load CSV and prepare video paths
df = load_entube_csv(csv_path, video_root)

# Load pretrained model, tokenizer, and image processor
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.input_model_filename, None, "cambrian_qwen"
)

# Freeze the backbone if specified
if args.freeze_backbone:
    print("Freezing backbone. Only fine-tuning the classification head.")
    for param in model.get_model().parameters():  # Freeze all parameters except cls_head
        param.requires_grad = False

# Sample frames and preprocess them
videos, labels = sample_and_preprocess_frames(df, image_processor, model.config)

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({"input_ids": videos, "labels": labels})
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir=args.output_model_filename,                # From bash argument
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,                     # From bash argument
    per_device_train_batch_size=args.per_device_train_batch_size,  # From bash argument
    per_device_eval_batch_size=args.per_device_eval_batch_size,    # From bash argument
    num_train_epochs=args.num_train_epochs,               # From bash argument
    weight_decay=0.01,
    logging_dir="./logs",                                 # Static, but can be updated
    logging_steps=5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="tensorboard",
    fp16=True,
)

# Step 5: Define metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the final model
if not os.path.exists("./checkpoints/entube_longvu/"):
    os.makedirs("./checkpoints/entube_longvu/")
trainer.save_model("./checkpoints/entube_longvu/")
