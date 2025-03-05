import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader
import argparse
import json
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/longvu_llama3_2")
    parser.add_argument("--model_name", type=str, default="cambrian_llama")
    parser.add_argument("--data_path", type=str, default="./EnTube")
    parser.add_argument("--json_path", type=str, default="./EnTube_preprocessing/data/EnTube_50m_test.json")

    args = parser.parse_args()


    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, args.model_name,
    )

    model.eval()
    with open(args.json_path) as f:
        data = json.load(f)
        for item in data:
            video_path = os.path.join(args.data_path, item["video"])

            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = float(vr.get_avg_fps())
            frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
            video = []
            for frame_index in frame_indices:
                img = vr[frame_index].asnumpy()
                video.append(img)
            video = np.stack(video)
            image_sizes = [video[0].shape[:2]]
            video = process_images(video, image_processor, model.config)
            video = [item.unsqueeze(0) for item in video]

            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates["llama3"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=video,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=1,
                    max_new_tokens=128,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            extract_label = extract_engagement_label(pred)
            print(f"video_path: {video_path}, pred={pred}, extract_label={extract_label}")