import os
import json

video_dir = "E:\estudy\VLM-Finetuning-PSL\cropped_videos"
jsonl_path = "E:\estudy\VLM-Finetuning-PSL\dataset.jsonl"

with open(jsonl_path, "w") as out_file:
    for idx, filename in enumerate(os.listdir(video_dir)):
        if filename.endswith(".mp4"):
            word = os.path.splitext(filename)[0]  # title is the label
            entry = {
                "id": idx,
                "video": filename,
                "conversations": [
                    {"from": "human", "value": "<video>\nWhat is the sign being shown in this video?"},
                    {"from": "gpt", "value": word}
                ]
            }
            out_file.write(json.dumps(entry) + "\n")
