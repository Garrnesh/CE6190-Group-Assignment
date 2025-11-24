import t2v_metrics
import json
import os
import tqdm
from PIL import Image
import numpy as np

index_json_path = '/notebooks/t2v_metrics/infinity_index.json'
img_path = '/notebooks/t2v_metrics/imagesInf1024/notebooks/Infinity/images'
score_output_path = '/notebooks/t2v_metrics/1024_infinity_score.json'

with open(index_json_path, 'r') as file:
    data = json.load(file)

new_data = []
print('Load model', 'clip-flant5-xl')
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl')

print('Starting Generation')
for i, d in enumerate(data):
    print(f'Processing {i+1}/{len(data)}')
    prompt = d["prompt"]
    cur_img = d["image_path"]
    full_path = os.path.join(img_path, cur_img)
    
    img = Image.open(full_path)
    img_np = np.array(img.convert("L"))
    black = (img_np==0).sum()
    isBlack = bool(black / img_np.size >= 0.98)
    
    score = clip_flant5_score(images=[full_path], texts=[prompt])
    score_val = score.item()
    d['score'] = score_val
    d['blank'] = isBlack
    new_data.append(d)

print('Ended Generation')
with open(score_output_path, 'w') as file:
    json.dump(new_data, file, indent=4)