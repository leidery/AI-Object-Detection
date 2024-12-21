import os
import cv2
import matplotlib.pyplot as plt
import neptune
from ultralytics import YOLO
from dotenv import load_dotenv


def init_run(tags=None):
    
    load_dotenv()  

    api_token = os.getenv("NEPTUNE_API_TOKEN")
    run = neptune.init_run(
        project="leidery/Deteccao",
        api_token=api_token,
        tags=tags,
    )
    return run
MODEL_NAME = "yolov8n.pt"


model = YOLO(MODEL_NAME) # carregando o modelo


run = init_run(["yolo-detection"]) # iniciando o neptune run

run["model/task"] = "Object Detection"
run["model/name"] = MODEL_NAME

img1_path = "images/walk.jpg"
img2_path = "images/vases.jpg"


if not os.path.exists(img1_path):
    raise FileNotFoundError(f"Imagem {img1_path} não encontrada.")
if not os.path.exists(img2_path):                                           # verificando se as imagens existem
    raise FileNotFoundError(f"Imagem {img2_path} não encontrada.")


results = model(img1_path)
results = model(img2_path)

fig, ax = plt.subplots(figsize=(12, 8))

ax.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
ax.axis("off")

output_image_path = "output_image.png"
fig.savefig(output_image_path)

run["predictions/sample"].upload(output_image_path)

plt.show()

run.stop()
