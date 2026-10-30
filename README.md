# Fabric Defect Detection Using Deep Learning
**Live demo:** https://fabric-defect-detection-7wcedp6ue77xbssadeacpt.streamlit.app
B.Tech (CSE – AI & ML) major project, Pimpri Chinchwad University, Pune.

Two pre-trained YOLOv8 models (YOLOv8n and YOLOv8s) were fine-tuned to detect ten fabric defect classes:
Hole, Knot, Line, Open Seam, Pull Thread, Scratch, Seam Puckering, Slub, Stain and Tear.
A Streamlit web app lets you upload a fabric image and see the detected defects, their classes and confidence scores.

## Results (test set, 324 images)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8n | 0.761 | 0.720 | 0.796 | 0.538 |
| YOLOv8s | 0.801 | 0.777 | 0.810 | 0.570 |

Both models were trained for 50 epochs at image size 640 (batch 16) on Google Colab (Tesla T4 GPU).

## Run the app

```
pip install -r requirements.txt
streamlit run app.py
```

Keep `best_v8s.pt` and `best_v8n.pt` in the same folder as `app.py`.

## Dataset

Roboflow Universe: [new_dataset_10_08_25 (version 5)](https://universe.roboflow.com/defect-c8fe4/new_dataset_10_08_25-knx4h/dataset/5),
published by the user "Defect", licensed under CC BY 4.0. 3,234 images split 70/20/10 into train, validation and test.

## Team

Aaditya Kumar Dixit, Abhishek Kumar, Anurag Ubarhande, Abhijeet Arya
Guide: Leena Waghulde
