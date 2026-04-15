# CMPE401 Advanced Object Detection and Comparative Study using YOLOv26

## Introduction
In this project, I iteratively trained a YOLO26 model on the visdrone data set. I used my local resources (RTX 3070) for training and validation. Then, I compared the results from the YOLOv26 model to a YOLOv5 model, YOLOv8 model and YOLOv3u model.

### Where to find results of each iteration
In the folder runs/detect, there is a subfolder for each iteration titled "train_visdrone_yolo{version}-{iteration}". There you can find several files indicating the arguments used to train the model, the best and last weights, and results of that iteration.

The models for yolov3u are not in this github repository because they are too big (over 100MB). To train them, run train.py with WEIGHTS = "yolov3u.pt"

### How to reproduce

1. Clone this repository
2. Install the requirements outlined in requirements.txt
3. Install the visdrone data set from here: https://github.com/VisDrone/VisDrone-Dataset?tab=readme-ov-file 
3. Update the paths in visdrone.yaml to point to where you installed the above files
4. YOLO uses a different annotation scheme than what's provided in visdrone, so run convert_visdrone_to_yolo_and_check.py with the correct path to the dataset
5. Run train.py with parameters and variables modified to train and validate your desired model
6. To run inference tests, run predict.py

## Baseline model
For my first iteration (folder train_visdrone_yolo26n-01), I trained the yolov26n model using mostly default settings. The two parameters I changed were to use 50 epochs, and an image size of 512. I chose this to have a short training time for rapid iteration. 

#### Discussion
This resulted in an mAP50 of 0.25, mAP50-95 of 0.136, a precision of 0.35 and recall of 0.26. This is poor performance, though it took only 1hr 6 mins to train.

Looking at the training loss and validation loss curves, both losses were steadily declining while having close values. This means that the model was underfit and still improving. The low recall indicated that many objects were missed. The confusion matrix confirms this: the bottom row which indicates objects predicted as background has high numbers. This is particularly true for smaller objects like people, pedestrians and motors. These shortcomings are likely because of the nano model's low capacity, and the training setup. The low image size (512) greatly reduces the amount of detail that can be captured by the model. An often brought up point in dataset size. The training visdrone dataset contains 6471 static annotated images across different lighting conditions, environments and densities (https://docs.ultralytics.com/datasets/detect/visdrone/). This quality of this massive and diverse dataset is unlikely to be the bottleneck in this iteration.

## Structured Experimental Design and Iterative Model Improvement

### Results from each iteration (last model)
| Metric | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 | Iteration 5 | Iteration 6 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| mAP50 | 0.25 | 0.334 | 0.395 | 0.300 | 0.324 | 0.344 |
| mAP50-95 | 0.136 | 0.189 | 0.244 | 0.172 | 0.173 | 0.1965 |
| Precision | 0.35 | 0.445 | 0.492 | 0.401 | 0.422 | 0.445 |
| Recall | 0.26 | 0.331 | 0.382 | 0.299 | 0.326 | 0.34 |
| Training Time | 1h 6m | 2h 35m | 9h 2m | 3h 13m | 4h 32m | 3h 38m |
| Epochs | 50 | 62 | 100 | 100 | 98 | 83 |

### Iteration 2
I increased the image size from 512 to 640 to capture smaller objects (improve recall) and capture details to differentiate objects between different classes. I also increased the number of epochs to 100 and set a patience of 10 to avoid prematurely ending the training while the model was still improving. Iteration 1 seemed to still be decreasing in both: validation and training loss.

#### Evaluation and Analysis
Training stopped after 62 epochs, so the first iteration with 50 epochs was probably not too far off from its peak. The increased image size to 640 and extended training with early stopping resulted in significant improvements across all metrics. Low recall was still the most prominent issue. Looking at the confusion matrix for iteration 2, there are still many objects mistaken for background. Due to the large improvement from iteration 1 while retaining a reasonable training time, I decided to use this as my new baseline.

### Iteration 3
I retained epochs=100 with patience=10. I further increased image size to 1024 hoping that recall would improve even more. Due to the limited 8GB of VRAM in my graphics card, I was forced to reduce the batch size. I used batch size =-1, which automatically sets the optimal batch size for my GPU. In my case, it used a batch size of 3. To improve efficiency, I also set the training parameter 'rect=true'.

#### Evaluation and Analysis
The model improved in all aspects, except the training time increased to 9 hours. Training time increases with the square of image size, so I wanted to shift my efforts to other changes. Also, setting 'rect=true' in this iteration was a mistake. I should have only changed 1 thing at a time for a better controlled experimentation. Another thing worth noting is that the model didn't finish converging within 100 epochs because the 'patience' feature didn't kick in. However, looking at the loss graphs, performance was already at a plateau.

### Iteration 4
This time, I tried accelerating the learning schedule to reduce training time. I set the initial learning rate to 0.015 (default is 0.01) and final learning rate to 0.02 (default is 0.01). I also set cos_lr=True and kept 'rect=True'. I set batch size to -1, which used a batch size of 12. Otherwise this is the same as iteration 2.

#### Evaluation and Analysis
I made the mistake of changing too many things in one iteration, so it's hard to pinpoint what change is responsible for what change in performance. Nevertheless, this model is worse in all aspects. Counterintuitively, increasing the learning rate increased the training time. This is because it trained for the full 100 epochs instead of stopping early; likely due to noisier learning bringing up small improvements later in training. Regardless, the original training schedule was better.

### Iteration 5
Because I changed too many things in iteration 4, I wanted to isolate changes by retaining all settings from iteration 2, except use a batch size of -1 (it used 12). I also increased epochs to 150, but kept patience of 10 so the model can improve as much as it can.

#### Evaluation and Analysis
The difference between iterations 2 and 5 is a reduction in batch size from 16 to 12. This meant each training step was less reliable since it has seen fewer data points before updating the weights. So, iteration 5 converged to a slightly worse result and took longer because it took more epochs to see no change in validation loss.

The differences between iterations 4 and 5 are that 4 has an accelerated learning rate and 'rect=true'. Iteration 4 took over an hour less time to train than iteration 5. This is most likely because of the efficiency provided by 'rect=true'. Noisier learning is probably why iteration 4 is otherwise worse than iteration 5. 'rect=true' could also be the cause of this.

### Iteration 6
I wanted to try improving recall by doing data augmentation. I used the settings of iteration 4 with the following changes:
- degrees=20
- shear=10
- perspective=0.001

These decisions were informed by the Ultralytics guide for data augmentation (https://docs.ultralytics.com/guides/yolo-data-augmentation/#custom-albumentations-transforms-augmentations). Degrees, shear and perspective all distort images in a way that a drone would as it tilts and rotates. So, these modifications effectively expose the training to new variants of images it could run into in reality.

#### Evaluation and Analysis
This yielded better results overall compared to all other iteration with image size of 640. However, the results were not as good as in iteration 3 with image size of 1024.

## Comparing to other models
I compared YOLOv26n to YOLOv3u, YOLOv5n and YOLOv8n. For a fair comparison, I used the arguments as in iteration 2. The results are shown in the table below. I decided to include iteration 3 of YOLO26n because it is my best model.

| Metric | v26n (Itr 2) | v5n | v8n | v3u | v26n (Itr 3) |
|--------|--------------|---------|---------|---------|---|
| mAP50 | 0.334 | 0.335 | 0.344 | 0.470 | 0.395 |
| mAP50-95 | 0.189 | 0.192 | 0.198 | 0.289 | 0.244 |
| Precision | 0.445 | 0.438 | 0.446 | 0.577 | 0.492 |
| Recall | 0.331 | 0.339 | 0.344 | 0.442 | 0.382 |
| Training Time | 2h 35m | 2h 30m | 2h 8m | 8h 20m | 9h 2m |
| Epochs | 62 | 100 | 99 | 82 | 100 |
| Optimal Batch Size (RTX3070) | 12 | 11 | 11 | 4 | 3 |
| Avg inference time GPU (ms) | 18.77 | 14.94 | 13.98 | 24.78 | 19.01 |
| Avg inference time CPU (ms) | 35.16 | 34.70 | 31.67 | 394.63 | 35.41 |
| Model Size (MB) | 5.294 | 5.139 | 6.094 | 202.892 | 5.302 |

Inference times were measured using predict.py and were taken across an average of 100 random samples. The random pictures were the same for each trial (I used the same seed).

### Discussion

YOLOv3u performs far better than the other three in mAP, precision and recall, but has much higher model size, training time and inference time. Since the objective of a YOLO model is real time object detection, 394ms inference time is too slow for me to consider it my best model. So, using an image size of 640, it seems YOLOv8 is the best overall, with v26n and v5 obtaining comparable results. It may seem counterintuitive that YOLOv26n (the latest model) performs the worst. Although YOLOv26 is a newer architecture, the nano variant did not outperform YOLOv5n or YOLOv8n on VisDrone under the same training recipe. This suggests that, for this dataset, model capacity and training configuration had a larger impact than model age alone.

If we include the larger image size of 1024, YOLOv26n iteration 3 is the best model overall. One could estimate that increasing the image size for the other versions could yield better performance, but that would require more testing. So, YOLOv26n iteration 3 is my final model.

## Final evaluation

I evaluated YOLOv26n iteration 3 using the test-dev dataset. The results are as follows:

| P | R | mAP50 | mAP50-95 |
|---|---|-------|----------|
| 0.438 | 0.337 | 0.306 | 0.178 |

The confusion matrix and other stats can be found in runs/detect/evaluation-01 subfolder.

## Conclusion
In this project I compared different versions and iterations of YOLO models for object detection in the visdrone dataset. Key take-aways are:
- Higher image size matters the most for datasets with many small objects. This was proven by iterations 2 and 3 where mAP increased the most
- For this application, 'rect=True' improves training time at the cost of slightly worse results. This is proven in iterations 4 and 5.
- Data augmentation can provide small improvements (iteration 6)
- When iterating, it's important to change 1 thing at a time to isolate what caused improvements or regressions
- There is a trade-off between accuracy and inference time. This is highlighted when comparing YOLOv3u to other YOLO models.
