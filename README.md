-----
# HARLEM - Automated Zooming & Tracking System

\<p align="center"\>\<i\>Fig. Automated Zooming System using YOLOv5.\</i\>\</p\>

As a core component of the **HARLEM (Human Action Recognition)** project, this automated zooming and tracking system utilizes YOLOv5 to dynamically process real-time video streams. It is designed to detect, track, and centralize specific human subjects, cropping the original footage into a standardized, focus-driven resolution (`480x800`).

\<p align="center"\>
\<img src="[https://github.com/user-attachments/assets/0c9eebed-b602-41db-b623-faf726529fb2](https://github.com/user-attachments/assets/0c9eebed-b602-41db-b623-faf726529fb2)" alt="Before & After Zooming process"\>
\</p\>
\<p align="center"\>\<i\>Fig. Before & After Zooming process.\</i\>\</p\>

## ⚙️ Core Mechanism: The 2-Pass Pipeline

Unlike basic bounding-box cropping which causes severe jitter and inconsistent aspect ratios, this system implements a robust **2-Pass Processing Algorithm** coupled with mathematical smoothing to ensure a cinematic and seamless tracking experience.

### Pass 1: Global Maximum Bounding Box Extraction

In the first pass, the YOLOv5 model (`yolov5s`) scans the entire video specifically targeting the 'person' class (COCO class `0`). It evaluates all detections to find the maximum bounding box dimensions (width and height) occupied by the subject throughout the whole video. This ensures that the final cropped view will never clip the subject, even during dynamic movements.

### Pass 2: RoI Centralization & Moving Average Smoothing

During the second pass, the system actively tracks the subject and calculates the dynamic center of the Region of Interest (RoI). Instead of applying abrupt camera cuts, it uses the global maximum dimensions (found in Pass 1) plus a customizable `padding` margin to frame the subject.

To eliminate mechanical jitter caused by bounding box fluctuations between frames, a **Moving Average Filter** (`window_size = 5`) is applied to the $(x, y)$ coordinates. This acts as a virtual steady-cam, smoothing out the tracking trajectory before cropping and resizing the frame to the fixed `480x800` output resolution.

## 🚀 Key Features

  * **Zero-Jitter Tracking:** Custom Moving Average smoothing function prevents abrupt viewport jumps.
  * **Consistent Aspect Ratio:** The 2-Pass method guarantees that the output video maintains a strictly fixed dimension, crucial for downstream Human Action Recognition (HAR) models.
  * **GPU-Accelerated Inference:** Powered by PyTorch (`torch.hub`), fully utilizing CUDA environments (like Google Colab) for high-speed object detection.

## 🛠️ Quick Start

1.  Clone the YOLOv5 repository and install dependencies:

<!-- end list -->

```bash
!git clone https://github.com/ultralytics/yolov5
!cd yolov5
!pip install -r /content/yolov5/requirements.txt
```

2.  Update the `input_folder` and `output_folder` paths in the script.
3.  Run the script. Processed videos will be saved with an `Output_` prefix.

-----
