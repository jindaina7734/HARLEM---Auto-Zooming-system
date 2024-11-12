# HARLEM - Automated Zooming System

![Auto zooming (3)](https://github.com/user-attachments/assets/c8d8cf8c-c26d-4bcd-889b-9363df311026)

<p align="center">Fig. Automated Zooming System using YOLOv5.</p>

As a part of HARLEM - Human Recognition Action Recognition project. This is an automated zooming system using YOLOv5 for real-time video, which detects and tracks specified objects to adjust the view dynamically. Key features include Region of Interest (RoI) Centralization to keep objects centered, adaptive zoom based on object size and position, and smooth transitions to avoid abrupt changes.


<p align="center">
  <img src="https://github.com/user-attachments/assets/0c9eebed-b602-41db-b623-faf726529fb2" alt="Untitled design (1)">
</p>
<p align="center">Fig. Before & After Zooming process.</p>



The auto-zooming mechanism developed in this study functions by dynamically adjusting the view based on the objects detected by the YOLOv5 model in real-time video streams. The process begins with the detection of an object of interest, such as a particular class or region specified by the user. Once the YOLOv5 model identifies and localizes the object by predicting a bounding box around it, the auto-zooming system extracts the bounding box coordinates to initiate zooming adjustments.

The central focus of the auto-zooming mechanism is the Region of Interest (RoI) Centralization. This process ensures that the detected object is not only located in the frame but also positioned at the optimal focal point for clear visibility. The system automatically repositions the camera or viewport to bring the object to the center of the frame, allowing it to dominate the visual field. This re-centering is especially critical when the detected object is near the frame’s periphery, ensuring that the object is adequately captured in the visual output.

The adaptive zooming feature plays a significant role in determining how much to zoom in or out based on the size and position of the object within the frame. If the object is relatively small and positioned far from the center, the system zooms in to increase the object's prominence. Conversely, if the object is already large or well-centered, the system may adjust the zoom minimally to maintain balance. This dynamic adjustment allows for consistent object visibility without overwhelming the frame or losing focus on critical areas.

To ensure a seamless viewing experience, the system incorporates smooth zoom transitions. Abrupt changes in zoom level can create a jarring effect, especially in real-time video applications. The smoothing function calculates incremental adjustments between zoom levels, allowing the transition to occur gradually and fluidly. This prevents sudden jumps in magnification or position, ensuring that the auto-zoom mechanism maintains visual clarity and does not disrupt the user’s experience.

In terms of performance, the auto-zooming mechanism is optimized for real-time applications through several strategies. Frame skipping is employed to reduce the computational burden by applying the auto-zoom logic only to select frames, thus avoiding the need to process every single frame in a high-frame-rate video. This optimization ensures that the system remains responsive without sacrificing accuracy in detecting and zooming in on objects. Additionally, GPU acceleration is leveraged to speed up the entire detection and zooming pipeline, utilizing high-performance computing resources to handle the heavy computational load associated with real-time object detection and dynamic zooming.

