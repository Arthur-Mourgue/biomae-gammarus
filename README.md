# Biomae Gammarus Sorting

<table width="100%">
  <tr>
    <td width="60%" valign="middle" style="padding-right: 15px;">
      <b>Automated Gammarus sorting for water quality biomonitoring at <a href="https://www.biomae.fr/">Biomae</a></b>
      <br><br>
      The company Biomae uses gammarus, which are small freshwater shrimp, to assess water pollution levels. For these tests to work, the individuals must be separated according to their sex or if they are in a couple.
      <br><br>
    </td>
    <td width="40%" align="center" valign="middle">
      <img src="docs/pipeline.png" height="550">
    </td>
  </tr>
</table>

## The Problem

<table width="100%">
  <tr>
    <td width="60%" valign="middle" style="padding-right: 15px;">
      Today, sorting is done by hand under a microscope. It is a long, eye-straining task that is prone to errors. A human can only sort about 1,000 specimens per hour.<br><br>
      Automation is complicated because of the observation conditions: to keep the gammarus alive and avoid stressing them, the lighting must remain very low. <br><br>
      The gammarus move quickly inside their transparent tubes, which creates a lot of motion blur in the videos.
    </td>
    <td width="40%" align="center" valign="middle">
      <img src="docs/manual-sorting.png" height="250">
    </td>
  </tr>
</table>

## The Solution

<table width="100%">
  <tr>
    <td width="60%" valign="middle" style="padding-right: 15px;">
      Our automated pipeline handles these complex conditions and significantly speeds up the process to completely replace manual sorting by operators.
    </td>
    <td width="40%" align="center" valign="middle">
      <img src="docs/automated-sorting-machine.png" height="200">
    </td>
  </tr>
  <tr>
    <td width="60%" valign="middle" style="padding-right: 15px;">
      <b>1. Detection and Cropping</b><br><br>
      The computer starts by looking at the overall image and locates the gammarus. It then crops this small area to remove all the unnecessary background of the tube. This allows the system to focus solely on the gammarus.
    </td>
    <td width="40%" align="center" valign="middle">
      <img src="docs/raw-vs-cropped.png" height="120">
    </td>
  </tr>
  <tr>
    <td width="60%" valign="middle" style="padding-right: 15px;">
      <b>2. Deblurring</b><br><br>
      We created a tool capable of making blurry images sharper. 
      However, we noticed that correcting all the images degraded the results. Our trick is therefore to only deblur the images where the computer has a genuine doubt.
    </td>
    <td width="40%" align="center" valign="middle">
      <img src="docs/unblurr.png" height="150">
    </td>
  </tr>
  <tr>
    <td width="60%" valign="middle" style="padding-right: 15px;">
      <b>3. Classification and Decision</b><br><br>
      Once the image is clean, the model analyzes the shape of the shrimp and instantly classifies it into four categories: male, female, couple, or undetermined.<br><br>
    </td>
    <td width="40%" align="center" valign="middle">
      <img src="docs/classification-classes.png" height="180">
    </td>
  </tr>
</table>

## Results

Our system achieves good results with an average precision of 92% using the F1 score. It perfectly recognizes couples and performs very well at identifying ambiguous cases and females. 

| Class | Precision | Recall | F1 Overall Success |
|-------|-----------|--------|-----|
| couple | 1.00 | 1.00 | 1.00 |
| female | 1.00 | 0.85 | 0.92 |
| undetermined | 0.92 | 0.92 | 0.92 |
| male | 0.81 | 0.91 | 0.86 |

## Technologies Used

* **Detection:** YOLOv11n 
* **Classification:** MobileNetV3 
* **Deblurring:** NAFNet
* **Measurement:** DeepLabCut 
* **Various Tools:** OpenCV, Pillow, PyTorch

## Installation and Execution

Install the prerequisites. Python 3.10 and a graphics card with CUDA are recommended. 

```bash
pip install -e . && pip install -r requirements.txt

```

Run the Python program to analyze an image:

```python
from biomae.pipeline import GammarusPipeline
from biomae.paths import checkpoint_path, data_path

pipeline = GammarusPipeline(
    yolo_weights=checkpoint_path("yolov11n_best.pt"),
    clf_weights=checkpoint_path("mobilenet_v3_v4.pth"),
    clf_meta=checkpoint_path("model_meta.json")
)

results = pipeline.process_image(data_path("dataset/images/image_00049.png"))

```

## Current Limitations

* Males and females look very similar, which remains the most difficult part for the machine.
* The couple class has a perfect score of 100%, but this figure should be taken with a grain of salt as we had very few examples of couples to run the test.
* This computer program still needs to be integrated directly into the actual physical sorting machine.
