# SAWEC

This is the implementation of the paper [SAWEC: Sensing-Assisted Wireless Edge Computing](https://arxiv.org/abs/2402.10021) and [
Integrated Sensing and Communication for Efficient Edge Computing](https://ieeexplore.ieee.org/document/10770523). The repository shares both the datasets and the source code of **SAWEC.**

If you find the project useful and you use this code, please cite our papers:

```
@inproceedings{haque2024integrated,
  title={Integrated Sensing and Communication for Efficient Edge Computing},
  author={Haque, Khandaker Foysal and Meneghello, Francesca and Restuccia, Francesco},
  booktitle={2024 20th International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob)},
  pages={611--614},
  year={2024},
  organization={IEEE}
}

```

and 

```
@article{haque2024sawec,
  title={SAWEC: Sensing-Assisted Wireless Edge Computing},
  author={Haque, Khandaker Foysal and Meneghello, Francesca and Karim, Md Ebtidaul and Restuccia, Francesco},
  journal={arXiv preprint arXiv:2402.10021},
  year={2024}
}

```
Firstly clone the repository with ```git clone git@github.com:kfoysalhaque/SAWEC.git```

## Facial Recognition Test (Preliminary Test)

(I) Go into the test directory with ```cd Facial_Recognition``` and download the [trained_model](https://drive.google.com/file/d/1VOARVhf4gsvTRfvgbEakc8K9oRXo_MWz/view?usp=drive_link) and the [dataset](https://drive.google.com/file/d/16fZ3t14EEy2LpYMHbW164ZTKkdRA4Q27/view?usp=drive_link) within the directory. </br>
</br>

(II) Unzip the dataset and delete the zip file with ``` sudo unzip Test_Data.zip && sudo rm -rf Test_Data.zip. ```</br>
</br>

(III) Create different compression quality dataset
```
python compression.py <'Compression Quality'>  && python preprocess_compressed.py <'Compression Quality'>
```
Example: ``` python compression.py 10  && python preprocess_compressed.py 10 ```
**compression quality of the image ranges from 0-100, higher the value, the better the image**
</br>

(IV) Create different reshaped (downsized) dataset
```
python preprocess_image_downsize.py <'image_resolution'>
```
Example: ``` python preprocess_image_downsize.py 256 ```
</br>

(V) Train the model with 1024x1024 image shapes: ``` python main.py ```
</br>

(VI) Evaluate the performance with different image compression and image downsize ratio

```
python evaluate.py <'test_name'> <'compression_quality_or_image_resolution'>
```

Example: 
``` 
python evaluate.py compression 10
```

```
python evaluate.py downsize 256
```
**test name- "compression" / "downsize" and compression quality==> (0, 10, 25, 50, 75, 100), the higher the value, the better the image quality. For the image downsize test, input image resolution==> (1024/512/256/128/64/32/16)**

## Sensing Assisted Wireless Edge Computing

(I) Please follow our [mD-Track](https://github.com/Restuccia-Group/SAWEC-Localization-mD-Track) implementation for localization. However, you might perform localization with any other algorithms, resulting in different performances.
</br>
(II) Please [download](https://drive.google.com/file/d/1loY_GjhkcU7ue2BUwc4tLfW8JHo7pgdf/view?usp=drive_link) and unzip the localization information and the corresponding 10K frames.
```
sudo unzip Stitched_Video.zip && sudo rm -rf Stitched_Video.zip 
```
</br>
(III) Extract the partial frames (10K resolution) using the localization information

```
cd Partial_Frame
```

```
python main.py <'environment'>
```

Example: python main.py Classroom1

**Name of the environment could be, "Classroom1" or "Anechoic1"**

The partial frames will be in the newly created directory "Partial_Frames_10K". 

You can downsize or compress the partial Frames by executing ``` python downsize_or_compress.py  <'environment'> <'conversion_type'> <'ratio'> ```

Example: ```python downsize_or_compress.py  Anechoic1 downsize 256```

</br>

(IV) Move into the directory **Segmentation_Performance** for evaluating object detection and segmentation task

```
cd ../Segmentation_Performance
```
Then fetch the partial frames with ```python fetch_images.py``` 
**Please set the source and target directory path accordingly**

</br>

(V) download the YOLOV8 models from [here](https://drive.google.com/file/d/1TAqfFLLMEJvHKlh3gspC6xf0e39Dk8Bx/view?usp=drive_link) and unzip them with ``` unzip yolov8_trained_weights.zip && rm -rf yolov8_trained_weights.zip ```

Execute ```fetch_images.py``` with the correct directories to fetch the partial frames.
Execute ```evaluate.py``` to analyze the performances. Please change the yaml files accordingly.
