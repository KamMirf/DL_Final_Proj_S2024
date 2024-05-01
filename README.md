# DL_Final_Proj_S2024
Deepfake Detection

### Organize files as such
```
|-- code
    |-- download_FF.py
    |-- hyperparameters.py
    |-- main.py
    |-- model.py
    |-- preprocess.py
|-- cropped_data
    |-- deepfake_cropped
    |-- original_cropped
    |-- test
        |-- deepfake
        |-- original
    |-- train
        |-- deepfake
        |-- original
|-- data
    |-- manipulated_sequences
    |-- original_sequences
|-- filtered_data
    |-- Deepfakes
    |-- originals
|-- sample_data
    |-- deepfake
    |-- original
```

### TensorFlow Metal

```
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
pip install numpy upgrade
pip install pandas upgrade
pip install matplotlib --upgrade
pip install scipy --upgrade
```

## Verify TF uses GPU
https://discuss.tensorflow.org/t/tensorflow-on-apple-m2/14804
In terminal do the following:
```
python3
Python 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:35:25) [Clang 16.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import platform
>>> platform.platform()
'macOS-13.5-arm64-arm-64bit'
>>> import tensorflow as tf
>>> tf.__version__
'2.13.0'
>>> gpu = len(tf.config.list_physical_devices("GPU"))>0
>>> gpu
True
>>> tf.config.list_physical_devices("GPU")
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> exit()
```