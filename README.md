# 🧔 Gender Classification on TinyML (Phase 1: Battery-Powered) 👩

[![TensorFlow Lite Micro](https://img.shields.io/badge/TensorFlow%20Lite%20Micro-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite/microcontrollers)
[![Arduino](https://img.shields.io/badge/-Arduino-00979D?style=for-the-badge&logo=Arduino&logoColor=white)](https://www.arduino.cc/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?style=for-the-badge&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This repository contains the complete codebase, model training workflows, and hardware deployment instructions for an **on-device gender detection system** running on an ultra-low-power microcontroller. The project utilizes post-training **post-training INT8 quantization** to deploy a custom Convolutional Neural Network (CNN) on an **Arduino Nano 33 BLE** using **TensorFlow Lite Micro** and an **Arducam Mini OV2640** camera module.

This work represents Phase 1 (Battery-Powered Prototype) of the research topic: **"ML-Driven 6G AR Optimization for Self-Sustainable IoT Devices"**. Phase 2 aims to transition this prototype into a solar-powered, battery-less, energy-harvesting edge node.

---

## 📖 Table of Contents

- [🚀 Motivation & Background](#-motivation--background)
- [🔌 Hardware Requirements & Wiring](#-hardware-requirements--wiring)
- [📦 Software & Library Setup](#-software--library-setup)
- [⚙️ Model Architecture & Training](#-model-architecture--training)
- [📊 Post-Training Quantization & Export](#-post-training-quantization--export)
- [💻 Deploying to Arduino BLE 33](#-deploying-to-arduino-ble-33)
- [📈 Benchmarking Results](#-benchmarking-results)
- [⚠️ Current Limitations & Future Work](#-current-limitations--future-work)
- [🗂️ Project Directory Structure](#-project-directory-structure)
- [👤 Author & Acknowledgements](#-author--acknowledgements)
- [📄 License](#-license)

---

## 🚀 Motivation & Background

Integrating machine learning into industrial Internet of Things (IoT) devices and Sixth-Generation (6G) communication networks is vital to enable real-time local processing. Traditional configurations require edge nodes to stream high-resolution data to centralized cloud systems, which incurs high latency and power consumption. 

**TinyML** resolves this by executing optimized neural networks directly on microcontrollers. This repository provides a reference implementation of a complete on-device vision pipeline. The system captures images, decompresses JPEGs, performs image preprocessing, runs local integer inference, and displays classification results ("Men" or "Women") along with probability metrics on an LCD screen.

---

## 🔌 Hardware Requirements & Wiring

### Hardware Checklist
1. **Arduino Nano 33 BLE** (Central Microcontroller, ARM Cortex-M4F @ 64MHz, 256KB SRAM, 1MB Flash)
2. **Arducam Mini OV2640 2 Megapixels Plus** Camera Module (SKU: `B0067`)
3. **LiquidCrystal 16x2 LCD** (Visual output module)
4. **Breadboard & Jumper Wires**
5. **Battery Pack** (3.7V Li-Po battery or standard 5V USB battery bank)

### Circuit Connections Pinout

#### 1. Arducam Mini OV2640 to Arduino Nano 33 BLE
The camera communicates via SPI (for image data) and I2C (for camera settings control).

| Arducam Pin | Arduino Pin | Description |
| :--- | :--- | :--- |
| **CS** | D10 | SPI Chip Select (Active Low) |
| **MOSI** | D11 / MOSI | SPI Master Out Slave In |
| **MISO** | D12 / MISO | SPI Master In Slave Out |
| **SCK** | D13 / SCK | SPI Serial Clock |
| **SDA** | A4 / SDA | I2C Serial Data |
| **SCL** | A5 / SCL | I2C Serial Clock |
| **VCC** | +3.3V | Power Supply (3.3V) |
| **GND** | GND | Ground |

#### 2. LCD 16x2 Display to Arduino Nano 33 BLE
The LCD is wired in 4-bit data mode.

| LCD Pin | Arduino Pin | Description |
| :--- | :--- | :--- |
| **RS** | D2 | Register Select |
| **EN** | D3 | Enable Pin |
| **D4** | D4 | Data Bit 4 |
| **D5** | D5 | Data Bit 5 |
| **D6** | D6 | Data Bit 6 |
| **D7** | D7 | Data Bit 7 |
| **VSS** | GND | Ground |
| **VDD** | +5V or +3.3V | Power Supply |
| **VO** | Potentiometer | Contrast Adjustment Pin |
| **RW** | GND | Read/Write (Write Mode) |

---

## 📦 Software & Library Setup

### 1. Python Environment Setup
Install the necessary packages for training, exporting, and quantizing the model:
```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 2. Arduino IDE Settings
To compile the sketch, install the following libraries using the **Arduino Library Manager**:
* **Arduino_TensorFlowLite** (Specifically version **2.4.0-ALPHA** or compatible with TFLite Micro)
* **ArduCAM** (For interfacing with the camera shield)
* **TJpg_Decoder** (For decoding JPEG image buffers to raw arrays)
* **LiquidCrystal** (For 16x2 LCD output)

Under **Tools** -> **Board**, select **Arduino Mbed OS Nano Boards** -> **Arduino Nano 33 BLE**.

---

## ⚙️ Model Architecture & Training

To fit within the **256 KB SRAM** limit of the Arduino Nano 33 BLE, we developed a highly compact Convolutional Neural Network (CNN) based on the **MicroNet-M3** architecture.

### 1. Training Setup
- **Dataset:** Grayscale cropped facial images from a subset of the **UTKFace dataset**. (A Google Drive folder link is located inside the [dataset_link.txt](dataset_link.txt) file).
- **Preprocessing:** Images are resized to **96x96** and normalized to the range `[0.0, 1.0]`.
- **Model Output:** Single node with a Sigmoid activation (predicting values from `0` to `1`: values `< 0.5` represent Men, and values `≥ 0.5` represent Women).

### 2. Model Structure (Keras Sequential)
The model consists of 5 sequential micro-blocks containing convolution, batch normalization, and ReLU activations:
```python
def micro_block(x, filters, strides=1):
    x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_micronet_m3(input_shape=(96, 96, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    x = micro_block(inputs, 8, 2)    # 48x48
    x = micro_block(x, 16, 2)       # 24x24
    x = micro_block(x, 32, 2)       # 12-12
    x = micro_block(x, 64, 2)       # 6x6
    x = micro_block(x, 128, 2)      # 3x3
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, x)
```

The training script compiles the model using the **Adam optimizer** and **Binary Crossentropy loss**, running for **25 epochs**:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))
```

The training history and implementation details are fully documented in the notebook: [model_training/gender_detection_model_train.ipynb](model_training/gender_detection_model_train.ipynb).

---

## 📊 Post-Training Quantization & Export

To deploy the deep neural network on the Arduino's limited hardware, the model must be optimized to reduce memory footprint and execution latency.

### 1. Post-Training INT8 Quantization
The floating-point model (32-bit floats) is quantized to an 8-bit integer format (`int8`). A representative dataset from the training set calibrates the range of activation parameters:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Representative dataset generator
def representative_dataset_gen():
    for i in range(X_train.shape[0]):
        yield [X_train[i:i+1]]

converter.representative_dataset = representative_dataset_gen
tflite_model_quantized = converter.convert()

with open('gender_classification_model_quantized_int8_12_07.tflite', 'wb') as f:
    f.write(tflite_model_quantized)
```

### 2. Exporting to C++ Header File
The quantized `.tflite` model is converted into a C++ static char array using a Python script. This outputs the file [gender_detection_model_12_07.h](arduino_deployment/gender_detection_model_12_07.h):
```cpp
#ifndef GENDER_DETECTION_MODEL_H_
#define GENDER_DETECTION_MODEL_H_

static const unsigned char model_data[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
static const int model_data_len = 108520; // 108 KB

#endif
```

---

## 💻 Deploying to Arduino BLE 33

The Arduino firmware manages camera re-initialization (to prevent FIFO length buffer errors), image capture, JPEG decoding, quantization mapping, and model execution.

### Deployment Walkthrough
1. **Camera Setup:** The camera sensor is initialized to outputs JPEGs at a resolution of $160\times 120$ pixels.
2. **JPEG Decompression:** The `TJpg_Decoder` library uses a callback to process 8x8 or 16x16 JPEG blocks on the fly, scaling the image to a $96\times 96$ resolution, converting color values to grayscale, and populating the TFLite input buffer:
   ```cpp
   bool decodeJpegToGrayscale(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
     for (int16_t row = 0; row < h; row++) {
       for (int16_t col = 0; col < w; col++) {
         uint16_t pixel = bitmap[row * w + col];
         uint8_t r = ((pixel >> 11) & 0x1F) * 255 / 31;
         uint8_t g = ((pixel >> 5) & 0x3F) * 255 / 63;
         uint8_t b = (pixel & 0x1F) * 255 / 31;
         uint8_t gray = (r * 30 + g * 59 + b * 11) / 100;
         if (input_index < input->bytes) {
           input->data.int8[input_index++] = (int8_t)(gray - 128); // Quantize input
         }
       }
     }
     return true;
   }
   ```
3. **Execution:** The runtime allocates a 100 KB Tensor Arena on the heap and invokes the interpreter:
   ```cpp
   interpreter->Invoke();
   ```
4. **De-quantization:** The output node maps the integer score back to a float value to output classifications and probability scores on the LCD screen.

Check the complete firmware sketch for deployment instructions: [arduino_deployment/arduino_deployment.ino](arduino_deployment/arduino_deployment.ino).

---

## 📈 Benchmarking Results

### Model Size & Parameters Footprint

| Format | File Size | Parameters | Memory Type |
| :--- | :--- | :--- | :--- |
| **Keras Model (`.h5`)** | 1.22 MB | 297,525 | Host PC Disk Space |
| **TFLite Float (`.tflite`)** | 389 KB | 297,525 | Host PC Disk Space |
| **TFLite Quantized (`.tflite`)**| 105 KB | 297,525 | Microcontroller ROM (Flash) |
| **INT8 Quantized (`.h`)** | **108 KB** | **297,525** | **Microcontroller ROM (Flash)** |

### On-Device Memory Utilization (Arduino Nano 33 BLE)
- **Available SRAM (RAM):** 256 KB
- **Model Runtime Overhead (Tensor Arena):** 100 KB
- **Model Storage Overhead (ROM/Flash):** 108 KB (Fits comfortably within the 1 MB Flash limit)
- **Available SRAM Left for Program Logic:** ~148 KB (Ensures zero out-of-memory crashes)

---

## ⚠️ Current Limitations & Future Work

### Limitations
* **Model Accuracy:** The current prototype yields an initial classification accuracy of **50–60%**. This is baseline and requires training improvements.
* **Power Profiling:** The power draw of the camera capture and inference cycles has not yet been quantitatively measured.

### Future Work Roadmap
1. **Training Enhancements:** Introduce data augmentation (cropping, rotations, and contrast adjustments) and expand the dataset to improve classification accuracy beyond 85%.
2. **Current Profiling:** Connect the hardware to a **Nordic Power Profiler Kit II (PPK2)** to record milliwatt-second metrics during the capture and inference loops.
3. **Transition to Phase 2 (Solar Battery-less Operation):** Integrated a solar panel (AM-5608) and a Power Management Unit (AEM10941) to power the board via a 1.5 F supercapacitor. Apply the Double Q-Learning Energy Management Strategy studied in Phase 1 to sleep the device during low harvesting conditions.

---

## 🗂️ Project Directory Structure

```
Gender_Classification_TinyML_with_battery/
├── arduino_deployment/
│   ├── arduino_deployment.ino          # Main Arduino deployment sketch
│   └── gender_detection_model_12_07.h  # Model weights array header (108 KB)
├── model_training/
│   ├── gender_detection_model_train.ipynb # Jupyter notebook for model training
│   └── models/                         # Directory for intermediate trained weights
├── .gitignore                          # Standard git ignore file
├── dataset_link.txt                    # Contains the Google Drive dataset download link
├── Internship_Report_Polished.md        # Polished academic-grade internship report
└── README.md                           # This README file
```

---

## 👤 Author & Acknowledgements

* **Author:** Kaushal Sharma (B.Tech Mathematics and Computing, 2nd year, RGIPT Jais)
* **Supervisor:** Dr. Ashwani Sharma, Department of Electronics Engineering, RGIPT Jais
* **Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT), Jais, Amethi, UP, India

Special thanks to Dr. Ashwani Sharma for guidance and supervision during the two-month internship (21 May – 20 July 2025).

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
