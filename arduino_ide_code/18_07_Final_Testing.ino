#include "gender_detection_model_12_07.h"
#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>
#include <LiquidCrystal.h>
#include <TJpg_Decoder.h>

#ifdef swap
#undef swap
#endif

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// -------------------- Hardware --------------------
#define CS_PIN 10
ArduCAM myCAM(OV2640, CS_PIN);
LiquidCrystal lcd(2, 3, 4, 5, 6, 7);

// -------------------- TensorFlow Lite Globals --------------------
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Index to track position in the input tensor
int input_index = 0;

constexpr int kTensorArenaSize = 100 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// This callback function is the key to our image processing.
// It's called by the TJpg_Decoder for each block of the JPEG image.
bool decodeJpegToGrayscale(int16_t x, int16_t y, uint16_t w, uint16_t h, uint16_t* bitmap) {
  // Loop through all the pixels in the decoded block
  for (int16_t row = 0; row < h; row++) {
    for (int16_t col = 0; col < w; col++) {
      uint16_t pixel = bitmap[row * w + col];

      // Extract RGB565 color components
      uint8_t r = (pixel >> 11) & 0x1F;
      uint8_t g = (pixel >> 5) & 0x3F;
      uint8_t b = pixel & 0x1F;

      // Convert to full 8-bit color
      r = (r * 255) / 31;
      g = (g * 255) / 63;
      b = (b * 255) / 31;

      // Convert to grayscale using the standard formula
      uint8_t gray = (r * 30 + g * 59 + b * 11) / 100;

      // Make sure we don't write past the end of the input tensor buffer
      if (input_index < input->bytes) {
        // Quantize and store the grayscale value into the TFLite input tensor
        input->data.int8[input_index++] = (int8_t)(gray - 128);
      }
    }
  }
  return true;  // Continue decoding
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  lcd.begin(16, 2);
  lcd.print("Initializing...");

  // Initialize SPI and I2C for the camera
  SPI.begin();
  Wire.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  // Initialize the camera
  myCAM.write_reg(0x07, 0x80);
  delay(100);
  myCAM.write_reg(0x07, 0x00);
  delay(100);

  lcd.clear();
  lcd.print("Checking SPI...");
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  uint8_t temp = myCAM.read_reg(ARDUCHIP_TEST1);
  if (temp != 0x55) {
    lcd.clear();
    lcd.print("SPI Error");
    Serial.println("SPI Interface Error!");
    while (1)
      ;
  }

  // Initialize the camera sensor
  myCAM.InitCAM();
  // Set the format to JPEG
  myCAM.set_format(JPEG);
  // Set resolution. Your model needs 96x96. The closest is 160x120.
  // The TJpg_Decoder will scale this down.
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);
  myCAM.clear_fifo_flag();
  Serial.println("Camera initialized.");

  // Initialize the JPEG decoder
  TJpgDec.setJpgScale(1);
  TJpgDec.setCallback(decodeJpegToGrayscale);

  // Load model
  lcd.clear();
  lcd.print("Loading Model...");
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // <<< ACTION REQUIRED: Use the actual model data array name from your .h file >>>
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    lcd.clear();
    lcd.print("Model mismatch");
    while (1)
      ;
  }

  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddMean();
  resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    lcd.clear();
    lcd.print("Alloc Failed");
    Serial.println("❌ Tensor allocation failed");
    while (1)
      ;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  lcd.clear();
  lcd.print("System Ready");
  Serial.println("System Ready");
  delay(1000);
}

void loop() {
  lcd.clear();
  lcd.print("Ready to Capture");
  delay(1000); // Wait a moment before starting

  // --- <<< NEW CAMERA RE-INITIALIZATION LOGIC >>> ---
  // Re-initialize the camera settings before every capture to ensure
  // it's in a known good state. This is the key to fixing the "FIFO length 0" error.
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);
  delay(100); // Give the sensor a moment to stabilize
  // --- <<< END OF NEW LOGIC >>> ---

  lcd.clear();
  lcd.print("Capturing Image...");
  Serial.println("Starting image capture...");

  input_index = 0;

  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  // --- Timeout logic ---
  unsigned long start_time = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - start_time > 3000) {  // 3-second timeout
      Serial.println("❌ ERROR: Camera timed out. Check connections.");
      lcd.clear();
      lcd.print("Camera Timeout");
      delay(5000);
      return;  // Go back to the start of loop()
    }
  }

  Serial.println("Capture complete.");
  lcd.clear();
  lcd.print("Processing...");

  // --- DECODING LOGIC (Unchanged) ---

  // 1. Get the length of the image data in the FIFO buffer.
  uint32_t fifo_len = myCAM.read_fifo_length();
  if (fifo_len == 0 || fifo_len > 30000) { // Safety check
      Serial.println("Error: FIFO length is 0 or too large.");
      // This is where your error was coming from.
      // The re-initialization above should fix this.
      return;
  }

  // 2. Dynamically allocate a buffer on the heap to hold the entire image.
  uint8_t *jpeg_buffer = (uint8_t *)malloc(fifo_len);
  if (!jpeg_buffer) {
      Serial.println("Error: malloc failed to allocate buffer.");
      return;
  }

  // 3. Read the entire JPEG image from the camera into the buffer.
  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  for (uint32_t i = 0; i < fifo_len; i++) {
      jpeg_buffer[i] = SPI.transfer(0x00);
  }
  myCAM.CS_HIGH();
  
  // 4. Decode the image from the buffer we just filled.
  JRESULT result = TJpgDec.drawJpg(0, 0, jpeg_buffer, fifo_len);

  // 5. CRITICAL: Free the memory that was allocated for the buffer.
  free(jpeg_buffer);

  if (result != JDR_OK) {
      Serial.print("Error: JPEG decode failed. Result: ");
      Serial.println(result);
      lcd.clear();
      lcd.print("Decode Error");
      delay(2000);
      return;
  }
  
  // --- END OF DECODING LOGIC ---

  Serial.println("Decoding complete.");

  if (input_index < 1000) {
    Serial.print("Error: Image processing only filled ");
    Serial.print(input_index);
    Serial.println(" bytes. Check camera focus and lighting.");
    lcd.clear();
    lcd.print("Image Proc Error");
    delay(2000);
    return;
  }

  lcd.clear();
  lcd.print("Inferencing...");
  Serial.println("Running inference...");

  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Inference failed!");
    lcd.clear();
    lcd.print("Infer Error");
    delay(2000);
    return;
  }

  int8_t output_value = output->data.int8[0];
  float prediction = (output_value - output->params.zero_point) * output->params.scale;

  lcd.clear();
  lcd.print("Gender: ");
  if (prediction >= 0.5) {
    lcd.print("Women");
    Serial.println("Prediction: Women");
  } else {
    lcd.print("Men");
    Serial.println("Prediction: Men");
  }

  lcd.setCursor(0, 1);
  lcd.print("Prob: ");
  lcd.print(prediction, 2);

  delay(5000);
}