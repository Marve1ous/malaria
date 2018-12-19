from tensorflow.python.keras.applications import mobilenet_v2
import load

model = mobilenet_v2.MobileNetV2(include_top=False)

a, b = load.load_resized_data(224, 224)
