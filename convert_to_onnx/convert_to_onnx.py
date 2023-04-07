import torch
import torchvision
import onnx

# Load the PyTorch model
model = torch.load('craft_mlt_25k.pth', map_location='cpu')

# Set the input shape
input_shape = (1, 3, 768, 768)

# Create a dummy input tensor
dummy_input = torch.randn(input_shape)

# Convert the model to ONNX format
onnx_path = 'craft_mlt_25k.onnx'
# torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=11)

input_names = [ "actual_input" ]
output_names = [ "output" ]
torch.onnx.export(model,
                 dummy_input,
                 "craft_mlt_25k.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True,
                 )

# Verify the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
