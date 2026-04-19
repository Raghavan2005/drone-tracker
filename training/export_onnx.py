"""Export trained DroneNet-Pico model to ONNX format."""

import argparse

import torch
import onnx

from models import DroneNetPico


def export(checkpoint_path, output_path, num_classes=5, input_size=416):
    model = DroneNetPico(num_classes=num_classes, input_size=input_size)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        out = model(dummy)
        print(f"Output shape: {out.shape}")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=17,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to: {output_path}")
    print(f"Input: {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"Output: {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs/train/best.pt")
    parser.add_argument("--output", type=str, default="models/drone_net_pico.onnx")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=416)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.num_classes, args.input_size)
