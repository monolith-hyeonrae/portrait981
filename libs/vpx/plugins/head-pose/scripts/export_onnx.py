"""Export 6DRepNet PyTorch checkpoint to ONNX.

Usage:
    python scripts/export_onnx.py --checkpoint 6DRepNet_300W_LP_AFLW2000.pth --output models/6drepnet/sixdrepnet.onnx

Requires: torch, sixdrepnet (pip install git+https://github.com/thohemp/6DRepNet.git)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export(checkpoint_path: str, output_path: str) -> None:
    import torch
    from sixdrepnet.model import SixDRepNet

    model = SixDRepNet(
        backbone_name="RepVGG-B1g2",
        backbone_file="",
        deploy=True,
        pretrained=False,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["rotation_matrix"],
        dynamic_axes={"input": {0: "batch"}, "rotation_matrix": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export 6DRepNet to ONNX")
    parser.add_argument("--checkpoint", required=True, help="PyTorch .pth file")
    parser.add_argument("--output", default="models/6drepnet/sixdrepnet.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
