import argparse
import json
import sys
from pathlib import Path

import torch
from ray.rllib.policy.policy import Policy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wrappers.action_wrappers import VALID_HEADING_TYPES


DEFAULT_FRAME_STACK = 3
DEFAULT_OBS_HEIGHT = 84
DEFAULT_OBS_WIDTH = 84
DEFAULT_CROP_TOP_RATIO = 0.33
DEFAULT_HEADING_TYPE = "heading"
DEFAULT_FORWARD_SPEED = 1.0
DEFAULT_MAX_STEER = 1.0
DEFAULT_OPSET_VERSION = 11


class DeterministicHeadingPolicy(torch.nn.Module):
    """Wrap an RLlib Torch policy model and expose only the action mean."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model({"obs": obs}, [], None)
        return logits[..., :1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an RLlib PPO heading policy checkpoint to ONNX for deployment.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the RLlib policy checkpoint directory, e.g. .../policies/default_policy",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional JSON path for deployment metadata. Defaults to <output>.json",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help="ONNX opset version. Use 11 for broad compatibility on Jetson/ORT 1.8.",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=DEFAULT_FRAME_STACK,
        help="Number of RGB frames stacked on the channel axis during training/deployment.",
    )
    parser.add_argument(
        "--obs-height",
        type=int,
        default=DEFAULT_OBS_HEIGHT,
        help="Preprocessed observation height.",
    )
    parser.add_argument(
        "--obs-width",
        type=int,
        default=DEFAULT_OBS_WIDTH,
        help="Preprocessed observation width.",
    )
    parser.add_argument(
        "--crop-top-ratio",
        type=float,
        default=DEFAULT_CROP_TOP_RATIO,
        help="Top crop ratio applied before resizing.",
    )
    parser.add_argument(
        "--heading-type",
        default=DEFAULT_HEADING_TYPE,
        choices=VALID_HEADING_TYPES,
        help="Heading wrapper type expected at deployment.",
    )
    parser.add_argument(
        "--forward-speed",
        type=float,
        default=DEFAULT_FORWARD_SPEED,
        help="Forward speed multiplier used by HeadingToWheelsWrapper.",
    )
    parser.add_argument(
        "--max-steer",
        type=float,
        default=DEFAULT_MAX_STEER,
        help="Max steering magnitude used by HeadingToWheelsWrapper.",
    )
    return parser.parse_args()


def build_metadata(*, args: argparse.Namespace, checkpoint_dir: Path, policy: Policy) -> dict:
    return {
        "export_format": "onnx",
        "checkpoint": str(checkpoint_dir),
        "policy_class": type(policy).__name__,
        "observation_space": tuple(int(v) for v in policy.observation_space.shape),
        "action_space": tuple(int(v) for v in policy.action_space.shape),
        "input_name": "obs",
        "output_name": "heading",
        "input_dtype": "float32",
        "input_range": [0.0, 1.0],
        "input_layout": "NHWC",
        "frame_stack": int(args.frame_stack),
        "rgb_channels": 3,
        "crop_top_ratio": float(args.crop_top_ratio),
        "resize_hw": [int(args.obs_height), int(args.obs_width)],
        "heading_type": str(args.heading_type),
        "forward_speed": float(args.forward_speed),
        "max_steer": float(args.max_steer),
        "heading_to_wheels": {
            "heading": "left=clip(1+heading,0,1), right=clip(1-heading,0,1)",
            "heading_smooth": "heading=(model_output**3)*max_steer; left=clip(1+heading,0,1); right=clip(1-heading,0,1)",
            "heading_clipped_08": "heading=round(clip(model_output,-0.8,0.8),2); heading=round(clip(heading*max_steer,-0.8,0.8),2); left=clip(1+heading,0,1); right=clip(1-heading,0,1)",
        },
        "notes": [
            "Model output is the deterministic Gaussian mean used by policy.compute_single_action(explore=False).",
            "Deployment must preserve crop, resize, normalization, and channel-stacked frame order.",
        ],
        "opset_version": int(args.opset_version),
    }


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        Path(args.metadata_output).expanduser().resolve()
        if args.metadata_output
        else output_path.with_suffix(output_path.suffix + ".json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    policy = Policy.from_checkpoint(str(checkpoint_dir))
    base_model = policy.model.cpu().eval()
    export_model = DeterministicHeadingPolicy(base_model).cpu().eval()

    dummy_obs = torch.zeros(
        1,
        int(args.obs_height),
        int(args.obs_width),
        3 * int(args.frame_stack),
        dtype=torch.float32,
    )

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_obs,
            str(output_path),
            export_params=True,
            opset_version=int(args.opset_version),
            do_constant_folding=True,
            input_names=["obs"],
            output_names=["heading"],
            dynamic_axes={
                "obs": {0: "batch"},
                "heading": {0: "batch"},
            },
        )

    metadata = build_metadata(args=args, checkpoint_dir=checkpoint_dir, policy=policy)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Exported ONNX model to: {output_path}")
    print(f"Wrote metadata JSON to: {metadata_path}")


if __name__ == "__main__":
    main()
