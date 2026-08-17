"""Render labeled initial-layout previews without running any interaction."""
from pathlib import Path
import sys
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active_perception.config import load_config
from active_perception.environment import create_environment
from active_perception.perception import capture_observation
from active_perception.visualization import point_overview_camera

CONFIGS = ROOT / "configs/final_showcase"
OUTPUT = ROOT / "outputs/final_showcase_previews"

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(CONFIGS.glob("case_*_boxes.yaml")):
        config = load_config(path)
        environment = create_environment(config)
        try:
            raw = environment.reset()
            point_overview_camera(environment, config)
            raw = environment._get_observations(force_update=True)
            observation = capture_observation(environment, raw, config)
            destination = OUTPUT / f"{path.stem}.png"
            image = cv2.cvtColor(observation.labeled_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f"failed to save {destination}")
            print(destination)
        finally:
            environment.close()

if __name__ == "__main__":
    main()
