"""Run one active-perception experiment with local Qwen ball inspection."""
import argparse
from active_perception.config import load_config
from active_perception.policy import decision_from_expression, legal_actions
from active_perception.qwen import QwenAgent
from active_perception.runner import ExperimentRunner

class OrderedAgent:
    """Deterministic action ordering plus local-Qwen visual inspection."""
    def __init__(self, model_name):
        self.vision = QwenAgent(model_name)
        self.model_name = model_name
        self.requires_render_context_rebuild = True
        self.ball_inspection_requires_render_context_rebuild = True
        self.last_raw = None
    def decide(self, task, knowledge, history, step, max_steps, images,
               last_measurement, legal, corrections=()):
        choice = legal_actions(knowledge)[0]
        self.last_raw = choice
        return decision_from_expression(choice)
    def inspect_green_ball(self, image, target):
        return self.vision.inspect_green_ball(image, target)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()
    ExperimentRunner(load_config(args.config), args.output, args.model,
                     agent=OrderedAgent(args.model)).run()

if __name__ == "__main__":
    main()
