"""Qwen3-VL adapter: build multimodal input, run inference, parse one decision."""

import json
import re
from pathlib import Path

from PIL import Image

from .policy import Decision


SYSTEM_PROMPT = """你是主动因果感知机器人的高层动作选择器，不是自由问答助手。

必须严格遵守以下优先级规则：
1. LEGAL_DECISIONS_JSON 是程序计算出的权威动作对象列表。必须完整复制其中一个对象的 action 和 target 值。
2. KNOWN_JSON 是已测量事实。false 表示已被证据否定，禁止当成 true；null 表示未知，禁止声称已确认。
3. CORRECTIONS_JSON 是本步骤已经被拒绝的输出。禁止再次输出其中相同的 action-target。
4. 图像只用于识别位置和观察外观，不能推翻 KNOWN_JSON，也不能证明隐藏属性。
5. action 字段只能是 push、press、lift_box、stop 之一，禁止在 action 中写 push(B) 这种带目标的表达式。

只输出一个 JSON object，不要输出 Markdown 或额外文字：
{"thinking":"仅描述已知证据","action":"push|press|lift_box|stop","target":"从合法列表复制","reason":"为何该动作属于合法集合"}"""


BALL_SYSTEM_PROMPT = """你是机器人的视觉检查器。只判断当前 RGB 图像中，罩盒被移开后的原位置是否清楚出现绿色小球。

只能根据图像本身判断，不得使用文件名、配置、历史真值或颜色像素统计。绿色小球必须是桌面上的独立球形物体；文字、标记、按钮或盒子色块不算。即使不确定也必须给出 true 或 false，并降低 confidence。

只输出一个 JSON object，不要输出 Markdown 或额外文字：
{"contains_green_ball":true,"confidence":"high|medium|low","reason":"简短描述可见依据"}"""


def parse_decision(text: str) -> Decision:
    """Decode the first complete JSON object and validate its basic shape."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Qwen did not return JSON: {text}")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    action = data.get("action")
    target = data.get("target")
    # Small models occasionally place the complete expression in action even
    # when the schema asks for separate fields. Normalize it before validation.
    actions = "push|press|lift_box|stop"
    expression = re.fullmatch(rf"({actions})\(([^()]+)\)", str(action))
    if expression:
        action, embedded_target = expression.groups()
        if target not in (None, embedded_target):
            raise ValueError(
                f"conflicting targets: action={embedded_target}, target={target}"
            )
        target = embedded_target
    if action not in {"push", "press", "lift_box", "stop"}:
        raise ValueError(f"invalid action: {action}")
    if not isinstance(target, str) or not target:
        raise ValueError(f"invalid target: {target}")
    return Decision(action, target, data.get("thinking", ""), data.get("reason", ""))


def parse_ball_inspection(text: str):
    """Parse Qwen's binary visual judgment without any rule-based fallback."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Qwen did not return ball inspection JSON: {text}")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    value = data.get("contains_green_ball")
    confidence = data.get("confidence")
    reason = data.get("reason")
    if not isinstance(value, bool):
        raise ValueError("contains_green_ball must be a JSON boolean")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError("confidence must be high, medium, or low")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("ball inspection reason must be non-empty")
    return {"value": value, "confidence": confidence, "reason": reason.strip()}


def build_user_text(task, knowledge, history, measurement, step, max_steps,
                    legal_options, corrections):
    """Keep the prompt construction testable without loading a GPU model."""
    known = knowledge.snapshot()
    legal_decisions = []
    for expression in legal_options:
        action, rest = expression.split("(", 1)
        legal_decisions.append({"action": action, "target": rest[:-1]})
    return (
        f"TASK: {task}\nSTEP: {step}/{max_steps}\n"
        f"KNOWN_JSON: {json.dumps(known, ensure_ascii=False)}\n"
        f"LEGAL_DECISIONS_JSON: {json.dumps(legal_decisions, ensure_ascii=False)}\n"
        f"LAST_MEASUREMENT_JSON: {json.dumps(measurement, ensure_ascii=False)}\n"
        f"INTERACTION_HISTORY_JSON: {json.dumps(history[-4:], ensure_ascii=False)}\n"
        f"CORRECTIONS_JSON: {json.dumps(corrections, ensure_ascii=False)}\n"
        "FINAL_CHECK: 从 LEGAL_DECISIONS_JSON 完整复制一个 action 和 target；"
        "action 只能写动词，禁止写成 push(B)；"
        "若 CORRECTIONS_JSON 非空，不得重复其中被拒绝的 action-target。"
        "现在只输出一个 JSON object。"
    )


class QwenAgent:
    requires_render_context_rebuild = True
    ball_inspection_requires_render_context_rebuild = True
    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", max_pixels=200704,
                 max_new_tokens=420, attention="sdpa"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-VL requires an NVIDIA CUDA GPU")
        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=100352, max_pixels=max_pixels
        )
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.float16, attn_implementation=attention,
                device_map="cuda", low_cpu_mem_usage=True,
            ).eval()
        except RuntimeError:
            if attention != "sdpa":
                raise
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.float16, attn_implementation="eager",
                device_map="cuda", low_cpu_mem_usage=True,
            ).eval()
        self.last_raw = ""

    @staticmethod
    def _open_image(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        return Image.open(Path(value)).convert("RGB")

    def decide(self, task, knowledge, history, step, max_steps, images,
               measurement, legal_options, corrections=None) -> Decision:
        content = []
        for label, value in images:
            content.append({"type": "text", "text": f"IMAGE {label}:"})
            content.append({"type": "image", "image": self._open_image(value)})
        content.append({
            "type": "text",
            "text": build_user_text(
                task, knowledge, history, measurement, step, max_steps,
                legal_options, corrections or [],
            ),
        })
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        self.last_raw = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        return parse_decision(self.last_raw)

    def inspect_green_ball(self, image, target):
        """Use Qwen alone to judge the revealed RGB image."""
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": BALL_SYSTEM_PROMPT}
            ]},
            {"role": "user", "content": [
                {"type": "image", "image": self._open_image(image)},
                {"type": "text", "text": (
                    f"这是盒子 {target} 被移开后的检查图像。"
                    "判断原位置是否出现独立的绿色小球。"
                )},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=180, do_sample=False
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        self.last_ball_raw = raw
        parsed = parse_ball_inspection(raw)
        return {
            "property": "contains_target", "value": parsed["value"],
            "confidence": parsed["confidence"], "reason": parsed["reason"],
            "method": "qwen_vl_rgb_visual_judgment",
            "model": self.model_name, "model_raw": raw,
            "provenance": "qwen_only_no_fixed_color_rule",
        }
