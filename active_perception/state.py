"""Knowledge available to Qwen; simulator ground truth never enters this object."""


class KnowledgeState:
    def __init__(self, object_ids, properties):
        self.properties = tuple(properties)
        self.known = {
            object_id: {name: None for name in self.properties}
            for object_id in object_ids
        }

    def update(self, target, evidence):
        """Accept only declared properties and preserve measurement failure as unknown."""
        if target not in self.known:
            raise ValueError(f"unknown target: {target}")
        for name, value in evidence.items():
            if name in self.properties and value is not None:
                self.known[target][name] = bool(value)

    def snapshot(self):
        return {object_id: dict(values) for object_id, values in self.known.items()}

    def text(self):
        lines = []
        for object_id, values in self.known.items():
            facts = ", ".join(
                f"{name}={'?' if value is None else str(value).lower()}"
                for name, value in values.items()
            )
            lines.append(f"{object_id}: {facts}")
        return "\n".join(lines)

