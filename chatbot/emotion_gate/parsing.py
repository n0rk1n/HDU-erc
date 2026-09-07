import json
from chatbot.emotion_gate.types import GateResult


def parse_gate_result(raw: str) -> GateResult:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate gate result key')
            result[key] = value
        return result
    data = json.loads(raw, object_pairs_hook=unique)
    if (not isinstance(data, dict) or set(data) != {'should_analyze', 'reason'}
            or type(data['should_analyze']) is not bool
            or not isinstance(data['reason'], str) or not data['reason'].strip()):
        raise ValueError('invalid gate result')
    return GateResult(data['should_analyze'], data['reason'].strip())
