import json
import math
import re
from chatbot.emotion.types import EmotionResult
from chatbot.emotion.config import unique_object

class ResultValidationError(ValueError):
    pass

def parse_result(raw,taxonomy):
    text=raw.strip()
    match=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',text)
    if match:
        text=match[1]
    data=json.loads(text,object_pairs_hook=unique_object)
    if not isinstance(data,dict):
        raise ResultValidationError('result must be an object')
    required={'primary_emotion','confidence','secondary_emotions','evidence','reply_strategy','trajectory_note','safety_level'}
    if not required<=data.keys():
        raise ResultValidationError('missing emotion result fields')
    primary=data['primary_emotion']; secondary=data['secondary_emotions']; confidence=data['confidence']
    if not isinstance(primary,str) or primary not in taxonomy.labels:
        raise ResultValidationError('unknown primary emotion')
    if (not isinstance(secondary,list) or any(not isinstance(x,str) or x not in taxonomy.labels for x in secondary)
        or len(set(secondary))!=len(secondary) or primary in secondary or 'no_emotion' in secondary
        or (primary=='no_emotion' and secondary)):
        raise ResultValidationError('invalid secondary emotions')
    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:
        raise ResultValidationError('invalid confidence')
    if any(not isinstance(data[k],str) for k in ('evidence','reply_strategy','trajectory_note','safety_level')):
        raise ResultValidationError('invalid text field')
    if data['safety_level'] not in ('normal','supportive','crisis'):
        raise ResultValidationError('invalid safety level')
    return EmotionResult(**{k:data[k] for k in required},primary_family=taxonomy.families[primary])
