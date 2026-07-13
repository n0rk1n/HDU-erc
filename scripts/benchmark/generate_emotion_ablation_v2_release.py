"""Generate the formal 500-record emotion ablation v2 release."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chatbot.emotion_labels import EMOTION_LABELS
from scripts.benchmark.emotion_benchmark import BENCHMARK_ROOT
from scripts.benchmark.emotion_benchmark import parallel_equivalence_errors
from scripts.benchmark.emotion_benchmark import validate_records
from scripts.benchmark.emotion_benchmark import write_jsonl

RELEASE_DIR = BENCHMARK_ROOT / "release"
REPORTS_DIR = BENCHMARK_ROOT / "reports"

CORE_PAIR_COUNT = 128
EXTENDED_COUNT_PER_LANGUAGE = 90
CHALLENGE_COUNT_PER_LANGUAGE = 32

SCENARIOS = [
    ("academic_exam", "the exam date moved closer", "考试时间突然提前"),
    ("workplace_deadline", "the project deadline changed overnight", "项目截止时间一夜之间改了"),
    ("family_expectation", "my family kept asking for an answer", "家里一直催我给出答复"),
    ("friendship_distance", "my friend stopped replying after our talk", "朋友在那次聊天后一直没回我"),
    ("health_followup", "the clinic asked me to come back", "医院让我再去复查"),
    ("money_pressure", "the bill was higher than I planned", "账单比我预计的高"),
    ("team_feedback", "the team read my draft out loud", "团队当面读了我的草稿"),
    ("moving_home", "the last box left the old apartment", "最后一个箱子也搬出了旧房子"),
    ("public_speaking", "I imagined standing in front of the room", "我想到要站在一屋子人面前"),
    ("relationship_repair", "they finally wrote back after the argument", "争吵后对方终于回了消息"),
    ("daily_routine", "the quiet evening stretched longer than usual", "安静的晚上比平时更漫长"),
    ("career_change", "the interview panel asked for another call", "面试小组又约了下一通电话"),
]

LABEL_PROFILES: dict[str, dict[str, Any]] = {
    "surprised": {
        "en": "After {event}, I sat there blinking because I truly did not see that coming.",
        "zh": "{event}之后，我愣了好一会儿，真的完全没想到会这样。",
        "evidence_en": "did not see that coming",
        "evidence_zh": "完全没想到",
        "rationale_en": "The user reacts to an unexpected turn.",
        "rationale_zh": "用户在回应意外变化。",
        "intensity": 0.58,
        "secondary": ["impressed"],
    },
    "excited": {
        "en": "After {event}, I kept pacing around with this bright rush in my chest.",
        "zh": "{event}之后，我一直来回走，心里有种亮起来的劲儿。",
        "evidence_en": "bright rush in my chest",
        "evidence_zh": "亮起来的劲儿",
        "rationale_en": "The user shows high-energy positive anticipation.",
        "rationale_zh": "用户表现出高能量的积极期待。",
        "intensity": 0.74,
        "secondary": ["anticipating"],
    },
    "annoyed": {
        "en": "After {event}, I kept thinking, seriously, this again?",
        "zh": "{event}之后，我脑子里一直是：怎么又来这一套？",
        "evidence_en": "this again",
        "evidence_zh": "又来这一套",
        "rationale_en": "The user expresses irritation rather than full anger.",
        "rationale_zh": "用户表达的是烦躁而非强烈愤怒。",
        "intensity": 0.55,
        "secondary": ["angry"],
    },
    "proud": {
        "en": "After {event}, I caught myself smiling because I knew how much work I had put in.",
        "zh": "{event}之后，我忍不住笑了，因为我知道自己真的付出了很多。",
        "evidence_en": "how much work I had put in",
        "evidence_zh": "付出了很多",
        "rationale_en": "The user links the outcome to personal effort and achievement.",
        "rationale_zh": "用户把结果和自己的努力成就联系起来。",
        "intensity": 0.70,
        "secondary": ["confident"],
    },
    "angry": {
        "en": "After {event}, I could feel my jaw tighten because it was so unfair.",
        "zh": "{event}之后，我气得下巴都绷紧了，觉得这太不公平。",
        "evidence_en": "so unfair",
        "evidence_zh": "太不公平",
        "rationale_en": "The user reacts to perceived unfairness with anger.",
        "rationale_zh": "用户因感到不公平而愤怒。",
        "intensity": 0.75,
        "secondary": ["furious"],
    },
    "sad": {
        "en": "After {event}, everything felt quieter, like something had gone missing.",
        "zh": "{event}之后，周围一下子安静下来，像少了什么。",
        "evidence_en": "something had gone missing",
        "evidence_zh": "像少了什么",
        "rationale_en": "The user expresses loss and low mood.",
        "rationale_zh": "用户表达了失落和低落。",
        "intensity": 0.66,
        "secondary": ["lonely"],
    },
    "grateful": {
        "en": "After {event}, I kept thinking how kind it was that someone showed up for me.",
        "zh": "{event}之后，我一直想着有人这样帮我，真的很难得。",
        "evidence_en": "someone showed up for me",
        "evidence_zh": "有人这样帮我",
        "rationale_en": "The user appreciates support received from someone else.",
        "rationale_zh": "用户在感激他人的支持。",
        "intensity": 0.68,
        "secondary": ["caring"],
    },
    "lonely": {
        "en": "After {event}, I wanted to tell someone, but there was no one I could text.",
        "zh": "{event}之后，我很想找人说说，却不知道能发给谁。",
        "evidence_en": "no one I could text",
        "evidence_zh": "不知道能发给谁",
        "rationale_en": "The user emphasizes lack of connection.",
        "rationale_zh": "用户强调缺少可以联系的人。",
        "intensity": 0.69,
        "secondary": ["sad"],
    },
    "impressed": {
        "en": "After {event}, I just kept thinking, wow, they handled that beautifully.",
        "zh": "{event}之后，我一直在想，他们处理得也太漂亮了。",
        "evidence_en": "handled that beautifully",
        "evidence_zh": "处理得也太漂亮",
        "rationale_en": "The user admires someone else's ability or action.",
        "rationale_zh": "用户在欣赏他人的能力或表现。",
        "intensity": 0.62,
        "secondary": ["surprised"],
    },
    "afraid": {
        "en": "After {event}, my stomach dropped because I thought something bad might happen.",
        "zh": "{event}之后，我心里一沉，觉得可能会出事。",
        "evidence_en": "something bad might happen",
        "evidence_zh": "可能会出事",
        "rationale_en": "The user fears a concrete negative outcome.",
        "rationale_zh": "用户害怕具体的不良后果。",
        "intensity": 0.73,
        "secondary": ["anxious"],
    },
    "disgusted": {
        "en": "After {event}, I felt a wave of revulsion and wanted to step away from the whole thing.",
        "zh": "{event}之后，我一阵反感，只想离这件事远一点。",
        "evidence_en": "wave of revulsion",
        "evidence_zh": "一阵反感",
        "rationale_en": "The user expresses revulsion rather than ordinary anger.",
        "rationale_zh": "用户表达的是厌恶而非普通愤怒。",
        "intensity": 0.72,
        "secondary": ["angry"],
    },
    "confident": {
        "en": "After {event}, I felt steady because I knew I could handle the next step.",
        "zh": "{event}之后，我反而稳了，因为知道自己能处理下一步。",
        "evidence_en": "I could handle the next step",
        "evidence_zh": "能处理下一步",
        "rationale_en": "The user expresses belief in their ability.",
        "rationale_zh": "用户表达了对自身能力的确信。",
        "intensity": 0.67,
        "secondary": ["prepared"],
    },
    "terrified": {
        "en": "After {event}, I froze so hard that I could barely get a sentence out.",
        "zh": "{event}之后，我僵住了，几乎连一句完整的话都说不出来。",
        "evidence_en": "could barely get a sentence out",
        "evidence_zh": "说不出来",
        "rationale_en": "The user shows intense fear with a freezing response.",
        "rationale_zh": "用户表现出强烈恐惧和僵住反应。",
        "intensity": 0.90,
        "secondary": ["afraid"],
    },
    "hopeful": {
        "en": "After {event}, I started thinking maybe there is still a way through this.",
        "zh": "{event}之后，我开始觉得也许这件事还有转机。",
        "evidence_en": "still a way through this",
        "evidence_zh": "还有转机",
        "rationale_en": "The user sees a possible positive path forward.",
        "rationale_zh": "用户看到了积极的可能性。",
        "intensity": 0.64,
        "secondary": ["anticipating"],
    },
    "anxious": {
        "en": "After {event}, I kept replaying every possible way it could go wrong.",
        "zh": "{event}之后，我反复想着每一种可能出错的情况。",
        "evidence_en": "could go wrong",
        "evidence_zh": "可能出错",
        "rationale_en": "The user shows anticipatory worry.",
        "rationale_zh": "用户表现出预期性的担忧。",
        "intensity": 0.76,
        "secondary": ["apprehensive"],
    },
    "disappointed": {
        "en": "After {event}, I just stared at the message because I had expected better.",
        "zh": "{event}之后，我盯着消息看了很久，因为原本真的期待更多。",
        "evidence_en": "expected better",
        "evidence_zh": "期待更多",
        "rationale_en": "The user reacts to unmet expectations.",
        "rationale_zh": "用户因期待落空而失望。",
        "intensity": 0.63,
        "secondary": ["sad"],
    },
    "joyful": {
        "en": "After {event}, I felt light enough to laugh out loud by myself.",
        "zh": "{event}之后，我整个人轻快起来，一个人也笑出了声。",
        "evidence_en": "laugh out loud",
        "evidence_zh": "笑出了声",
        "rationale_en": "The user expresses clear happiness and uplift.",
        "rationale_zh": "用户表达了明显的愉快和轻松。",
        "intensity": 0.78,
        "secondary": ["excited"],
    },
    "prepared": {
        "en": "After {event}, I checked my notes once and felt ready instead of rushed.",
        "zh": "{event}之后，我又看了一遍笔记，感觉准备好了，不再慌。",
        "evidence_en": "felt ready",
        "evidence_zh": "准备好了",
        "rationale_en": "The user emphasizes readiness.",
        "rationale_zh": "用户强调准备充分。",
        "intensity": 0.61,
        "secondary": ["confident"],
    },
    "guilty": {
        "en": "After {event}, I kept thinking about what I should have done differently.",
        "zh": "{event}之后，我一直在想自己本来应该怎么做得更好。",
        "evidence_en": "should have done differently",
        "evidence_zh": "应该怎么做得更好",
        "rationale_en": "The user focuses on personal responsibility.",
        "rationale_zh": "用户把重点放在自己的责任上。",
        "intensity": 0.66,
        "secondary": ["ashamed"],
    },
    "furious": {
        "en": "After {event}, I was shaking with anger and had to leave before I snapped.",
        "zh": "{event}之后，我气得发抖，只能先离开，不然真的会爆发。",
        "evidence_en": "shaking with anger",
        "evidence_zh": "气得发抖",
        "rationale_en": "The user describes anger at a near-explosive intensity.",
        "rationale_zh": "用户描述的是接近爆发的强烈愤怒。",
        "intensity": 0.88,
        "secondary": ["angry"],
    },
    "nostalgic": {
        "en": "After {event}, I suddenly missed how that old season of my life used to feel.",
        "zh": "{event}之后，我突然很想念以前那段日子的感觉。",
        "evidence_en": "missed how that old season",
        "evidence_zh": "想念以前那段日子",
        "rationale_en": "The user longs for a past period.",
        "rationale_zh": "用户在怀念过去的一段时光。",
        "intensity": 0.59,
        "secondary": ["sentimental"],
    },
    "jealous": {
        "en": "After {event}, I hated that I kept comparing their news to where I am.",
        "zh": "{event}之后，我很讨厌自己一直拿他们的消息和自己比较。",
        "evidence_en": "comparing their news to where I am",
        "evidence_zh": "拿他们的消息和自己比较",
        "rationale_en": "The user compares themselves unfavorably to someone else.",
        "rationale_zh": "用户在和他人比较中产生嫉妒。",
        "intensity": 0.64,
        "secondary": ["ashamed"],
    },
    "anticipating": {
        "en": "After {event}, I kept watching the time because I was waiting for what comes next.",
        "zh": "{event}之后，我一直看时间，等着下一步发生。",
        "evidence_en": "waiting for what comes next",
        "evidence_zh": "等着下一步",
        "rationale_en": "The user is focused on an expected future event.",
        "rationale_zh": "用户关注即将发生的事。",
        "intensity": 0.57,
        "secondary": ["hopeful"],
    },
    "embarrassed": {
        "en": "After {event}, my face got hot because everyone had seen the mistake.",
        "zh": "{event}之后，我脸一下子热了，因为大家都看见了那个错误。",
        "evidence_en": "everyone had seen the mistake",
        "evidence_zh": "大家都看见了那个错误",
        "rationale_en": "The user feels exposed in a social mistake.",
        "rationale_zh": "用户因社交场合中的失误而尴尬。",
        "intensity": 0.65,
        "secondary": ["ashamed"],
    },
    "content": {
        "en": "After {event}, I felt quietly okay with where things landed.",
        "zh": "{event}之后，我对现在的结果有种安稳的接受。",
        "evidence_en": "quietly okay",
        "evidence_zh": "安稳的接受",
        "rationale_en": "The user describes calm satisfaction.",
        "rationale_zh": "用户表达平静的满足。",
        "intensity": 0.52,
        "secondary": ["grateful"],
    },
    "devastated": {
        "en": "After {event}, it felt like the floor dropped out from under me.",
        "zh": "{event}之后，我像脚下一空，整个人都塌了。",
        "evidence_en": "floor dropped out",
        "evidence_zh": "整个人都塌了",
        "rationale_en": "The user describes overwhelming emotional collapse.",
        "rationale_zh": "用户描述了强烈的情绪崩塌。",
        "intensity": 0.91,
        "secondary": ["sad"],
    },
    "sentimental": {
        "en": "After {event}, I held onto the small details because they suddenly meant so much.",
        "zh": "{event}之后，我反复想着那些小细节，突然觉得它们很珍贵。",
        "evidence_en": "small details",
        "evidence_zh": "小细节",
        "rationale_en": "The user attaches tender meaning to memory details.",
        "rationale_zh": "用户赋予回忆细节温柔的意义。",
        "intensity": 0.58,
        "secondary": ["nostalgic"],
    },
    "caring": {
        "en": "After {event}, all I wanted was to make sure they did not have to face it alone.",
        "zh": "{event}之后，我只想确认他们不用一个人扛着。",
        "evidence_en": "not have to face it alone",
        "evidence_zh": "不用一个人扛着",
        "rationale_en": "The user focuses on protecting or supporting someone else.",
        "rationale_zh": "用户关注照顾和支持他人。",
        "intensity": 0.63,
        "secondary": ["grateful"],
    },
    "trusting": {
        "en": "After {event}, I felt comfortable letting them handle it because they have been steady with me.",
        "zh": "{event}之后，我愿意交给他们处理，因为他们一直很可靠。",
        "evidence_en": "comfortable letting them handle it",
        "evidence_zh": "愿意交给他们处理",
        "rationale_en": "The user expresses confidence in another person.",
        "rationale_zh": "用户表达了对他人的信任。",
        "intensity": 0.60,
        "secondary": ["faithful"],
    },
    "ashamed": {
        "en": "After {event}, I wanted to disappear because it felt like it said something bad about me.",
        "zh": "{event}之后，我只想躲起来，好像这说明我这个人很糟。",
        "evidence_en": "something bad about me",
        "evidence_zh": "我这个人很糟",
        "rationale_en": "The user turns the event into negative self-judgment.",
        "rationale_zh": "用户把事件转化为对自我的负面评价。",
        "intensity": 0.77,
        "secondary": ["guilty"],
    },
    "apprehensive": {
        "en": "After {event}, I was not panicking, but I could not quite relax either.",
        "zh": "{event}之后，我没有慌到失控，但也一直放松不下来。",
        "evidence_en": "could not quite relax",
        "evidence_zh": "放松不下来",
        "rationale_en": "The user shows mild unease rather than strong anxiety.",
        "rationale_zh": "用户表现出轻度不安而非强烈焦虑。",
        "intensity": 0.49,
        "secondary": ["anxious"],
    },
    "faithful": {
        "en": "After {event}, I still believed they would come through, even if it took time.",
        "zh": "{event}之后，我还是相信他们会做到，只是可能需要时间。",
        "evidence_en": "still believed",
        "evidence_zh": "还是相信",
        "rationale_en": "The user holds steady belief despite uncertainty.",
        "rationale_zh": "用户在不确定中保持坚定相信。",
        "intensity": 0.56,
        "secondary": ["trusting"],
    },
}


def main() -> int:
    core = build_core_parallel()
    extended = build_extended_independent()
    challenge = build_challenge()
    formal = [*core, *extended, *challenge]

    errors = validate_records(formal)
    errors.extend(parallel_equivalence_errors(core))
    if errors:
        for error in errors:
            print(error)
        return 1

    write_jsonl(RELEASE_DIR / "core_parallel.jsonl", core)
    write_jsonl(RELEASE_DIR / "extended_independent.jsonl", extended)
    write_jsonl(RELEASE_DIR / "challenge.jsonl", challenge)
    write_jsonl(RELEASE_DIR / "labels.jsonl", [
        {
            "id": record["case_id"],
            "expected": record["expected"],
            "label_provenance": record["label_provenance"],
        }
        for record in formal
    ])
    write_distribution(REPORTS_DIR / "label_distribution.csv", "label", Counter(
        record["expected"] for record in formal
    ))
    write_distribution(REPORTS_DIR / "scenario_distribution.csv", "scenario", Counter(
        record["scenario"] for record in formal
    ))
    write_quality_report(formal)
    print("Generated 500 formal release records")
    return 0


def build_core_parallel() -> list[dict[str, Any]]:
    records = []
    for index in range(1, CORE_PAIR_COUNT + 1):
        label = EMOTION_LABELS[(index - 1) % len(EMOTION_LABELS)]
        scenario, en_event, zh_event = SCENARIOS[(index - 1) % len(SCENARIOS)]
        pair_id = f"core-pair-{index:04d}"
        records.append(make_record(
            case_id=f"core-{index:04d}-en",
            language="en",
            subset="core_parallel",
            label=label,
            scenario=scenario,
            event=en_event,
            index=index,
            pair_id=pair_id,
        ))
        records.append(make_record(
            case_id=f"core-{index:04d}-zh",
            language="zh",
            subset="core_parallel",
            label=label,
            scenario=scenario,
            event=zh_event,
            index=index,
            pair_id=pair_id,
        ))
    return records


def build_extended_independent() -> list[dict[str, Any]]:
    records = []
    for language, offset in [("en", 0), ("zh", EXTENDED_COUNT_PER_LANGUAGE)]:
        for local_index in range(1, EXTENDED_COUNT_PER_LANGUAGE + 1):
            global_index = offset + local_index
            label = EMOTION_LABELS[(global_index - 1) % len(EMOTION_LABELS)]
            scenario, en_event, zh_event = SCENARIOS[(global_index * 3 - 1) % len(SCENARIOS)]
            records.append(make_record(
                case_id=f"extended-{language}-{local_index:04d}",
                language=language,
                subset="extended_independent",
                label=label,
                scenario=scenario,
                event=en_event if language == "en" else zh_event,
                index=global_index,
            ))
    return records


def build_challenge() -> list[dict[str, Any]]:
    records = []
    for language in ["en", "zh"]:
        for index, label in enumerate(EMOTION_LABELS, start=1):
            scenario, en_event, zh_event = SCENARIOS[(index * 5 - 1) % len(SCENARIOS)]
            records.append(make_record(
                case_id=f"challenge-{language}-{index:04d}",
                language=language,
                subset="challenge",
                label=label,
                scenario=f"challenge_{scenario}",
                event=en_event if language == "en" else zh_event,
                index=index,
                challenge=True,
            ))
    return records


def make_record(
    *,
    case_id: str,
    language: str,
    subset: str,
    label: str,
    scenario: str,
    event: str,
    index: int,
    pair_id: str | None = None,
    challenge: bool = False,
) -> dict[str, Any]:
    profile = LABEL_PROFILES[label]
    context_dependency = context_level(index, challenge)
    history = make_history(language, event, index, context_dependency)
    flags = quality_flags(label, index, context_dependency, challenge)
    intensity = min(0.98, max(0.05, profile["intensity"] + intensity_shift(index, language, challenge)))
    record = {
        "case_id": case_id,
        "language": language,
        "subset": subset,
        "target_emotion": label,
        "expected": label,
        "label_provenance": "synthetic_generator_target",
        "secondary_emotions": profile["secondary"][:2],
        "intensity": round(intensity, 2),
        "ambiguity_level": ambiguity_level(index, challenge),
        "scenario": scenario,
        "context_dependency": context_dependency,
        "turn_count": max(1, len(history) // 2 + 1),
        "history": history,
        "current_input": current_input(profile, language, event, challenge),
        "evidence_span": profile[f"evidence_{language}"],
        "rationale": profile[f"rationale_{language}"],
        "quality_flags": flags,
        "annotation_status": "released",
        "source_stage": "release",
    }
    if pair_id is not None:
        record["pair_id"] = pair_id
    return record


def write_quality_report(records: list[dict[str, Any]]) -> None:
    language_counts = Counter(record["language"] for record in records)
    text = f"""# Quality Report

This quality report covers the deterministic synthetic version `0.1.0` formal release and the retained seed reference set.

## Formal Release Checks

- Records: {len(records)}
- Splits: `core_parallel=256`, `extended_independent=180`, `challenge=64`
- Languages: {language_counts['en']} English, {language_counts['zh']} Chinese
- Label coverage: all 32 generator target labels appear in both languages
- Label provenance: `synthetic_generator_target`; `expected` is the generator target, not an independently annotated ground truth label
- Annotation/adjudication files: zero-byte placeholders reserved for future human dual annotation and adjudication
- `annotation_status=released`: packaging state only; it does not assert human review
- Generation command: `python scripts/benchmark/generate_emotion_ablation_v2_release.py`

## Seed Release Checks

- Records: 64
- Languages: 32 English, 32 Chinese
- Label coverage: all 32 labels appear exactly twice
- Parallel seed pairs: 16

Human dual annotation, adjudication, agreement statistics, and rejection-reason reporting remain future work.
"""
    (REPORTS_DIR / "quality_report.md").write_text(text, encoding="utf-8")


def make_history(language: str, event: str, index: int, context_dependency: str) -> list[dict[str, str]]:
    if context_dependency == "none":
        return []
    if language == "en":
        history = [
            {"role": "human", "content": "I am trying to sort out what this reaction means."},
            {"role": "ai", "content": "Tell me the moment that brought it up."},
        ]
        if context_dependency in {"medium", "high"}:
            history.extend([
                {"role": "human", "content": f"It started after {event}."},
                {"role": "ai", "content": "That context helps narrow the feeling."},
            ])
    else:
        history = [
            {"role": "human", "content": "我想弄清楚自己为什么会这样反应。"},
            {"role": "ai", "content": "可以说说是哪一刻触发的吗？"},
        ]
        if context_dependency in {"medium", "high"}:
            history.extend([
                {"role": "human", "content": f"是{event}之后开始的。"},
                {"role": "ai", "content": "这个背景能帮助判断感受。"},
            ])
    if context_dependency == "high" and index % 2 == 0:
        history = history[-2:]
    return history


def current_input(profile: dict[str, Any], language: str, event: str, challenge: bool) -> str:
    text = profile[language].format(event=event)
    if not challenge:
        return text
    if language == "en":
        return f"I keep saying it is fine, but {text[0].lower()}{text[1:]}"
    return f"我嘴上说没事，但{text}"


def context_level(index: int, challenge: bool) -> str:
    if challenge:
        return "high" if index % 2 == 0 else "medium"
    return ["none", "low", "medium", "medium", "high"][index % 5]


def ambiguity_level(index: int, challenge: bool) -> str:
    if challenge:
        return "high" if index % 3 else "medium"
    return ["low", "medium", "low", "medium"][index % 4]


def intensity_shift(index: int, language: str, challenge: bool) -> float:
    shift = ((index % 5) - 2) * 0.01
    if language == "zh":
        shift -= 0.02
    if challenge:
        shift += 0.03
    return shift


def quality_flags(label: str, index: int, context_dependency: str, challenge: bool) -> list[str]:
    flags = []
    if context_dependency in {"medium", "high"}:
        flags.append("requires_context")
    if challenge or label in {
        "anxious",
        "apprehensive",
        "afraid",
        "terrified",
        "angry",
        "annoyed",
        "furious",
        "guilty",
        "ashamed",
        "embarrassed",
        "confident",
        "prepared",
        "hopeful",
        "anticipating",
    }:
        flags.append("label_boundary_case")
    if challenge and index % 4 == 0:
        flags.append("mixed_emotion")
    if challenge and index % 5 == 0:
        flags.append("contains_irony")
    if index % 17 == 0:
        flags.append("cultural_specificity")
    return flags


def write_distribution(path: Path, key_name: str, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([key_name, "count"])
        for key, count in sorted(counts.items()):
            writer.writerow([key, count])


if __name__ == "__main__":
    raise SystemExit(main())
