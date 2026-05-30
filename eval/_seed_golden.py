"""Seed eval golden prompt files.

Run from repo root:
    python3 eval/_seed_golden.py

Outputs:
  eval/golden_prompts.json          dense sliding-window pool
  eval/golden_prompts_diverse.json  downsampled pool for labeling/RM/ORPO
"""

import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VAL_PATH = REPO / "val.jsonl"
OUT_PATH = REPO / "eval" / "golden_prompts.json"
DIVERSE_OUT_PATH = REPO / "eval" / "golden_prompts_diverse.json"

random.seed(42)

POISON_RE = re.compile(
    r"\bзел[её]н\w*|ч[её]\s+вы\s+гомики\s+молчите|а\s+ч[её]\s+вы\s+все\s+молчите|молчание\s+знак\s+согласия",
    re.IGNORECASE,
)


def chat_prompts_from_val() -> list[dict]:
    """Each val example -> one golden prompt of category=chat.

    val.jsonl example schema: {"messages": [system, user, assistant]}
    User content is "\\n"-joined context lines. We split back into a list,
    which is what chat() expects when messages_history[0] is a str.
    """
    out = []
    with open(VAL_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            ex = json.loads(line)
            user_msg = next((m for m in ex["messages"] if m["role"] == "user"), None)
            if user_msg is None:
                continue
            context_lines = [l for l in user_msg["content"].split("\n") if l.strip()]
            if len(context_lines) < 2:
                continue
            if any(POISON_RE.search(line) for line in context_lines):
                continue
            out.append({
                "id": f"chat_{i:04d}",
                "category": "chat",
                "context": context_lines,
            })
    return out


def _overlap(a: list[str], b: list[str]) -> float:
    """Containment-style overlap for adjacent sliding windows."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return max(inter / len(sa), inter / len(sb))


def diverse_chat_prompts(chat: list[dict], *, max_per_run: int = 8, threshold: float = 0.85) -> list[dict]:
    """Downsample near-duplicate sliding-window chat prompts.

    Consecutive validation windows often share >85% of lines. Keep evenly spaced
    representatives from each run so labels are closer to independent examples.
    """
    if not chat:
        return []

    runs = []
    cur = [chat[0]]
    for prev, item in zip(chat, chat[1:]):
        if _overlap(prev["context"], item["context"]) >= threshold:
            cur.append(item)
        else:
            runs.append(cur)
            cur = [item]
    runs.append(cur)

    out = []
    for run_idx, run in enumerate(runs, 1):
        if len(run) <= max_per_run:
            picked = run
        else:
            # Even spacing, preserving first and last. Avoid duplicate indices
            # when max_per_run is small relative to run length.
            raw = [round(i * (len(run) - 1) / (max_per_run - 1)) for i in range(max_per_run)]
            indices = []
            for idx in raw:
                if idx not in indices:
                    indices.append(idx)
            picked = [run[i] for i in indices]

        for item in picked:
            item = dict(item)
            item["dense_source_id"] = item["id"]
            item["dense_run"] = run_idx
            item["dense_run_size"] = len(run)
            item["id"] = f"chatd_{len(out) + 1:04d}"
            out.append(item)
    return out


RAP_ARTISTS = [
    "Pharaoh", "Элджей", "Тима Белорусских", "Скриптонит", "Пошлая Молли",
    "ЛСП", "Егор Крид", "Макс Корж", "Каспийский Груз", "Гуф", "АК-47",
    "Yanix", "Slava Marlow", "Oxxxymiron", "MORGENSHTERN", "Lizer", "Kizaru",
    "GONE.Fludd", "Дора", "Мэйби Бэйби", "INSTASAMKA", "Платина",
    "FRIENDLY THUG 52 NGG", "Boulevard Depo", "SALUKI",
]

RAP_TRIGGER_TEMPLATES = [
    "зачитай {artist}",
    "кинь трек {artist}",
    "давай {artist}",
    "спой {artist}",
    "а ну {artist} ебани",
    "{artist} читни",
]

RAP_NAMES = [
    "Некит Русанов", "Егориус", "A. H.", "Александр Блок", "Саня Блок",
    "Булгак", "Влад Блок", "Старый Мельник", "Вован Крюк",
]


def rap_prompts() -> list[dict]:
    out = []
    idx = 1
    for artist in RAP_ARTISTS:
        # One trigger per artist, plus a couple with extra context for 3 artists
        trigger = random.choice(RAP_TRIGGER_TEMPLATES).format(artist=artist)
        name = random.choice(RAP_NAMES)
        out.append({
            "id": f"rap_{idx:02d}",
            "category": "rap_trigger",
            "context": [f"{name}: {trigger}"],
        })
        idx += 1
    # 5 extra: rap trigger embedded in a mini chat context (tests bridging)
    for artist in random.sample(RAP_ARTISTS, 5):
        ctx = [
            f"{random.choice(RAP_NAMES)}: ну чо",
            f"{random.choice(RAP_NAMES)}: скучно",
            f"{random.choice(RAP_NAMES)}: {random.choice(RAP_TRIGGER_TEMPLATES).format(artist=artist)}",
        ]
        out.append({
            "id": f"rap_{idx:02d}",
            "category": "rap_trigger",
            "context": ctx,
        })
        idx += 1
    return out


FACT_PROMPTS = [
    "столица австралии?",
    "сколько будет 2+2",
    "кто написал войну и мир",
    "почему небо синее",
    "на каком языке говорят в бразилии",
    "что такое фотосинтез",
    "объясни теорему пифагора",
    "в каком году распался ссср",
    "какая самая большая планета солнечной системы",
    "сколько костей в человеке",
]


def fact_prompts() -> list[dict]:
    return [
        {
            "id": f"fact_{i:02d}",
            "category": "fact",
            "context": [f"{random.choice(RAP_NAMES)}: {q}"],
        }
        for i, q in enumerate(FACT_PROMPTS, 1)
    ]


EDGE_PROMPTS = [
    ["сосал?"],
    ["+"],
    ["?"],
    ["крaб"],
    ["да"],
    ["как дела"],
    ["Some text in English — respond"],
    [f"{random.choice(RAP_NAMES)}: я тебя ненавижу"],
    [f"{random.choice(RAP_NAMES)}: расскажи анекдот"],
    [f"{random.choice(RAP_NAMES)}: что думаешь про путина"],
]


def edge_prompts() -> list[dict]:
    return [
        {"id": f"edge_{i:02d}", "category": "edge", "context": ctx}
        for i, ctx in enumerate(EDGE_PROMPTS, 1)
    ]


def main():
    if VAL_PATH.is_file():
        chat = chat_prompts_from_val()
        rap = rap_prompts()
        fact = fact_prompts()
        edge = edge_prompts()
    elif OUT_PATH.is_file():
        print(f"{VAL_PATH} missing; deriving diverse set from existing {OUT_PATH}")
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        chat = [p for p in existing if p["category"] == "chat"]
        rap = [p for p in existing if p["category"] == "rap_trigger"]
        fact = [p for p in existing if p["category"] == "fact"]
        edge = [p for p in existing if p["category"] == "edge"]
    else:
        raise FileNotFoundError(f"Neither {VAL_PATH} nor {OUT_PATH} exists")

    all_prompts = chat + rap + fact + edge
    diverse_chat = diverse_chat_prompts(chat)
    diverse_prompts = diverse_chat + rap + fact + edge

    # Mark 20 random chat prompts as held-out for regression eval
    held_out_count = 20
    held_out_ids = set(random.sample([p["id"] for p in chat], held_out_count))
    for p in all_prompts:
        p["held_out"] = p["id"] in held_out_ids

    diverse_held_out_count = min(20, len(diverse_chat))
    diverse_held_out_ids = set(random.sample([p["id"] for p in diverse_chat], diverse_held_out_count))
    for p in diverse_prompts:
        p["held_out"] = p["id"] in diverse_held_out_ids

    if VAL_PATH.is_file():
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_prompts, f, ensure_ascii=False, indent=1)
    with open(DIVERSE_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(diverse_prompts, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(all_prompts)} prompts to {OUT_PATH}")
    print(f"  chat: {len(chat)}  rap: {len(rap)}  fact: {len(fact)}  edge: {len(edge)}")
    print(f"  held_out: {held_out_count}")
    print(f"Wrote {len(diverse_prompts)} prompts to {DIVERSE_OUT_PATH}")
    print(f"  chat: {len(diverse_chat)}  rap: {len(rap)}  fact: {len(fact)}  edge: {len(edge)}")
    print(f"  held_out: {diverse_held_out_count}")


if __name__ == "__main__":
    main()
