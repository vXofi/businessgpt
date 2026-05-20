"""One-shot script to seed eval/golden_prompts.json.

Run once from repo root:
    python3 eval/_seed_golden.py

Output: eval/golden_prompts.json with ~600 prompts across 4 categories.
User edits afterwards (remove garbage, add more rap/fact/edge).
"""

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VAL_PATH = REPO / "val.jsonl"
OUT_PATH = REPO / "eval" / "golden_prompts.json"

random.seed(42)


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
            out.append({
                "id": f"chat_{i:04d}",
                "category": "chat",
                "context": context_lines,
            })
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
    chat = chat_prompts_from_val()
    rap = rap_prompts()
    fact = fact_prompts()
    edge = edge_prompts()

    all_prompts = chat + rap + fact + edge

    # Mark 20 random chat prompts as held-out for regression eval
    held_out_count = 20
    held_out_ids = set(random.sample([p["id"] for p in chat], held_out_count))
    for p in all_prompts:
        p["held_out"] = p["id"] in held_out_ids

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(all_prompts)} prompts to {OUT_PATH}")
    print(f"  chat: {len(chat)}  rap: {len(rap)}  fact: {len(fact)}  edge: {len(edge)}")
    print(f"  held_out: {held_out_count}")


if __name__ == "__main__":
    main()
