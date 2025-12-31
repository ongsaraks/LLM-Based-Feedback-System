# -*- coding: utf-8 -*-
"""
Survey analyzer for chatlog {turns: [...] } schema

Usage:
  python llm_analyze.py --input logs/chatlog-7ed5d998-9109-45e8-9bd8-dae964efe2c8.json --out survey_results.csv
"""

import os
import re
import csv
import json
import argparse
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# -------------------- Config --------------------
load_dotenv()
TP_API_KEY = os.getenv("TP_API_KEY")
TP_BASE_URL = os.getenv("TP_BASE_URL", "https://api.opentyphoon.ai/v1")

chatLogJSON = "logs/chatlog-f6e18ec6-9c34-4be5-bcdf-a53a6fdd58f5.json"

if not TP_API_KEY:
    raise SystemExit("Missing TP_API_KEY in .env")

client = OpenAI(api_key=TP_API_KEY, base_url=TP_BASE_URL)

SYSTEM_PROMPT = (
    "คุณคือ Data analytics ที่สรุปผลตอบรับจากคนไข้ในโรงพยาบาล "
    "เมื่อได้รับบทสนทนา Q&A ให้สรุปเป็นข้อ ๆ โดยใช้หัวข้อหลักจากคำถาม "
    "แต่ละข้อให้ระบุเนื้อหาสั้น ๆ ที่ผู้ป่วยตอบไว้ในวงเล็บ "
    "จากนั้นให้ให้คะแนนแต่ละข้อในรูปแบบ:\n"
    "**[หัวข้อ]:** [สรุป (ใส่คำตอบในวงเล็บ)]\n"
    "คะแนน = [ตัวเลข 1-5]\n"
    "ห้ามแต่งเติมข้อมูลเอง ใช้เฉพาะข้อมูลที่ได้รับ"
)

FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Q: คุณรู้สึกว่าโรงพยาบาลทำงานโปร่งใสพอหรือไม่\nA:(ใสแจ๋ว)\n"
            "Q: คุณได้รับข้อมูลชัดเจนเรื่องค่าใช้จ่ายไหมคะ\nA:(ชัดแจ๋ว ไม่ซีเรียส รวยอยู่แล้ว)\n"
            "Q: มีเรื่องใดที่คุณอยากให้เปิดเผยชัดเจนขึ้นคะ\nA:(เปิดใจ)"
        )
    },
    {
        "role": "assistant",
        "content": (
            "ความโปร่งใส: ผู้ป่วยรู้สึกว่าโรงพยาบาลมีความโปร่งใสสูง\n"
            "คะแนน = 5\n\n"
            "ค่าใช้จ่าย: ผู้ป่วยได้รับข้อมูลเกี่ยวกับค่าใช้จ่ายอย่างชัดเจนและไม่ติดขัด\n"
            "คะแนน = 4\n\n"
            "สิ่งที่ต้องการให้เปิดเผยเพิ่มเติม: ผู้ป่วยต้องการให้มีการเปิดเผยข้อมูลที่มากขึ้น\n"
            "คะแนน = 3"
        )
    }
]

# -------------------- Turn → QA pairing --------------------
_ROBOT_IGNORE_PAT = re.compile(
    r"(สวัสดี|ขอรบกวน|ขออนุญาต|ขออนุญาติ|ขอบคุณ|เริ่ม|สิ้นสุด|จบเซสชัน)",
    re.IGNORECASE
)

def choose_answer_text(turn: Dict[str, Any]) -> str:
    # Prefer raw capture if present; else fallback to text
    return (turn.get("text_raw") or turn.get("text") or "").strip()

def looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    if not t or _ROBOT_IGNORE_PAT.search(t):
        return False
    # Thai questions often end with particles; also check "?" and common patterns
    return (
        t.endswith("?")
        or "หรือไม่" in t
        or "ไหม" in t
        or "อะไร" in t
        or "อย่างไร" in t
        or "เพิ่มเติม" in t
        or t.startswith("วันนี้ท่านมาใช้บริการอะไร")
        or t.startswith("ท่าน")  # heuristic for your script style
    )

def best_submodule_for(question: str, submodules: List[Dict[str, Any]]) -> Optional[str]:
    # Fuzzy match the robot question to provided submodule questions
    best_name = None
    best_ratio = 0.0
    for sub in submodules or []:
        name = sub.get("name", "")
        for q in sub.get("questions", []):
            ratio = difflib.SequenceMatcher(a=question, b=q).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = name
    # Threshold: keep modest to tolerate typos
    return best_name if best_ratio >= 0.45 else None

def chatlog_turns_to_qa_list(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    From chatlog dict with keys:
      - main_module_name: str
      - submodules: [ { name, questions: [...] }, ... ]
      - turns: [ {who: 'robot'|'patient', text, text_raw?}, ... ]
    Produce [{module, q, a}, ...]
    """
    main_module = data.get("main_module_name") or ""
    submodules = data.get("submodules") or []
    turns = data.get("turns") or []

    qa_list: List[Dict[str, str]] = []
    pending_q: Optional[str] = None
    pending_module: Optional[str] = None

    for t in turns:
        who = t.get("who")
        if who == "robot":
            q_text = (t.get("text") or "").strip()
            if looks_like_question(q_text):
                pending_q = q_text
                # Guess submodule; fall back to main module
                guessed = best_submodule_for(pending_q, submodules)
                pending_module = guessed or main_module or ""
            else:
                # ignore greetings / meta lines
                continue
        elif who == "patient":
            if pending_q:
                a_text = choose_answer_text(t)
                qa_list.append({
                    "module": pending_module or main_module or "",
                    "q": pending_q,
                    "a": a_text
                })
                pending_q, pending_module = None, None
            else:
                # patient spoke without a captured question → skip
                continue

    # If the very first "general" question exists without clear submodule, keep it under "ทั่วไป"
    for qa in qa_list:
        if not qa["module"]:
            qa["module"] = "ทั่วไป"
    return qa_list

# -------------------- LLM, parsing, CSV --------------------
def build_prompt_text(conversation_log: List[Dict[str, str]]) -> str:
    parts = []
    for qa in conversation_log:
        parts.append(f"{qa.get('module','')}\nQ: {qa.get('q','')}\nA: {qa.get('a','')}".strip())
    return "\n".join(parts)

def summarize_and_score(conversation_text: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOT + [
        {"role": "user", "content": conversation_text}
    ]
    resp = client.chat.completions.create(
        model="typhoon-v2.1-12b-instruct",
        messages=messages,
        max_tokens=700,
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()

def parse_for_csv(result_text: str, conversation_log: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows = []
    blocks = re.split(r"\n\s*\n", result_text.strip()) if result_text.strip() else []

    for i, qa in enumerate(conversation_log):
        block = blocks[i] if i < len(blocks) else ""
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

        summary_text = ""
        for ln in lines:
            if ":" in ln:
                summary_text = ln.split(":", 1)[1].strip()
                break

        score = ""
        for ln in lines:
            if "คะแนน" in ln:
                m = re.search(r"(\d+)", ln)
                if m:
                    score = m.group(1)
                    break

        rows.append({
            "module": qa.get("module", ""),
            "questions": qa.get("q", ""),
            "answers": qa.get("a", ""),
            "summary": summary_text,
            "score": score
        })
    return rows

def append_rows_to_csv(rows: List[Dict[str, str]], csv_path: str | Path):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    fieldnames = ["module", "questions", "answers", "summary", "score"]
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

# -------------------- Main flow --------------------
def run(input_path: str | Path, out_csv: str | Path):
    p = Path(input_path)
    if not p.exists():
        raise SystemExit(f"Input JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Prefer the chatlog schema; fallback to generic
    if isinstance(data, dict) and "turns" in data:
        conv = chatlog_turns_to_qa_list(data)
    else:
        # Generic fallback: expect list[dict] with module/q/a
        conv = data if isinstance(data, list) else [data]
        conv = [r for r in conv if isinstance(r, dict) and (r.get("q") or r.get("a"))]

    if not conv:
        raise SystemExit("No Q/A pairs found in the input JSON.")

    # 1) Build prompt
    prompt_text = build_prompt_text(conv)

    # 2) Call model
    result = summarize_and_score(prompt_text)

    # 3) Show and save
    print("\n=== Model Output ===\n")
    print(result)

    rows = parse_for_csv(result, conv)
    append_rows_to_csv(rows, out_csv)
    print(f"\nSaved {len(rows)} row(s) to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i",
                        default=chatLogJSON,
                        help="Path to chatlog JSON")
    parser.add_argument("--out", "-o",
                        default="survey_results.csv",
                        help="CSV file to append results")
    args = parser.parse_args()
    run(args.input, args.out)