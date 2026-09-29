from __future__ import annotations

import os

MODES = ("None", "Basic", "Emergency/Medical")

_BASIC_PROMPT = """You are an ASL fingerspelling error-correction engine. Find the user's true
intent in a noisy character sequence from a vision model.

The ONLY valid target words are: BUS, JAPAN, CAT, DATE, ICELAND, KENYA, KUMQUAT, NORWAY, PAPAYA,
PASSION FRUIT, SOUTH KOREA, TIP, UKRAINE.

Known noise: injected URL artefacts (".com", ".pk", "www"), inserted phonetic text or spaces
("that is p"), and visual confusions ('1' for 't', '0' for 'o').

If the core structure of a target word is present, output that word in ALL CAPS. If the input is
random noise, lacks the core consonants of any target word, or is too long with no concentrated
match, output EXACTLY: "System Error: Meaningless Sign."
Output ONLY the word or the exact error phrase."""

_EMERGENCY_PROMPT = """You are a real-time ASL fingerspelling translator in Emergency/Medical mode.
Scan a noisy character sequence and output a predefined intent for the FIRST trigger found
(ignore case and surrounding noise):
- '1', '2' or '3': "I am in severe pain."
- 'w': "I need water."
- 't' or "tip": "I need assistance to use the restroom."
- 'f': "I am exhausted and need to rest."
- 'e' or 'l': "Please call my emergency contact."
If no trigger is present output EXACTLY: "System Error: Unrecognized Sign."
Output ONLY the exact phrase."""


def enhance_text_with_llm(raw_text: str, mode: str) -> str:
    if mode == "None" or not raw_text:
        return raw_text
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "System Error: GROQ_API_KEY is not set."
    from groq import Groq

    prompt = _EMERGENCY_PROMPT if mode == "Emergency/Medical" else _BASIC_PROMPT
    try:
        response = Groq(api_key=api_key).chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": raw_text}],
            temperature=0.0,
            max_tokens=20,
        )
    except Exception as err:
        return f"LLM API Error: {err}"
    return response.choices[0].message.content.strip()
