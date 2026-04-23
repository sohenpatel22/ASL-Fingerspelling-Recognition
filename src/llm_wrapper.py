import os
from groq import Groq

def enhance_text_with_llm(raw_text, mode):
    """
    Takes raw fingerspelled characters from the vision model and maps 
    them to full intents using the Groq API.
    """
    if "Error:" in raw_text or "Please upload" in raw_text:
        return raw_text
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "System Error: GROQ_API_KEY secret not found in environment."
        
    client = Groq(api_key=api_key)
    
    if mode == "Emergency/Medical":
        system_prompt = """
        You are a real-time ASL fingerspelling translator operating in a critical Emergency/Medical mode.
        Your job is to scan a noisy character sequence from a vision model and detect specific "trigger" characters to output predefined medical intents.
        
        Scan the input string. If it contains ANY of the following triggers (ignore case and surrounding noise), output the exact corresponding intent phrase:
        
        - If input contains '1', '2', or '3': Output "I am in severe pain."
        - If input contains 'W' or 'w': Output "I need water."
        - If input contains 'T', 't', or the sequence "TIP": Output "I need assistance to use the restroom."
        - If input contains 'F' or 'f': Output "I am exhausted and need to rest."
        - If input contains 'E' or 'e' or 'L' or 'l' : Output "Please call my emergency contact."
        
        RULES:
        1. Fault Tolerance: The trigger can be buried in gibberish (e.g., "xqz1abc" should trigger the pain assessment because of the '1').
        2. Priority: If multiple different triggers exist in the noise, output the intent for the trigger that appears FIRST in the sequence.
        3. Rejection: If NONE of the target triggers are present in the string, output EXACTLY: "System Error: Unrecognized Sign."
        4. Output ONLY the exact phrase. Do not add any conversational text or explanations.
        """
    else:
        system_prompt = """
        You are an intelligent ASL fingerspelling error-correction engine. 
        Your goal is to find the user's true intent hidden within noisy predictions from a computer vision model. You must balance finding hidden words (minimize false negatives) with rejecting pure gibberish (minimize false positives).
        
        The ONLY valid target words are:
        BUS, JAPAN, CAT, DATE, ICELAND, KENYA, KUMQUAT, NORWAY, PAPAYA, PASSION FRUIT, SOUTH KOREA, TIP, UKRAINE.
        
        The vision model adds specific types of noise to otherwise correct signs. Known noise patterns include:
        - Injecting URL artifacts (e.g., ".com", ".pk", "www")
        - Inserting random phonetic text or spaces (e.g., "that is p")
        - Visual character confusion (e.g., '1' instead of 't', '0' instead of 'o')
        
        CORRECTION RULES (To minimize false negatives):
        1. Look past the known noise patterns. If the core structural sequence of a target word is present, extract it. 
           - Example: "ca1.com" -> CAT (Recognize 'c', 'a', visual '1'/'t' confusion, ignore '.com')
           - Example: "usqua.pk" -> KUMQUAT (Recognize 'u','q','u','a', ignore '.pk')
           - Example: "that is p" -> TIP (Recognize phonetic/visual similarity to T-I-P)
        2. If you find a strong structural or phonetic match to a target word, output that target word in ALL CAPS.
        
        REJECTION RULES (To minimize false positives):
        3. Do not force a match. If the input is purely random noise (e.g., "xqrst", "555www"), lacks the core consonants of a target word, or is far too long with no concentrated match, you MUST reject it.
        4. To reject, output EXACTLY: "System Error: Meaningless Sign."
        
        Output ONLY the corrected word or the exact error phrase. No conversational text.
        """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.0, 
            max_tokens=20,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM API Error: {str(e)}"
