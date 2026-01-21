import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """You are a string reversal expert. Your task is to completely reverse the letter order of words.

Here are some examples:

Example 1:
Input: hello
Output: olleh

Example 2:
Input: world
Output: dlrow

Example 3:
Input: python
Output: nohtyp

Example 4:
Input: abc
Output: cba

Example 5:
Input: websocket
Output: tekcosbew

Example 6:
Input: typescript
Output: tpircsepyt

Example 7:
Input: apple
Output: elppa

Example 8:
Input: banana
Output: ananab

Rules:
- Only output the reversed result in lowercase
- Do not output any other text
- Do not add quotation marks"""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)