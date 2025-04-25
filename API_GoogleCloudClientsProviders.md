
# GOOGLE API MODELS
park
Gemini 2.5 Flash Preview 04-17
gemini-2.5-flash-preview-04-17
attach_money
Input/Output API Pricing
Thinking
Non-thinking
(API pricing per 1M tokens, UI remains free of charge)

Input
$0.15
$0.15
Output
$3.50
$0.60
star_rate
Best for
Large scale processing (e.g. multiple pdfs)
Low latency, high volume tasks which require thinking
Agentic use cases
person
Use case
Reason over complex problems
Show the thinking process of the model
Call tools natively
cognition_2
Knowledge cutoff
Jan 2025
timer
Rate limits
1000 RPM
Free
10 RPM 500 req/day

Gemini 2.0 Flash-Lite
gemini-2.0-flash-lite, alias that points to gemini-2.0-flash-lite-001
attach_money
Input/Output API Pricing
All context lengths
(API pricing per 1M tokens, UI remains free of charge)

Input
$0.075
Output
$0.30
star_rate
Best for
Long Context
Realtime streaming
Native tool use
person
Use case
Process 10,000 lines of code
Call tools natively
Stream images and video in realtime
cognition_2
Knowledge cutoff
Aug 2024


timer
Rate limits
4000 RPM

Free
30 RPM 1500 req/day

GEMMA-3 27B
Free, no limits for API

Sample code
```python
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)
```

```python
import os
from google import genai
import re 
# create client
api_key = os.getenv("GEMINI_API_KEY","xxx")
client = genai.Client(api_key=api_key)
 
# speicfy the model id
model_id = "gemma-3-27b-it"
 
# extract the tool call from the response
def extract_tool_call(text):
    import io
    from contextlib import redirect_stdout
 
    pattern = r"```tool_code\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Capture stdout in a string buffer
        f = io.StringIO()
        with redirect_stdout(f):
            result = eval(code)
        output = f.getvalue()
        r = result if output == '' else output
        return f'```tool_output\n{r}\n```'
    return None
```
