"""
Research assistant agent.
Setup:  pip install openai ddgs requests beautifulsoup4
Run:    python agent.py   (with $env:API_KEY set to your Groq key)
"""
import json
import os
import time

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import BadRequestError, OpenAI, RateLimitError

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["API_KEY"],
)
MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 8  # max tool-use rounds per question

SYSTEM_PROMPT = (
    "You are a careful research assistant. For each question: "
    "1) run web_search, 2) open at least 2 relevant pages with fetch_page before answering, "
    "3) write a clear summary. Rules: only state facts that appear in the text you retrieved; "
    "do not add details from memory; if something is uncertain or varies by region or community, say so; "
    "do not use tables unless asked. End with a 'Sources' list of full URLs, "
    "including only pages you actually fetched. Keep the summary under 250 words unless asked for more."
)


# ---------- Tools ----------
def web_search(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


def fetch_page(url: str) -> str:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:4000] or "Page had no readable text."
    except Exception as e:
        return f"Could not fetch page: {e}"


TOOLS = {"web_search": web_search, "fetch_page": fetch_page}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns titles, URLs and snippets for the top results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Download a web page and return its readable text (first ~4000 characters).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL to open"}},
                "required": ["url"],
            },
        },
    },
]


# ---------- Agent loop ----------
def call_model(messages):
    """Call the model, retrying a few times on rate limits."""
    for attempt in range(4):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_SCHEMAS
            )
        except RateLimitError:
            wait = 15 * (attempt + 1)
            print(f"  (rate limited, waiting {wait}s...)")
            time.sleep(wait)
        except BadRequestError as e:
            # Model produced a malformed tool call; a retry usually works
            print(f"  (model output was malformed, retrying {attempt + 1}/4...)")
            if attempt == 3:
                raise
            time.sleep(1)
    raise RuntimeError("Model kept failing or was rate limited. Try again in a minute.")


def run_tool(call, seen, on_step=None):
    name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
        if name not in TOOLS:
            return f"Unknown tool: {name}"
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            return "You already made this exact call. Use the earlier result or try something different."
        seen.add(key)
        print(f"  -> {name}({args})")
        if on_step:
            on_step(name, args)
        return TOOLS[name](**args)
    except Exception as e:
        return f"Tool error: {e}"


def research(messages, on_step=None):
    seen = set()  # remembers tool calls made for this question
    for _ in range(MAX_STEPS):
        resp = call_model(messages)
        msg = resp.choices[0].message

        # Store the assistant turn as a plain dict (avoids extra fields causing errors)
        entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in msg.tool_calls
            ]
        messages.append(entry)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            result = run_tool(call, seen, on_step)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

    return "I hit my step limit before finishing. Try a narrower question."


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Research assistant ready. Type a topic or question. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        try:
            answer = research(messages)
        except Exception as e:
            answer = f"Error: {e}"
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
