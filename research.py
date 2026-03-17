import anthropic
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

IDEAS_FILE = Path(__file__).parent / "product-hunt-ideas.md"

SYSTEM_PROMPT = """You are a product researcher who identifies the best solo-developer opportunities 
from Product Hunt each week. You maintain a running ranked list of problems worth solving, 
re-ranked every week by value-to-effort ratio."""

def read_existing_ideas():
    if IDEAS_FILE.exists():
        return IDEAS_FILE.read_text(encoding="utf-8")
    return ""

def extract_text_from_response(response):
    """Pull all text blocks from an Anthropic response."""
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)

def run_research():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    today = datetime.utcnow().strftime("%Y-%m-%d")
    week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    existing_content = read_existing_ideas()
    existing_section = (
        f"Here is the EXISTING ideas file you must merge into:\n\n```\n{existing_content}\n```"
        if existing_content
        else "There is no existing ideas file yet — this is the first run."
    )

    user_prompt = f"""Today is {today}. Your task is to research the latest Product Hunt launches 
(from {week_ago} to {today}), identify the top 5 problems a solo developer could build a solution for, 
merge them with existing tracked ideas, re-rank everything by value-to-effort ratio, 
and output a COMPLETE updated markdown file.

---

## STEP 1 — Search Product Hunt (last 7 days)

Use web search to find Product Hunt launches from {week_ago} to {today}. Try:
- "site:producthunt.com launches {week_ago}"
- "Product Hunt new products this week {today}"
- "top Product Hunt launches March 2026" (adjust month/year as needed)

For each product found, identify: name, tagline, core problem solved, target audience, upvotes.

## STEP 2 — Score problems

For each distinct problem extracted, score 1–10:
- **Prevalence**: How many people have this problem?
- **Build Ease**: How easy for a solo dev to build? (10 = weekend project)

Weekly Score = Prevalence × Build Ease. Pick Top 5.

## STEP 3 — Merge with existing ideas

{existing_section}

Merge rules:
- Existing problem seen again → increment Times Seen, update score:
  New Score = (Old Score × Times Seen + Weekly Score) / (Times Seen + 1)
- New problem → Times Seen = 1, Score = Weekly Score
- Final Score = Merged Score × (1 + 0.1 × Times Seen)
- Keep top 20 ideas max, drop lowest ranked.

## STEP 4 — Output the COMPLETE updated markdown file

Output ONLY the raw markdown content (no code fences around it, no preamble). 
Use this exact structure:

# 🚀 Solo Developer Opportunity Tracker
*Powered by weekly Product Hunt research. Last updated: {today}*

---

## 📊 Master Ranked List (Value-to-Effort)

| Rank | Problem | Final Score | Times Seen | Suggested Approach |
|------|---------|-------------|------------|-------------------|
| 1 | ... | 00.0 | 1 | max 10 words |

---

## 🗓️ This Week's Discoveries ({today})

### 1. [Problem] — Weekly Score: XX
- **Prevalence**: X/10 — rationale
- **Build Ease**: X/10 — rationale  
- **Example products**: product names
- **Solo Dev Angle**: 1–2 sentence idea

[repeat for top 5]

---

## 📅 Weekly History

### Week of {today}
- Added: [new problems]
- Re-ranked: [notable changes]

[Preserve ALL previous weekly history entries below this line unchanged]

---

CONSTRAINTS:
- Do not fabricate data. Only use what you found via search.
- Always output the complete file, not just the diff.
- If search is limited, note it clearly in the file."""

    # Agentic loop: let Claude use web_search until it produces a final answer
    messages = [{"role": "user", "content": user_prompt}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            # Final answer — extract markdown
            final_markdown = extract_text_from_response(response)
            break

        if response.stop_reason == "tool_use":
            # Append assistant message and tool results, then continue
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # web_search results are returned inline by the API — pass them back
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": block.input.get("query", ""),
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason — use whatever we have
            final_markdown = extract_text_from_response(response)
            break

    if not final_markdown.strip():
        raise ValueError("Claude returned an empty response — aborting file write.")

    IDEAS_FILE.write_text(final_markdown, encoding="utf-8")
    print(f"✅ ideas file updated: {IDEAS_FILE}")

if __name__ == "__main__":
    run_research()
