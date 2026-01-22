import re
from typing import List, Dict


def clean_text(t: str) -> str:
    # normalize whitespace and weird bullets
    t = t.replace("○", "-")
    t = " ".join(t.split())
    return t


def parse_policy_actions(policy_chunk: str) -> List[Dict[str, str]]:
    """
    Extract structured actions from RetentionPolicy chunk text.

    Output format:
    [
      {"title": "...", "details": "...", "eligibility": "..."},
      ...
    ]
    """
    if not policy_chunk:
        return []

    text = clean_text(policy_chunk)

    # focus after "Recommended actions" if available
    marker = "Recommended actions:"
    if marker.lower() in text.lower():
        idx = text.lower().find(marker.lower())
        text = text[idx + len(marker):].strip()

    # split by numbered actions: "1.", "2.", ...
    # keep the number markers by using regex split
    parts = re.split(r"(\d\.\s)", text)

    actions = []
    current_num = None
    buffer = ""

    def flush_action(buf: str):
        buf = buf.strip()
        if not buf:
            return

        title = buf
        eligibility = ""
        details = buf

        # Separate eligibility
        if "Eligibility:" in buf:
            details, eligibility = buf.split("Eligibility:", 1)
            details = details.strip(" -")
            eligibility = eligibility.strip()

        # Determine title from details
        # Better title extraction (first 6-10 words)
        title_candidate = details.split("-")[0].strip()
        title_candidate = re.sub(r"\s+", " ", title_candidate)

        # If title is too short or truncated, fallback to first 10 words
        if len(title_candidate) < 10:
            title_candidate = " ".join(details.split()[:10])

        title = title_candidate.strip()

        actions.append({
            "title": title.strip(),
            "details": details.strip(),
            "eligibility": eligibility.strip()
        })

    # rebuild based on split tokens
    for p in parts:
        if re.fullmatch(r"\d\.\s", p or ""):
            # flush previous
            flush_action(buffer)
            buffer = ""
            current_num = p.strip()
        else:
            buffer += " " + (p or "")

    flush_action(buffer)

    # post-clean: remove empty actions
    actions = [a for a in actions if a["details"]]

    return actions

