"""Write a completion marker a sweep runner can leave for Claude to pick up.

Why a file rather than a direct push: a sweep runs as a detached background
process, so it has no channel to Claude's notification tool. It writes a one-line
summary here instead; Claude reads it when the run finishes and sends the
notification (desktop, and phone too once Remote Control is connected).

    from notify import done
    done("wp025", f"{ok}/{total} fits ok, {failed} failed, {mins:.0f}m")
"""
import json
import time
from pathlib import Path

MARKERS = Path(__file__).resolve().parent / ".sweep_status"


def done(name, summary):
    MARKERS.mkdir(exist_ok=True)
    p = MARKERS / f"{name}.json"
    p.write_text(json.dumps({"name": name, "summary": summary,
                             "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print(f"[notify] {name}: {summary}")
    return p
