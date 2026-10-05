#!/usr/bin/env python3
"""Does the Kolibri plugin behave? Checks the reasoning switch, the reasoning/answer split, tool calls and
streaming against a running server, optionally plus needle-in-a-haystack retrieval at long context.

  tools/check.py                              # functional checks (~1 min)
  tools/check.py --needle 128000 400000       # also hide a key in N tokens of filler and ask for it back
                                              # (beyond 262144 needs CONTEXT=1048576)
  tools/check.py --url http://10.0.0.158:8888

Exit code 1 if anything fails. Standard library only.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request


class Server:
    def __init__(self, url: str, model: str):
        self.url, self.model = url.rstrip("/") + "/v1/chat/completions", model

    def _open(self, body: dict):
        data = json.dumps({"model": self.model, **body}).encode()
        req = urllib.request.Request(self.url, data, {"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as err:
            sys.exit(f"{self.url}: HTTP {err.code} {err.read().decode(errors='replace')[:300]}")
        except urllib.error.URLError as err:
            sys.exit(f"{self.url}: {err.reason} (is the server up? docker compose ps)")

    def ask(self, prompt: str, effort: str | None = None, **body) -> dict:
        """One non-streamed turn; returns the message plus usage."""
        if effort:
            body["chat_template_kwargs"] = {"reasoning_effort": effort}
        reply = json.load(self._open({"messages": [{"role": "user", "content": prompt}], **body}))
        return {**reply["choices"][0]["message"], "usage": reply.get("usage", {})}

    def stream(self, prompt: str, effort: str) -> tuple[str, str]:
        """One streamed turn; returns (reasoning, content) as they arrived in the deltas."""
        parts = {"reasoning": [], "content": []}
        body = {"stream": True, "chat_template_kwargs": {"reasoning_effort": effort},
                "messages": [{"role": "user", "content": prompt}]}
        for raw in self._open(body):
            event = raw.decode().removeprefix("data:").strip()
            if not event or event == "[DONE]":
                continue
            for choice in json.loads(event).get("choices", []):
                delta = choice.get("delta", {})
                parts["reasoning"].append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                parts["content"].append(delta.get("content") or "")
        return "".join(parts["reasoning"]), "".join(parts["content"])


def reasoning(msg: dict) -> str:
    return msg.get("reasoning") or msg.get("reasoning_content") or ""


def clean(text: str | None) -> str:
    return (text or "").strip()


# ---------------------------------------------------------------------------------------------------- checks
# Each returns (passed, one-line detail).

def thinking_off_effort(srv: Server):
    msg = srv.ask("Wie heißt die Hauptstadt von Belgien? Nur das Wort.", effort="none", max_tokens=32)
    ok = not reasoning(msg) and any(c in clean(msg["content"]).lower() for c in ("brüssel", "brussel", "bruxelles"))
    return ok, f"answer {clean(msg['content'])!r}, reasoning {len(reasoning(msg))} chars"


def thinking_off_switch(srv: Server):
    msg = srv.ask("What is 17 + 25? Digits only.", max_tokens=32, chat_template_kwargs={"enable_thinking": False})
    return not reasoning(msg) and "42" in clean(msg["content"]), f"answer {clean(msg['content'])!r}"


def effort_levels(srv: Server):
    prompt = "Ein Zug fährt um 9:40 los und kommt um 13:15 an. Wie viele Minuten dauert die Fahrt? Nur die Zahl."
    sizes, answers = {}, {}
    for effort in ("low", "high"):
        msg = srv.ask(prompt, effort=effort, max_tokens=8192)
        sizes[effort], answers[effort] = len(reasoning(msg)), clean(msg["content"])
    ok = all(sizes.values()) and all("215" in a for a in answers.values())
    return ok, f"reasoning low {sizes['low']} / high {sizes['high']} chars, answers {list(answers.values())}"


def tool_call(srv: Server):
    tool = {"type": "function", "function": {
        "name": "create_event",
        "description": "Legt einen Kalendertermin an.",
        "parameters": {"type": "object", "required": ["title", "date", "attendees"], "properties": {
            "title": {"type": "string"},
            "date": {"type": "string", "description": "ISO-Datum, JJJJ-MM-TT"},
            "attendees": {"type": "array", "items": {"type": "string"}}}}}}
    msg = srv.ask("Trag für den 12. März 2027 den Termin 'Sprint Review' mit Anna und Jonas ein.",
                  effort="low", tools=[tool], max_tokens=4096)
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, f"no tool call; content {clean(msg.get('content'))[:120]!r}"
    args = json.loads(calls[0]["function"]["arguments"])
    ok = (calls[0]["function"]["name"] == "create_event" and args.get("date") == "2027-03-12"
          and isinstance(args.get("attendees"), list) and {"Anna", "Jonas"} <= set(args["attendees"]))
    return ok, f"{calls[0]['function']['name']}({json.dumps(args, ensure_ascii=False)})"


def streaming(srv: Server):
    think, answer = srv.stream("Nenne die drei größten Städte Deutschlands, kommagetrennt.", effort="low")
    ok = bool(think) and "Berlin" in answer and "<think>" not in answer + think
    return ok, f"reasoning {len(think)} chars, answer {clean(answer)[:70]!r}"


def needle(srv: Server, tokens: int):
    """Hide a key among ~`tokens` tokens of synthetic service logs, at a random depth, and ask for it."""
    rng = random.Random(tokens)
    key = f"{rng.randrange(10**7, 10**8)}"
    lines = [f"2026-10-05T{rng.randrange(24):02}:{rng.randrange(60):02}:{rng.randrange(60):02}Z "
             f"worker-{rng.randrange(64):02} {rng.choice(('GET', 'POST', 'PUT'))} /api/v2/{rng.choice(('orders', 'users', 'stock', 'invoices'))}/"
             f"{rng.randrange(10**6)} -> {rng.choice((200, 200, 200, 201, 404, 503))} in {rng.randrange(2, 900)} ms"
             for _ in range(tokens // 50)]   # a log line is ~50 tokens
    depth = rng.uniform(0.1, 0.9)
    lines.insert(int(len(lines) * depth), f"NOTICE maintenance key for tonight is {key}")
    start = time.time()
    msg = srv.ask("\n".join(lines) + "\n\nWhat is tonight's maintenance key? Reply with the key only.",
                  effort="none", max_tokens=32, temperature=0)
    got, n = clean(msg["content"]), msg["usage"].get("prompt_tokens", 0)
    return key in got, f"{n:,} tokens, depth {depth:.0%}, {time.time() - start:.0f} s: expected {key}, got {got!r}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("KOLIBRI_URL", "http://127.0.0.1:8888"))
    ap.add_argument("--model", default=os.environ.get("KOLIBRI_MODEL", "Kolibri-1"))
    ap.add_argument("--needle", type=int, nargs="*", default=[], metavar="TOKENS")
    args = ap.parse_args()

    srv = Server(args.url, args.model)
    checks = [(fn.__name__.replace("_", " "), fn) for fn in
              (thinking_off_effort, thinking_off_switch, effort_levels, tool_call, streaming)]
    checks += [(f"needle {n:,}", lambda s, n=n: needle(s, n)) for n in args.needle]
    failures = 0
    for name, check in checks:
        ok, detail = check(srv)
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name:20} {detail}", flush=True)
    sys.exit(failures > 0)


if __name__ == "__main__":
    main()
