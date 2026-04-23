#!/usr/bin/env python3

import argparse
import json
import sys
import urllib.error
import urllib.request

from rich.console import Console
from rich.text import Text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to /v1/completions and print prompt-token logprobs."
    )
    parser.add_argument(
        "prompt",
        help="Prompt text to score",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080/v1/completions",
        help="Completions endpoint URL",
    )
    parser.add_argument(
        "--model",
        default="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
        help="Model name/id served by llama-server",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=5,
        help="Top-k alternatives per token",
    )
    parser.add_argument(
        "--api-key",
        default="no-key",
        help="Bearer token for OAI-compatible auth",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic generation",
    )
    return parser.parse_args()


def pick_actual_token(entry: dict[str, dict], token_text: str) -> tuple[str, dict] | tuple[None, None]:
    for tok_id, info in entry.items():
        if info.get("decoded_token") == token_text:
            return tok_id, info

    best_tok_id = None
    best_info = None
    for tok_id, info in entry.items():
        rank = info.get("rank")
        if best_info is None or (isinstance(rank, int) and rank < best_info.get("rank", 10**9)):
            best_tok_id = tok_id
            best_info = info

    return best_tok_id, best_info


def lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def color_for_logprob(logprob: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 1.0
    else:
        t = (logprob - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))

    red = (180, 32, 32)
    yellow = (245, 200, 66)
    green = (35, 150, 95)

    if t < 0.5:
        tt = t / 0.5
        r = lerp(red[0], yellow[0], tt)
        g = lerp(red[1], yellow[1], tt)
        b = lerp(red[2], yellow[2], tt)
    else:
        tt = (t - 0.5) / 0.5
        r = lerp(yellow[0], green[0], tt)
        g = lerp(yellow[1], green[1], tt)
        b = lerp(yellow[2], green[2], tt)

    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> int:
    args = parse_args()
    console = Console()

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "echo": True,
        "stream": False,
        "logprobs": args.logprobs,
        "seed": args.seed,
        "cache_prompt": False,
    }

    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    choice = data["choices"][0]
    prompt_logprobs = choice.get("prompt_logprobs")
    all_tokens = choice["logprobs"]["tokens"]
    n_prompt = data["usage"]["prompt_tokens"]

    if not prompt_logprobs:
        print("No prompt_logprobs found. Ensure echo=true and logprobs>0.", file=sys.stderr)
        return 1

    print(f"Prompt token scores for {n_prompt} tokens:\n")

    token_rows: list[tuple[int, str, str | None, float | None, int | None]] = []

    for i in range(n_prompt):
        token_text = all_tokens[i]
        entry = prompt_logprobs[i]

        if entry is None:
            token_rows.append((i, token_text, None, None, None))
            continue

        tok_id, info = pick_actual_token(entry, token_text)
        if info is None:
            token_rows.append((i, token_text, None, None, None))
            continue

        token_rows.append((i, token_text, tok_id, info["logprob"], info["rank"]))

    values = [row[3] for row in token_rows if row[3] is not None]
    lo = min(values) if values else -1.0
    hi = max(values) if values else 0.0

    heat = Text()
    for _, token_text, _, logprob, _ in token_rows:
        token_piece = token_text if token_text else "<EMPTY>"
        if logprob is None:
            heat.append(token_piece, style="black on #a3a3a3")
        else:
            bg = color_for_logprob(logprob, lo, hi)
            heat.append(token_piece, style=f"black on {bg}")

    console.print("[bold]Token heatmap (prompt):[/bold]")
    console.print(heat)
    console.print()

    for i, token_text, tok_id, logprob, rank in token_rows:
        shown_token = token_text if token_text else "<EMPTY>"
        if logprob is None:
            reason = "first prompt token" if i == 0 else "missing from server"
            print(f"{i:>3}: token={shown_token!r} logprob=None ({reason})")
            continue
        print(f"{i:>3}: token={shown_token!r} id={tok_id} logprob={logprob:.6f} rank={rank}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
