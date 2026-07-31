#!/usr/bin/env python3
"""Send one minimal non-result API probe for each v5 model profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx
from openai import OpenAI
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", action="append", type=Path, required=True
    )
    parser.add_argument("--execute-api", action="store_true")
    args = parser.parse_args()
    profiles = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in args.model_config
    ]
    print(json.dumps({
        "mode": "execute" if args.execute_api else "dry-run",
        "models": [
            profile["generation"]["model"] for profile in profiles
        ],
        "requests": len(profiles),
        "writes_results": False,
    }, indent=2))
    if not args.execute_api:
        return
    base = os.path.expandvars(os.environ.get("API_BASE_URL", "")).strip()
    key = os.path.expandvars(os.environ.get("API_KEY", "")).strip()
    if not base.startswith(("http://", "https://")) or not key:
        raise RuntimeError("Resolved API_BASE_URL and API_KEY are required")
    client = OpenAI(
        base_url=base,
        api_key=key,
        timeout=60.0,
        http_client=httpx.Client(trust_env=False),
    )
    for profile in profiles:
        generation = profile["generation"]
        kwargs = {
            "model": generation["model"],
            "messages": [{
                "role": "user",
                "content": "Reply with exactly the single token OK.",
            }],
            "temperature": 0.0,
            "max_tokens": 128,
        }
        # The paid v5 configs explicitly disable provider-side thinking.  The
        # reusable model profiles do not repeat that dataset-level setting, so
        # the probe must exercise the same chat-template mode as the real run.
        kwargs["extra_body"] = generation.get("extra_body") or {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        response = client.chat.completions.create(
            **kwargs,
        )
        message = response.choices[0].message
        content = str(message.content or "").strip()
        extra = getattr(message, "model_extra", None) or {}
        reasoning = str(
            getattr(message, "reasoning_content", None)
            or extra.get("reasoning_content")
            or getattr(message, "reasoning", None)
            or extra.get("reasoning")
            or ""
        ).strip()
        if "OK" not in content.upper() and "OK" not in reasoning.upper():
            raise RuntimeError(
                f"Unexpected probe response for "
                f"{profile['experiment']['model_slug']}"
            )
        print(json.dumps({
            "model_slug": profile["experiment"]["model_slug"],
            "status": "ok",
        }))


if __name__ == "__main__":
    main()
