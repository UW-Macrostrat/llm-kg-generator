"""Discover served models and make logged inference calls. No extraction rules."""
import json
import math
import time

from .api import auth, request
from .logging import now


def served_model(context, stage):
    cfg = context.config
    if context.served_models is None:
        response, _ = request(context, "GET", f"{cfg.base_url.rstrip('/')}/models", stage=stage,
                              headers=auth(cfg.llm_api_key))
        value = response.json()
        data = value.get("data") if isinstance(value, dict) else None
        if not isinstance(data, list) or not data or any(not isinstance(m, dict) or not m.get("id") for m in data):
            raise ValueError("LLM /models returned no valid served models")
        context.served_models = data
    name = (cfg.ner_model if stage == "ner" else cfg.re_model) or cfg.model
    selected = next((m for m in context.served_models if m["id"] == name), None) if name else context.served_models[0]
    if selected is None:
        raise ValueError(f"Model {name!r} is not served. Available: {[m['id'] for m in context.served_models]}")
    return selected


def call_llm(context, stage, source_id, messages, schema):
    """Return raw response text and call metadata. Save requests/errors even when inference fails."""
    started = time.monotonic()
    info = {"stage": stage, "source_id": source_id, "started_at": now(), "attempts": 0}
    try:
        model = served_model(context, stage)
        info["model"] = model["id"]
        cfg = context.config
        body = {"model": model["id"], "messages": messages, "temperature": 0, "max_tokens": cfg.max_tokens}
        if cfg.structured_output == "vllm":
            body["structured_outputs"] = {"json": schema}
        elif cfg.structured_output == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": stage, "strict": True, "schema": schema}}
        else:
            body["messages"] = [*messages, {"role": "user", "content": "Return only JSON matching this schema: " + json.dumps(schema)}]
        if cfg.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        context.log.write(f"sources/{source_id}/{stage}_request.json", body)
        # Gateways often omit /tokenize. Record the estimate explicitly; it is NOT an exact token count.
        estimate = 64 + sum(16 + math.ceil(len(m["content"].encode("utf-8")) / 3) for m in body["messages"])
        info["estimated_prompt_tokens"] = estimate
        server_limit = model.get("max_model_len")
        server_limit = server_limit if type(server_limit) is int and server_limit > 0 else None
        if server_limit and cfg.context_length and cfg.context_length > server_limit:
            raise ValueError(f"--context-length exceeds served model limit {server_limit}")
        limit = cfg.context_length or server_limit
        if limit and estimate + cfg.max_tokens + 64 > limit:
            raise ValueError(f"Estimated prompt ({estimate}) + max_tokens ({cfg.max_tokens}) + margin (64) "
                             f"exceeds context {limit}. Use a larger-context served model, smaller source, or "
                             "lower --max-tokens. This simple pipeline does not truncate/chunk paragraphs.")
        response, attempts = request(context, "POST", f"{cfg.base_url.rstrip('/')}/chat/completions",
                                     stage=stage, source_id=source_id, json=body, headers=auth(cfg.llm_api_key),
                                     follow_redirects=False)
        info.update(attempts=attempts, status="response_received")
        context.log.write(f"sources/{source_id}/{stage}_response.json", {"raw_response": response.text})
        return response.text, info
    except Exception as exc:
        info.update(status="failed", error=str(exc), attempts=getattr(exc, "attempts", info["attempts"]))
        if hasattr(exc, "response_excerpt"):
            info["response_excerpt"] = exc.response_excerpt
        exc.call_info = info
        raise
    finally:
        info.update(finished_at=now(), seconds=time.monotonic() - started)
        context.log.write(f"sources/{source_id}/{stage}_call.json", info)
