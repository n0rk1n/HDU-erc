"""No hidden retries or response extraction before durable raw capture."""

from dataclasses import dataclass
import re

from .redaction import redact_secrets


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict
    body: bytes


def safe_facts(value, secret=""):
    value = redact_secrets(value)
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", value)
        return re.sub(
            r'https?://[^\s<>"\']+',
            lambda m: str(redact_secrets({"url": m[0]})["url"]),
            value,
        )
    if isinstance(value, dict):
        return {k: safe_facts(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_facts(v, secret) for v in value]
    return value


class HTTPTransport:
    def __init__(self, model, key):
        import httpx

        self.key = key
        self.url = model["base_url"].rstrip("/") + "/chat/completions"
        # HTTPX has no automatic status/transport retries here, and no redirects.
        self.client = httpx.Client(
            timeout=model["timeout_seconds"],
            follow_redirects=False,
            transport=httpx.HTTPTransport(retries=0),
        )

    def __call__(self, request):
        response = self.client.post(
            self.url,
            content=request,
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            },
        )
        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower()
            in {
                "x-request-id",
                "request-id",
                "content-type",
                "date",
                "retry-after",
                "openai-processing-ms",
            }
        }
        body = response.content
        # Preserve identical bytes unless credentials actually require redaction.
        text = body.decode("utf-8", errors="replace")
        safe = text.replace(self.key, "[REDACTED]")
        import json

        try:
            raw = json.loads(safe)
            redacted = safe_facts(raw, self.key)
            if redacted != raw:
                from emotion_lab.storage import json_bytes

                body = json_bytes(redacted)
            elif safe != text:
                body = safe.encode()
        except ValueError:
            body = (
                safe_facts(safe, self.key).encode()
                if safe != text or "Bearer " in safe
                else body
            )
        return Response(response.status_code, safe_facts(headers, self.key), body)

    def close(self):
        self.client.close()
