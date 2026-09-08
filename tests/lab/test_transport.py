import httpx
from emotion_lab.llm.transport import HTTPTransport


def test_http_boundary_sends_exact_bytes_and_redacts_credentials():
    secret = "fixture-api-secret"
    transport = HTTPTransport(
        {"base_url": "https://example.invalid/v1", "timeout_seconds": 1}, secret
    )
    transport.client.close()
    body = b'{"model":"fixture-model","messages":[]}'

    def server(request):
        assert request.content == body
        assert request.headers["authorization"] == "Bearer " + secret
        return httpx.Response(
            200,
            headers={"x-request-id": "fixture-request", "set-cookie": "private-cookie"},
            content=b'{"model":"fixture-model","error":{"api_key":"fixture-api-secret"},"usage":{"prompt_tokens":3}}',
        )

    transport.client = httpx.Client(transport=httpx.MockTransport(server))
    try:
        response = transport(body)
        assert secret.encode() not in response.body
        assert "set-cookie" not in response.headers
        assert response.headers["x-request-id"] == "fixture-request"
        assert b'"prompt_tokens":3' in response.body
    finally:
        transport.close()
