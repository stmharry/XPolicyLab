from __future__ import annotations

from unittest import mock

from client_server.ws import model_client


class _FakeClient:
    def __init__(self, config):
        self.config = config

    async def connect(self, *, handshake=True):
        return None

    async def close(self):
        return None


def test_sync_model_client_disables_idle_loop_websocket_keepalive():
    with mock.patch.object(model_client, "PolicyEvalClient", _FakeClient):
        client = model_client.WsModelClient(
            url="ws://127.0.0.1:19000",
            evaluation_id="evaluation",
            trial_id="trial",
        )
    try:
        assert client._client.config.ws_ping_interval_s is None
        assert client._client.config.ws_ping_timeout_s is None
    finally:
        client.close()
