"""`conversation_local_agent_profiles` limits which agents run on the host."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config
from openhands.agent_server.persistence import ServerStores
from openhands.sdk import LLM
from openhands.sdk.profiles import OpenHandsAgentProfile


def _client(tmp_path, allowed: list[str] | None) -> TestClient:
    config = Config(
        static_files_path=None,
        session_api_keys=[],
        secret_key=SecretStr("key"),
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspace",
        conversation_local_agent_profiles=allowed,
    )
    (tmp_path / "workspace").mkdir(exist_ok=True)
    stores = ServerStores.from_config(Config())
    stores.llm_profiles.save(
        "default",
        LLM(model="test-model", api_key=SecretStr("sk-test")),
        include_secrets=True,
        cipher=config.cipher,
    )
    return TestClient(create_app(config))


@pytest.fixture
def profiles(tmp_path, monkeypatch) -> Iterator[dict[str, str]]:
    monkeypatch.setenv("OH_PERSISTENCE_DIR", str(tmp_path / "persistence"))
    ids = {}
    for name in ("coordinator", "general"):
        profile = OpenHandsAgentProfile(name=name, llm_profile_ref="default")
        ServerStores.from_config(Config()).agent_profiles.save(profile)
        ids[name] = str(profile.id)
    yield ids


def _start(client: TestClient, workspace: str, **source) -> tuple[int, dict]:
    response = client.post(
        "/api/conversations",
        json={"workspace": {"working_dir": workspace}, **source},
    )
    return response.status_code, response.json()


def test_any_agent_starts_when_the_list_is_unset(tmp_path, profiles):
    with _client(tmp_path, None) as client:
        status, _ = _start(
            client, str(tmp_path / "workspace"), agent_profile_id=profiles["general"]
        )
    assert status == 201


def test_only_listed_profiles_start_on_the_host(tmp_path, profiles):
    workspace = str(tmp_path / "workspace")
    with _client(tmp_path, ["coordinator"]) as client:
        allowed, info = _start(
            client, workspace, agent_profile_id=profiles["coordinator"]
        )
        refused, detail = _start(
            client, workspace, agent_profile_id=profiles["general"]
        )
        inline, _ = _start(
            client,
            workspace,
            agent={"llm": {"model": "test-model", "api_key": "sk"}, "tools": []},
        )
        # A conversation that was admitted keeps starting (resume).
        resumed, _ = _start(
            client,
            workspace,
            conversation_id=info["id"],
            agent_profile_id=profiles["general"],
        )

    assert allowed == 201
    assert refused == 422
    assert "coordinator" in detail["detail"]
    assert inline == 422
    assert resumed in (200, 201)


def test_the_list_loads_from_the_environment(monkeypatch, tmp_path):
    from openhands.agent_server.config import load_config

    monkeypatch.setenv("OH_CONVERSATION_LOCAL_AGENT_PROFILES", '["coordinator"]')
    monkeypatch.setenv("OH_SECRET_KEY", "key")

    config = load_config(tmp_path / "missing.json")

    assert config.conversation_local_agent_profiles == ["coordinator"]
