from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from deeptutor.commercial.entitlement_context import (
    CommercialAccess,
    reset_commercial_access,
    set_commercial_access,
)
from deeptutor.commercial.memory import InMemoryCommercialRepository
from deeptutor.commercial.models import ResolvedEntitlements, SubscriptionStatus
from deeptutor.commercial.service import CommercialControlPlane
from deeptutor.multi_user.context import reset_current_user, set_current_user
from deeptutor.multi_user.models import CurrentUser, UserScope


async def _commercial_runtime(
    *,
    owner_id: str = "u-metered",
    quotas: dict[str, int] | None = None,
):
    repository = InMemoryCommercialRepository()
    control_plane = CommercialControlPlane(repository)
    customer = await control_plane.ensure_billing_customer(owner_id=owner_id)
    quota_values = quotas or {
        "llm_tokens": 100_000,
        "embedding_tokens": 100_000,
        "mineru_pages": 100,
    }
    entitlements = {
        f"quota.{meter}": {"limit": limit, "window": "subscription"}
        for meter, limit in quota_values.items()
    }
    entitlements["limits.mineru_max_pages_per_file"] = 10
    plan = await control_plane.publish_plan_version(
        plan_code="metered-test",
        version=1,
        name="Metered test",
        price_minor=0,
        currency="CNY",
        entitlements=entitlements,
    )
    subscription = await control_plane.ensure_trial(customer.id, plan.id)
    resolved = await control_plane.resolve_entitlements(customer.id)
    assert resolved.subscription_id == subscription.id
    runtime = SimpleNamespace(
        settings=SimpleNamespace(enabled=True, database_url=None),
        control_plane=control_plane,
        repository=repository,
    )
    return runtime, control_plane, customer, resolved


@contextmanager
def _request_context(tmp_path: Path, resolved: ResolvedEntitlements, *, role: str = "user"):
    user = CurrentUser(
        id="u-metered" if role == "user" else "u-admin",
        username="metered@example.com",
        role=role,
        scope=UserScope(
            kind="user" if role == "user" else "admin",
            user_id="u-metered" if role == "user" else "u-admin",
            root=tmp_path,
        ),
    )
    user_token = set_current_user(user)
    access_token = set_commercial_access(
        CommercialAccess(owner_id=user.id, resolved=resolved) if role == "user" else None
    )
    try:
        yield
    finally:
        reset_commercial_access(access_token)
        reset_current_user(user_token)


@pytest.mark.asyncio
async def test_metering_reserves_and_finalizes_unpriced_usage(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import metering

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)

    with _request_context(tmp_path, resolved):
        lease = await metering.reserve_commercial_usage(
            meter="llm_tokens",
            quantity=20,
            source="platform",
            provider="openai",
            model="gpt-test",
            request_id="turn-1/call-1",
        )
        assert lease is not None
        await lease.finalize(
            7,
            usage_units={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            is_estimated=False,
        )
        await lease.finalize(7, usage_units={"total_tokens": 7}, is_estimated=False)

    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(events) == 1
    event = events[0]
    assert event.meter == "llm_tokens"
    assert event.quantity == 7
    assert event.provider == "openai"
    assert event.model == "gpt-test"
    assert dict(event.usage_units) == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    assert event.price_version == "unpriced-v1"
    assert event.cost_micros == 0
    assert event.is_estimated is False


@pytest.mark.asyncio
async def test_metering_finalizes_with_catalog_price(tmp_path, monkeypatch) -> None:
    from datetime import timedelta

    from deeptutor.commercial import PriceEntryDraft, metering

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    await control_plane.publish_price_version(
        price_version="prices-test",
        kind="platform_cost",
        effective_from=datetime.now(timezone.utc) - timedelta(days=1),
        entries=[
            PriceEntryDraft(
                meter="llm_tokens",
                model_tier="default",
                unit_price_micros=2,
                currency="CNY",
            )
        ],
    )
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)

    with _request_context(tmp_path, resolved):
        lease = await metering.reserve_commercial_usage(
            meter="llm_tokens",
            quantity=20,
            source="platform",
            provider="openai",
            model="gpt-test",
            request_id="priced-1/call-1",
        )
        assert lease is not None
        await lease.finalize(
            7,
            usage_units={"total_tokens": 7},
            is_estimated=False,
        )

    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(events) == 1
    assert events[0].price_version == "prices-test"
    assert events[0].cost_micros == 14  # 7 tokens * 2 micros
    assert events[0].metadata["price_version"] == "unpriced-v1"  # reserve snapshot


@pytest.mark.asyncio
async def test_metering_bypasses_disabled_admin_and_byok(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import metering

    runtime, _control_plane, _customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)

    with _request_context(tmp_path, resolved):
        assert (
            await metering.reserve_commercial_usage(
                meter="llm_tokens",
                quantity=10,
                source="byok",
                provider="openai",
                model="gpt-test",
            )
            is None
        )

    with _request_context(tmp_path, resolved, role="admin"):
        assert (
            await metering.reserve_commercial_usage(
                meter="llm_tokens",
                quantity=10,
                source="platform",
                provider="openai",
                model="gpt-test",
            )
            is None
        )

    runtime.settings.enabled = False
    assert (
        await metering.reserve_commercial_usage(
            meter="llm_tokens",
            quantity=10,
            source="platform",
            provider="openai",
            model="gpt-test",
        )
        is None
    )


@pytest.mark.asyncio
async def test_byok_still_requires_active_commercial_access(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import metering

    runtime, _control_plane, _customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)
    expired = replace(
        resolved,
        status=SubscriptionStatus.EXPIRED,
        active=False,
        valid_until=datetime.now(timezone.utc),
    )

    with _request_context(tmp_path, expired):
        with pytest.raises(metering.CommercialUsageLimitExceeded):
            await metering.reserve_commercial_usage(
                meter="llm_tokens",
                quantity=10,
                source="byok",
                provider="openai",
                model="gpt-test",
            )


@pytest.mark.asyncio
async def test_source_independent_meter_can_count_byok_usage(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import metering

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)

    with _request_context(tmp_path, resolved):
        lease = await metering.reserve_commercial_usage(
            meter="mineru_pages",
            quantity=3,
            source="byok",
            provider="mineru",
            model="v1",
            count_byok=True,
        )
        assert lease is not None
        await lease.finalize(3, usage_units={"pages": 3}, is_estimated=False)

    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(events) == 1
    assert events[0].meter == "mineru_pages"
    assert events[0].quantity == 3
    assert events[0].metadata["source"] == "byok"


@pytest.mark.asyncio
async def test_agentic_stream_finalizes_once_with_provider_usage(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import metering
    from deeptutor.multi_user import token_quota
    from deeptutor.runtime.agentic.client import _TokenQuotaCompletions

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr(token_quota, "reserve_current_user_tokens", lambda **_kwargs: None)

    class FakeStream:
        def __init__(self) -> None:
            self.chunks = iter(
                [
                    SimpleNamespace(choices=[], usage=None),
                    SimpleNamespace(
                        choices=[],
                        usage=SimpleNamespace(
                            prompt_tokens=4,
                            completion_tokens=3,
                            total_tokens=7,
                        ),
                    ),
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        async def create(self, **_kwargs):
            return FakeStream()

    with _request_context(tmp_path, resolved):
        stream = await _TokenQuotaCompletions(
            FakeCompletions(),
            source="platform",
            provider="openai",
            model="gpt-test",
        ).create(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=20,
            stream=True,
        )
        assert len([chunk async for chunk in stream]) == 2
        await stream.close()

    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(events) == 1
    assert events[0].quantity == 7
    assert events[0].is_estimated is False


@pytest.mark.asyncio
async def test_agentic_cancelled_stream_with_output_is_still_metered(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import metering
    from deeptutor.multi_user import token_quota
    from deeptutor.runtime.agentic.client import _TokenQuotaCompletions

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr(token_quota, "reserve_current_user_tokens", lambda **_kwargs: None)

    class FakeStream:
        def __init__(self) -> None:
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="partial answer"))],
                usage=None,
            )

        async def close(self) -> None:
            return None

    class FakeCompletions:
        async def create(self, **_kwargs):
            return FakeStream()

    with _request_context(tmp_path, resolved):
        stream = await _TokenQuotaCompletions(
            FakeCompletions(),
            source="platform",
            provider="openai",
            model="gpt-test",
        ).create(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=20,
            stream=True,
        )
        await stream.__anext__()
        await stream.close()

    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(events) == 1
    assert events[0].quantity > 0
    assert events[0].is_estimated is True


@pytest.mark.asyncio
async def test_embedding_provider_failure_releases_commercial_reservation(
    tmp_path, monkeypatch
) -> None:
    from deeptutor.commercial import metering
    from deeptutor.multi_user import token_quota
    from deeptutor.services.embedding.client import EmbeddingClient
    from deeptutor.services.embedding.config import EmbeddingConfig

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr(token_quota, "reserve_current_user_units", lambda **_kwargs: None)

    client = EmbeddingClient(
        EmbeddingConfig(
            model="embed-test",
            api_key="test",
            base_url="https://example.invalid/v1/embeddings",
            binding="openai",
            provider_name="openai",
            source="platform",
        )
    )

    class FailingAdapter:
        async def embed(self, _request):
            raise RuntimeError("provider failed")

    client.adapter = FailingAdapter()
    with _request_context(tmp_path, resolved):
        with pytest.raises(RuntimeError, match="provider failed"):
            await client.embed(["hello"])

    reservations = await control_plane.repository.list_usage_reservations(customer.id)
    assert len(reservations) == 1
    assert reservations[0].state.value == "released"


@pytest.mark.asyncio
async def test_sync_mineru_boundary_records_exact_pages_before_parse(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import metering
    from deeptutor.multi_user import token_quota
    from deeptutor.services.parsing.engines.mineru import backend
    from deeptutor.services.parsing.engines.mineru.config import MinerUConfig

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr(token_quota, "current_user_resource_quota_policy", lambda _r: None)
    monkeypatch.setattr(backend, "_pdf_page_count", lambda _path: 3)
    monkeypatch.setattr(
        backend,
        "_parse_local",
        lambda _pdf, output, **_kwargs: Path(output) / "parsed",
    )

    with _request_context(tmp_path, resolved):
        result = backend.parse_pdf_to_workdir(
            tmp_path / "sample.pdf",
            tmp_path / "out",
            config=MinerUConfig(mode="local", source="platform"),
        )
    assert result == tmp_path / "out" / "parsed"

    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(events) == 1
    assert events[0].meter == "mineru_pages"
    assert events[0].quantity == 3
    assert dict(events[0].usage_units) == {"pages": 3}
    assert events[0].is_estimated is False


@pytest.mark.asyncio
async def test_byok_mineru_obeys_plan_file_and_total_page_limits(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import entitlement_context, metering
    from deeptutor.services.parsing.engines.mineru import backend
    from deeptutor.services.parsing.engines.mineru.config import MinerUConfig, MinerUError

    runtime, control_plane, customer, resolved = await _commercial_runtime(
        quotas={"mineru_pages": 4}
    )
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr(entitlement_context, "get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr(
        backend,
        "_pdf_page_count",
        lambda path: 11 if "large" in Path(path).stem else 3,
    )
    parse_calls = 0

    def fake_parse(_pdf, output, **_kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return Path(output) / "parsed"

    class ByokLease:
        def finalize(self, _quantity: int, **_kwargs) -> None:
            return None

        def release(self, **_kwargs) -> None:
            return None

    monkeypatch.setattr(backend, "_parse_local", fake_parse)
    monkeypatch.setattr(
        "deeptutor.multi_user.byok_usage.start_byok_usage",
        lambda **_kwargs: ByokLease(),
    )

    with _request_context(tmp_path, resolved):
        with pytest.raises(MinerUError, match="per-file safety limit of 10 pages"):
            backend.parse_pdf_to_workdir(
                tmp_path / "sample-large.pdf",
                tmp_path / "large-out",
                config=MinerUConfig(mode="local", source="byok"),
            )
        result = backend.parse_pdf_to_workdir(
            tmp_path / "sample.pdf",
            tmp_path / "out",
            config=MinerUConfig(mode="local", source="byok"),
        )
        with pytest.raises(MinerUError, match="quota.mineru_pages exceeded"):
            backend.parse_pdf_to_workdir(
                tmp_path / "sample.pdf",
                tmp_path / "out-2",
                config=MinerUConfig(mode="local", source="byok"),
            )

    assert result == tmp_path / "out" / "parsed"
    assert parse_calls == 1
    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(events) == 1
    assert events[0].quantity == 3
    assert events[0].metadata["source"] == "byok"


@pytest.mark.asyncio
async def test_agentic_passes_stable_request_id_and_finalizes_pg_after_legacy_failure(
    monkeypatch,
) -> None:
    from deeptutor.commercial import metering
    from deeptutor.multi_user import token_quota
    from deeptutor.runtime.agentic.client import _TokenQuotaCompletions

    captured: dict[str, object] = {}

    class CommercialLease:
        finalized = False

        async def finalize(self, *_args, **_kwargs) -> None:
            self.finalized = True

        async def release(self) -> None:
            return None

    commercial_lease = CommercialLease()

    async def reserve(**kwargs):
        captured["request_id"] = kwargs.get("request_id")
        return commercial_lease

    class LegacyLease:
        def finalize(self, _actual: int) -> None:
            raise RuntimeError("legacy settle failed")

        def release(self) -> None:
            return None

    class FakeCompletions:
        async def create(self, **kwargs):
            captured["provider_kwargs"] = kwargs
            return SimpleNamespace(usage=SimpleNamespace(total_tokens=2))

    monkeypatch.setattr(metering, "reserve_commercial_usage", reserve)
    monkeypatch.setattr(
        token_quota,
        "reserve_current_user_tokens",
        lambda **_kwargs: LegacyLease(),
    )

    with pytest.raises(RuntimeError, match="legacy settle failed"):
        await _TokenQuotaCompletions(FakeCompletions()).create(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=20,
            stream=False,
            _commercial_request_id="turn-7/llm-call-2",
        )

    assert captured["request_id"] == "turn-7/llm-call-2"
    assert "_commercial_request_id" not in captured["provider_kwargs"]
    assert commercial_lease.finalized is True


@pytest.mark.asyncio
async def test_agentic_releases_pg_after_legacy_release_failure(monkeypatch) -> None:
    from deeptutor.commercial import metering
    from deeptutor.multi_user import token_quota
    from deeptutor.runtime.agentic.client import _TokenQuotaCompletions

    class CommercialLease:
        released = False

        async def finalize(self, *_args, **_kwargs) -> None:
            return None

        async def release(self) -> None:
            self.released = True

    commercial_lease = CommercialLease()

    async def reserve(**_kwargs):
        return commercial_lease

    class LegacyLease:
        def release(self) -> None:
            raise RuntimeError("legacy release failed")

    class FailingCompletions:
        async def create(self, **_kwargs):
            raise RuntimeError("provider failed")

    monkeypatch.setattr(metering, "reserve_commercial_usage", reserve)
    monkeypatch.setattr(
        token_quota,
        "reserve_current_user_tokens",
        lambda **_kwargs: LegacyLease(),
    )

    with pytest.raises(RuntimeError, match="legacy release failed"):
        await _TokenQuotaCompletions(FailingCompletions()).create(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=20,
            stream=False,
        )

    assert commercial_lease.released is True


@pytest.mark.asyncio
async def test_factory_stable_request_id_reuses_reservation_and_stays_private(
    tmp_path,
    monkeypatch,
) -> None:
    from deeptutor.commercial import metering
    from deeptutor.multi_user import token_quota
    from deeptutor.services.llm import factory
    from deeptutor.services.llm.config import LLMConfig
    from deeptutor.services.llm.provider_core.base import LLMResponse

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr(token_quota, "reserve_current_user_tokens", lambda **_kwargs: None)
    config = LLMConfig(
        model="gpt-test",
        api_key="test",
        base_url="https://example.invalid/v1",
        effective_url="https://example.invalid/v1",
        binding="openai",
        provider_name="openai",
        provider_mode="standard",
        source="platform",
    )

    class Provider:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.stream_calls: list[dict[str, object]] = []

        async def chat_with_retry(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResponse(content="ok", usage={"total_tokens": 1})

        async def chat_stream_with_retry(self, **kwargs):
            self.stream_calls.append(kwargs)
            await kwargs["on_content_delta"]("A")
            return LLMResponse(content="A", usage={"total_tokens": 1})

    provider = Provider()
    monkeypatch.setattr(factory, "get_llm_config", lambda: config)
    monkeypatch.setattr(factory, "get_runtime_provider", lambda _config: provider)

    with _request_context(tmp_path, resolved):
        assert (
            await factory.complete(
                "hello",
                max_tokens=8,
                _commercial_request_id="turn-8/title-call",
            )
            == "ok"
        )
        assert (
            await factory.complete(
                "hello",
                max_tokens=8,
                _commercial_request_id="turn-8/title-call",
            )
            == "ok"
        )
        for _attempt in range(2):
            chunks = [
                chunk
                async for chunk in factory.stream(
                    "hello",
                    max_tokens=8,
                    stream_coalesce_seconds=0,
                    _commercial_request_id="turn-8/stream-call",
                )
            ]
            assert chunks == ["A"]

    reservations = await control_plane.repository.list_usage_reservations(customer.id)
    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(reservations) == 2
    assert len(events) == 2
    assert len(provider.calls) == 2
    assert len(provider.stream_calls) == 2
    assert all("_commercial_request_id" not in call for call in provider.calls)
    assert all("_commercial_request_id" not in call for call in provider.stream_calls)


@pytest.mark.asyncio
async def test_embedding_stable_request_id_deduplicates_each_batch(tmp_path, monkeypatch) -> None:
    from deeptutor.commercial import metering
    from deeptutor.multi_user import token_quota
    from deeptutor.services.embedding.client import EmbeddingClient
    from deeptutor.services.embedding.config import EmbeddingConfig

    runtime, control_plane, customer, resolved = await _commercial_runtime()
    monkeypatch.setattr(metering, "get_commercial_runtime", lambda: runtime)
    monkeypatch.setattr(token_quota, "reserve_current_user_units", lambda **_kwargs: None)
    client = EmbeddingClient(
        EmbeddingConfig(
            model="embed-test",
            api_key="test",
            base_url="https://example.invalid/v1/embeddings",
            binding="openai",
            provider_name="openai",
            source="platform",
        )
    )

    class FakeAdapter:
        async def embed(self, _request):
            return SimpleNamespace(embeddings=[[0.1]], usage={"total_tokens": 1})

    client.adapter = FakeAdapter()
    with _request_context(tmp_path, resolved):
        await client.embed(["hello"], request_id="kb-import-9")
        await client.embed(["hello"], request_id="kb-import-9")

    reservations = await control_plane.repository.list_usage_reservations(customer.id)
    events = await control_plane.repository.list_usage_events(customer.id)
    assert len(reservations) == 1
    assert len(events) == 1


@pytest.mark.asyncio
async def test_embedding_settle_and_release_always_reach_pg(monkeypatch) -> None:
    from deeptutor.services.embedding.client import EmbeddingClient
    from deeptutor.services.embedding.config import EmbeddingConfig

    client = EmbeddingClient(
        EmbeddingConfig(
            model="embed-test",
            api_key="test",
            base_url="https://example.invalid/v1/embeddings",
            binding="openai",
            provider_name="openai",
            source="platform",
        )
    )

    class CommercialLease:
        finalized = False
        released = False

        async def finalize(self, *_args, **_kwargs) -> None:
            self.finalized = True

        async def release(self) -> None:
            self.released = True

    class LegacyLease:
        def finalize(self, _actual: int) -> None:
            raise RuntimeError("legacy embedding settle failed")

        def release(self) -> None:
            raise RuntimeError("legacy embedding release failed")

    lease = CommercialLease()

    async def reserve(_quantity: int, *, request_id=None):
        assert request_id == "embed-op/batch-1"
        return lease

    monkeypatch.setattr(client, "_reserve_quota", lambda _quantity: LegacyLease())
    monkeypatch.setattr(client, "_reserve_commercial", reserve)

    class SuccessfulAdapter:
        async def embed(self, _request):
            return SimpleNamespace(embeddings=[[0.1]], usage={"total_tokens": 1})

    client.adapter = SuccessfulAdapter()
    with pytest.raises(RuntimeError, match="legacy embedding settle failed"):
        await client.embed(["hello"], request_id="embed-op")
    assert lease.finalized is True

    class FailingAdapter:
        async def embed(self, _request):
            raise RuntimeError("provider failed")

    client.adapter = FailingAdapter()
    with pytest.raises(RuntimeError, match="legacy embedding release failed"):
        await client.embed(["hello"], request_id="embed-op")
    assert lease.released is True


def test_mineru_request_id_and_pg_settlement_survive_legacy_failure(tmp_path, monkeypatch) -> None:
    from deeptutor.services.parsing.engines.mineru import backend
    from deeptutor.services.parsing.engines.mineru.config import MinerUConfig

    captured: dict[str, object] = {}

    class CommercialLease:
        finalized = False
        released = False

        def finalize(self, *_args, **_kwargs) -> None:
            self.finalized = True

        def release(self) -> None:
            self.released = True

    class LegacyLease:
        def finalize(self, _actual: int) -> None:
            raise RuntimeError("legacy MinerU settle failed")

        def release(self) -> None:
            raise RuntimeError("legacy MinerU release failed")

    commercial_lease = CommercialLease()

    def reserve(_pdf, _config, *, request_id=None):
        captured["request_id"] = request_id
        return LegacyLease(), commercial_lease, 3

    monkeypatch.setattr(backend, "_reserve_mineru_pages", reserve)
    monkeypatch.setattr(
        backend,
        "_parse_local",
        lambda _pdf, output, **_kwargs: Path(output) / "parsed",
    )

    with pytest.raises(RuntimeError, match="legacy MinerU settle failed"):
        backend.parse_pdf_to_workdir(
            tmp_path / "sample.pdf",
            tmp_path / "out",
            config=MinerUConfig(mode="local", source="platform"),
            request_id="upload-42/mineru",
        )
    assert captured["request_id"] == "upload-42/mineru"
    assert commercial_lease.finalized is True

    monkeypatch.setattr(
        backend,
        "_parse_local",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )
    with pytest.raises(RuntimeError, match="legacy MinerU release failed"):
        backend.parse_pdf_to_workdir(
            tmp_path / "sample.pdf",
            tmp_path / "out-2",
            config=MinerUConfig(mode="local", source="platform"),
            request_id="upload-43/mineru",
        )
    assert commercial_lease.released is True
