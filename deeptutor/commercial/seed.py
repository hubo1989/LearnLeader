"""Versioned, deterministic seed definitions for Commercial Foundation v1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from .service import CommercialControlPlane


@dataclass(frozen=True, slots=True)
class TrialV1ModelBinding:
    """Explicit model catalog snapshot; never resolved from a mutable default."""

    profile_id: str
    model_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("trial model profile_id is required")
        normalized = tuple(model.strip() for model in self.model_ids if model.strip())
        if not normalized:
            raise ValueError("trial model_ids must contain at least one model")
        if len(set(normalized)) != len(normalized):
            raise ValueError("trial model_ids must be unique")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        object.__setattr__(self, "model_ids", normalized)

    @classmethod
    def from_json(cls, raw: str) -> "TrialV1ModelBinding":
        try:
            value = json.loads(raw)
            profile_id = value["profile_id"]
            model_ids = value["model_ids"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                "DEEPTUTOR_COMMERCIAL_TRIAL_LLM_MODELS must be JSON with profile_id and model_ids"
            ) from exc
        if (
            not isinstance(profile_id, str)
            or not isinstance(model_ids, list)
            or not all(isinstance(item, str) for item in model_ids)
        ):
            raise ValueError("DEEPTUTOR_COMMERCIAL_TRIAL_LLM_MODELS has invalid field types")
        return cls(profile_id=profile_id, model_ids=tuple(model_ids))


@dataclass(frozen=True, slots=True)
class PlanSeed:
    plan_code: str
    version: int
    name: str
    price_minor: int
    currency: str
    entitlements: Mapping[str, Any]


def build_trial_v1_plan(model_binding: TrialV1ModelBinding) -> PlanSeed:
    """Build the complete Trial v1 definition from an explicit model snapshot."""
    entitlements: dict[str, Any] = {
        "capability.chat": True,
        "capability.mastery_path": True,
        "capability.deep_solve": True,
        "capability.deep_research": True,
        "capability.deep_question": True,
        "capability.visualize": True,
        "tool.sandbox": False,
        "tool.exec": False,
        "tool.cron": False,
        "feature.partners": False,
        "feature.mcp": False,
        "feature.byok": True,
        "feature.manim": False,
        "limits.kb_count": 2,
        "limits.storage_bytes": 524_288_000,
        "limits.upload_bytes": 26_214_400,
        "limits.concurrent_turns": 1,
        "limits.mineru_max_pages_per_file": 10,
        # Platform-cost budget target: 1 credit = 1000 micros.  Enforced by
        # the reserve-time budget gate once the platform_cost price catalog
        # attributes real costs; with no catalog rows everything stays
        # unpriced at zero and the gate stays open.
        "budget.platform_cost_credits": 2_000,
        "quota.llm_tokens": {"limit": 100_000, "window": "utc_day"},
        "quota.embedding_tokens": {"limit": 500_000, "window": "subscription"},
        "quota.mineru_pages": {"limit": 20, "window": "subscription"},
        "quota.concurrent_turns": {"limit": 1, "window": "subscription"},
        "models.llm": {
            "profile_id": model_binding.profile_id,
            "model_ids": list(model_binding.model_ids),
        },
    }
    return PlanSeed(
        plan_code="trial",
        version=1,
        name="Trial v1",
        price_minor=0,
        currency="CNY",
        entitlements=entitlements,
    )


async def seed_trial_v1(
    service: CommercialControlPlane,
    model_binding: TrialV1ModelBinding,
):
    """Publish Trial v1 idempotently; any model/limit drift requires Trial v2."""
    seed = build_trial_v1_plan(model_binding)
    return await service.publish_plan_version(
        plan_code=seed.plan_code,
        version=seed.version,
        name=seed.name,
        price_minor=seed.price_minor,
        currency=seed.currency,
        entitlements=seed.entitlements,
    )
