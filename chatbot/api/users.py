from __future__ import annotations

from fastapi import APIRouter, Request

from chatbot.api.schemas import ResolveUserRequest, ResolveUserResponse


router = APIRouter(prefix="/api/users", tags=["users"])


@router.post("/resolve", response_model=ResolveUserResponse)
async def resolve_user(
    payload: ResolveUserRequest, request: Request
) -> ResolveUserResponse:
    resolved = await request.app.state.identity.resolve(payload.identifier)
    return ResolveUserResponse(
        user={"id": resolved.user.id, "identifier": resolved.user.identifier},
        conversation={
            "id": resolved.conversation.id,
            "title": resolved.conversation.title,
        },
    )
