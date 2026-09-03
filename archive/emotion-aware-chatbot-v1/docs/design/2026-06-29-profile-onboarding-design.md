# Profile Onboarding Design

Date: 2026-06-29

## Goal

Add a first-run user profile onboarding flow for the local emotional chatbot. When no user profile exists, the app should invite the user to answer a few lightweight questions, ask the LLM to summarize those answers into a profile draft, and save the profile only after the user confirms it.

The user must also be able to open "我的画像" from the chat page at any time to view and manually edit the saved profile.

## Current Context

Runtime data is stored locally in SQLite. Static profile values already use `data/records/runtime.sqlite3` via `RuntimeStore.load_profile()` and `RuntimeStore.replace_profile()`. `chatbot/profile.py` formats that key-value profile into prompt text, and `chatbot/web.py` injects it into the chat chain at service startup.

The frontend currently has a single chat page with session loading, SSE chat streaming, emotion status, feedback controls, and regeneration controls. There is no profile API, first-run prompt, onboarding UI, or profile edit entry.

Long-term memory already extracts some durable facts from chat into `data/records/memory.sqlite3`, but that flow is conservative and implicit. This feature is for an explicit, user-confirmed static profile, not a replacement for long-term memory.

## User Experience

On first page load, the frontend calls a profile API. If the profile is empty, it shows a non-blocking prompt inviting the user to set up their profile. The user can start onboarding or skip and chat immediately.

The chat page header includes a visible "我的画像" button. Clicking it opens a right-side profile panel. The panel supports three states:

- Existing profile: show the five editable profile fields and a save button.
- Empty profile: show "开始录入画像" and "暂时跳过".
- Onboarding: ask one question at a time, then show an editable draft before saving.

The first version asks five lightweight questions. Each question can be skipped:

1. 希望我怎么称呼你？
2. 你现在大概是什么身份或阶段？比如学生、工作、备考、休息调整中。
3. 你希望这个聊天机器人主要怎么陪伴你？
4. 你更喜欢怎样的回应风格？比如简短、温柔、直接、慢慢分析。
5. 有哪些话题、表达方式或建议类型是你希望我避免的？

After the user answers, the backend asks the chat LLM to summarize the answers into a structured draft. The draft is shown in the profile panel as editable fields. Nothing is written to the database until the user clicks confirm/save.

## Profile Fields

Store the first version as the existing key-value profile shape in `profile_entries`. Use fixed keys:

- `preferred_name`
- `life_stage`
- `companion_expectation`
- `response_style`
- `avoidance`

Empty fields are allowed in the draft UI but should not be persisted. The backend should accept only these keys, trim whitespace, reject or clamp unreasonable lengths, and ignore unknown fields.

## Backend Design

Keep profile onboarding separate from `ChatService` so normal chat streaming remains focused on conversation, emotion analysis, memory retrieval, and message persistence.

Add profile API boundaries in the Web layer:

- `GET /api/profile`: returns `{profile, is_empty}`.
- `PUT /api/profile`: accepts confirmed profile fields and overwrites `profile_entries`.
- `GET /api/profile/onboarding/questions`: returns the fixed onboarding questions.
- `POST /api/profile/onboarding/draft`: accepts onboarding answers and returns a profile draft.

The draft endpoint uses the configured chat LLM to summarize answers. The prompt must instruct the model to:

- Output JSON only.
- Use only the allowed field names.
- Preserve only information the user provided.
- Leave a field empty when the answer was skipped or uncertain.
- Avoid inventing sensitive background, diagnoses, emotions, or preferences.

The backend must validate the model output before returning it. If the LLM call fails, returns invalid JSON, or includes unusable fields, return a deterministic rule-based draft from the raw answers instead.

## Prompt Refresh

The current chain includes profile text when the service is built. After a confirmed profile save, the next chat turn should use the updated profile.

The implementation should add a narrow refresh path that rebuilds only the chat chain with `format_profile(load_profile())`. It should not recreate the whole `ChatService`, reset `session_records`, clear emotion state, replace the memory provider, or lose current runtime counters.

## Frontend Design

Add a header-level "我的画像" button to the existing chat page. Keep the UI lightweight and consistent with the current static frontend. The profile editor should be a right-side panel on desktop and a full-width sheet on narrow mobile screens.

The profile panel should:

- Load profile data when opened.
- Show loading, error, empty, onboarding, draft, and edit states.
- Keep user edits in place when save fails.
- Let the user cancel or close without saving.
- Save only when the user confirms.

The first-run prompt should not disable the main chat input. Skipping onboarding closes the prompt for the current page session only; it does not write a profile marker to SQLite.

## Error Handling

Profile fetch failure should not block chat. The profile panel should show a retry affordance.

Draft generation failure should fall back to rule-based draft generation on the backend. If the frontend still receives an error, it should keep the user's answers and allow manual field editing.

Save failure should keep the panel open, preserve edited content, and show a retry message.

Invalid or excessive profile content should be rejected or normalized by the backend before persistence. Unknown fields should never be saved.

## Testing

Add backend tests for:

- `GET /api/profile` returning profile data and `is_empty`.
- `PUT /api/profile` saving only allowed, non-empty fields.
- Onboarding questions returning the fixed question list.
- Draft generation parsing valid LLM JSON.
- Draft generation falling back when the LLM fails or returns invalid JSON.
- Field filtering and skipped-answer handling.

Existing `RuntimeStore` and `profile.py` tests cover basic load/save behavior; extend them only if the profile validation boundary lives there.

If there is no JavaScript test setup, verify frontend behavior manually through the browser after implementation: first-run prompt, skip, onboarding questions, draft confirmation, save, reopen/edit, and immediate next-chat profile use.

## Acceptance Criteria

- A user with no saved profile sees a non-blocking profile onboarding invitation on first open.
- The user can skip onboarding and chat normally.
- The user can answer five lightweight questions and receive an editable LLM-generated profile draft.
- The profile is saved only after explicit user confirmation.
- "我的画像" is available from the chat page and allows viewing and editing saved fields.
- Saved profile fields are stored in `runtime.sqlite3.profile_entries`.
- Invalid LLM output cannot be saved directly.
- LLM draft failure falls back to a deterministic draft.
- The next chat turn after profile save uses the updated profile.
