"""API key storage. Uses Windows Credential Manager (via `keyring`); never written to disk
in plain text, never logged, never returned by the API. ANTHROPIC_API_KEY env var is a fallback."""
from __future__ import annotations

import os

SERVICE = "RogueSniperAI"
KEY_NAME = "anthropic_api_key"


class SecretStoreError(RuntimeError):
    pass


def _keyring():
    try:
        import keyring
        from keyring.backends import fail
        if isinstance(keyring.get_keyring(), fail.Keyring):
            return None
        return keyring
    except Exception:
        return None


def get_api_key() -> str | None:
    kr = _keyring()
    if kr:
        try:
            v = kr.get_password(SERVICE, KEY_NAME)
            if v:
                return v
        except Exception:
            pass
    return os.environ.get("ANTHROPIC_API_KEY") or None


def set_api_key(key: str) -> None:
    key = (key or "").strip()
    if not key:
        raise SecretStoreError("empty key")
    kr = _keyring()
    if not kr:
        raise SecretStoreError("No secure credential store available. On Windows this uses Credential "
                               "Manager; elsewhere set the ANTHROPIC_API_KEY environment variable instead.")
    kr.set_password(SERVICE, KEY_NAME, key)


def delete_api_key() -> None:
    kr = _keyring()
    if kr:
        try:
            kr.delete_password(SERVICE, KEY_NAME)
        except Exception:
            pass


def key_status() -> dict:
    kr = _keyring()
    if kr:
        try:
            if kr.get_password(SERVICE, KEY_NAME):
                return {"configured": True, "source": "credential_manager"}
        except Exception:
            pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"configured": True, "source": "environment"}
    return {"configured": False, "source": None}
