"""Accounts: password hashing, API tokens and the request's current user.

- Passwords: scrypt (stdlib) with a random salt, stored as ``scrypt$n$r$p$salt$hash`` (hex).
- Tokens: ``kx_`` + 32 random bytes; only the SHA-256 is stored, so a database leak gives no
  usable token. They expire after ``server.token_days`` and are revoked on logout.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select, update

from kartrix.config import settings
from kartrix.db.engine import session_scope
from kartrix.db.models import ApiToken, User

TOKEN_PREFIX = "kx_"  # noqa: S105 — a prefix, not a secret
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD, MAX_PASSWORD = 8, 256


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        got = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=len(digest) // 2
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), digest)


# Login with an unknown email still spends one scrypt, so response time doesn't tell which emails exist.
_DUMMY_HASH = hash_password(secrets.token_hex(8))


def normalize_email(email: str) -> str:
    email = email.strip().lower()
    if len(email) > 320 or not _EMAIL.match(email):
        raise ValueError("not a valid email address")
    return email


def check_password_rules(password: str) -> None:
    if not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
        raise ValueError(f"the password must be {MIN_PASSWORD}–{MAX_PASSWORD} characters")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def create_user(email: str, password: str) -> User:
    """A new account; ValueError if the email or password is not acceptable or the email is taken."""
    email = normalize_email(email)
    check_password_rules(password)
    async with session_scope() as s:
        if (await s.execute(select(User.id).where(User.email == email))).first() is not None:
            raise ValueError("an account with this email already exists")
        user = User(id=uuid.uuid4(), email=email, password_hash=hash_password(password))
        s.add(user)
    return user


async def set_password(email: str, password: str) -> bool:
    """Replace a user's password and revoke their tokens; False if there is no such user."""
    email = normalize_email(email)
    check_password_rules(password)
    async with session_scope() as s:
        user = (await s.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user is None:
            return False
        user.password_hash = hash_password(password)
        await s.execute(
            update(ApiToken)
            .where(ApiToken.user_id == user.id, ApiToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(UTC))
        )
    return True


async def list_users() -> list[User]:
    async with session_scope() as s:
        return list((await s.execute(select(User).order_by(User.created_at))).scalars())


async def ensure_bootstrap_admin() -> str | None:
    """``KARTRIX_ADMIN_EMAIL`` + ``KARTRIX_ADMIN_PASSWORD``: create that account if it doesn't exist yet — how
    the first user gets in when sign-up is closed and the host has no shell. Never changes an existing user."""
    import os

    email, password = os.environ.get("KARTRIX_ADMIN_EMAIL", "").strip(), os.environ.get("KARTRIX_ADMIN_PASSWORD", "")
    if not email or not password:
        return None
    try:
        await create_user(email, password)
    except ValueError as e:
        if "already exists" in str(e):
            return None
        raise
    return normalize_email(email)


async def authenticate(email: str, password: str) -> User | None:
    try:
        email = normalize_email(email)
    except ValueError:
        return None
    async with session_scope() as s:
        user = (await s.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        verify_password(password, _DUMMY_HASH)
        return None
    return user if verify_password(password, user.password_hash) else None


async def issue_token(user_id: uuid.UUID, name: str) -> str:
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    expires = datetime.now(UTC) + timedelta(days=settings.server.token_days)
    async with session_scope() as s:
        s.add(ApiToken(user_id=user_id, token_hash=token_hash(token), name=name[:100], expires_at=expires))
    return token


async def revoke_token(token: str) -> None:
    async with session_scope() as s:
        await s.execute(
            update(ApiToken)
            .where(ApiToken.token_hash == token_hash(token), ApiToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(UTC))
        )


async def user_for_token(token: str) -> User | None:
    if not token.startswith(TOKEN_PREFIX):
        return None
    now = datetime.now(UTC)
    async with session_scope() as s:
        row = (
            await s.execute(
                select(User, ApiToken.id)
                .join(ApiToken, ApiToken.user_id == User.id)
                .where(
                    ApiToken.token_hash == token_hash(token),
                    ApiToken.revoked_at.is_(None),
                    ApiToken.expires_at > now,
                )
            )
        ).first()
        if row is None:
            return None
        await s.execute(update(ApiToken).where(ApiToken.id == row[1]).values(last_used_at=now))
    return row[0]


def bearer_token(authorization: Annotated[str | None, Header()] = None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    return token.strip()


async def current_user(token: Annotated[str, Depends(bearer_token)]) -> User:
    user = await user_for_token(token)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
        )
    return user


CurrentUser = Annotated[User, Depends(current_user)]
