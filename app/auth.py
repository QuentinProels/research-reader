"""App-level shared password.

Second layer behind Cloudflare Access, for when the tunnel hostname leaks or
Access is misconfigured. Not a replacement for Access.
"""

import hmac
import secrets

import bcrypt
from fastapi import Request
from itsdangerous import BadSignature, URLSafeTimedSerializer

from app.config import settings

COOKIE_NAME = "reader_session"
MAX_AGE_SECONDS = 30 * 24 * 3600

_serializer = URLSafeTimedSerializer(settings.session_secret, salt="reader-session")


def enabled() -> bool:
    return bool(settings.app_password_hash)


def verify_password(password: str) -> bool:
    if not enabled():
        return True
    try:
        return bcrypt.checkpw(password.encode(), settings.app_password_hash.encode())
    except ValueError:
        return False


def issue_cookie() -> str:
    return _serializer.dumps({"nonce": secrets.token_hex(8)})


def is_authenticated(request: Request) -> bool:
    if not enabled():
        return True
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return False
    try:
        _serializer.loads(token, max_age=MAX_AGE_SECONDS)
    except BadSignature:
        return False
    return True


def constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
