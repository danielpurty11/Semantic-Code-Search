import pytest
from services.auth import create_jwt_token, validate_jwt_token


def test_create_and_validate_token():
    token = create_jwt_token(user_id=42)
    payload = validate_jwt_token(token)
    assert payload["sub"] == 42


def test_expired_token_raises():
    token = create_jwt_token(user_id=1, expires_minutes=-1)
    with pytest.raises(Exception):
        validate_jwt_token(token)
