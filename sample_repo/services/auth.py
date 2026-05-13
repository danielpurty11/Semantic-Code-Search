import jwt
import datetime
from fastapi import HTTPException

SECRET_KEY = "supersecret"


def validate_jwt_token(token: str) -> dict:
    """Validates a Bearer JWT token from the request header."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def create_jwt_token(user_id: int, expires_minutes: int = 60) -> str:
    """Create a signed JWT token for the given user."""
    payload = {
        "sub": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
