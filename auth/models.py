"""Pydantic models for authentication and user management."""
from pydantic import BaseModel
from typing import Optional


class UserInDB(BaseModel):
    """User record as stored in Neo4j."""
    id: str
    username: str
    hashed_password: str
    role: str           # "admin" | "viewer"
    full_name: Optional[str] = None
    is_active: bool = True


class UserPublic(BaseModel):
    """User info safe to return in API responses (no password hash)."""
    id: str
    username: str
    role: str
    full_name: Optional[str] = None
    is_active: bool


class TokenPayload(BaseModel):
    sub: str            # username
    role: str
    exp: Optional[int] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
