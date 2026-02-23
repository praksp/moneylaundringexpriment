"""
FastAPI dependency functions for authentication and role-based access control.

Usage:
    # Any authenticated user
    @router.get("/me")
    async def me(user: UserInDB = Depends(get_current_user)):
        ...

    # Admin-only
    @router.get("/admin-only")
    async def admin(user: UserInDB = Depends(require_admin)):
        ...
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from auth.security import decode_token, get_user
from auth.models import UserInDB

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        username: str = payload.sub
    except JWTError:
        raise credentials_exc

    user = get_user(username)
    if user is None or not user.is_active:
        raise credentials_exc
    return user


def require_admin(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to view customer profiles",
        )
    return current_user


def require_viewer_or_admin(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    if current_user.role not in ("admin", "viewer"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user
