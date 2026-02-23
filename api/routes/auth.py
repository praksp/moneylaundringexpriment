"""Authentication routes: login, logout, current user."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from auth.security import authenticate_user, create_access_token
from auth.dependencies import get_current_user
from auth.models import TokenResponse, UserPublic, LoginRequest, UserInDB

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(form: LoginRequest):
    """
    Exchange username + password for a JWT bearer token.
    Roles:
      admin  → full access including customer profiles
      viewer → aggregated view only (no PII / customer data)
    """
    user = authenticate_user(form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(username=user.username, role=user.role)
    return TokenResponse(
        access_token=token,
        user=UserPublic(
            id=user.id,
            username=user.username,
            role=user.role,
            full_name=user.full_name,
            is_active=user.is_active,
        ),
    )


@router.get("/me", response_model=UserPublic)
async def get_me(current_user: UserInDB = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return UserPublic(
        id=current_user.id,
        username=current_user.username,
        role=current_user.role,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
    )
