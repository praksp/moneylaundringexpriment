"""
JWT creation / verification and bcrypt password hashing.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
import bcrypt

from auth.models import UserInDB, TokenPayload

# ── Config ─────────────────────────────────────────────────────────────────────

# In production, load this from an environment variable.
SECRET_KEY = "aml-rbac-secret-key-change-in-production-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480   # 8 hours


# ── Password helpers (bcrypt direct — avoids passlib/Python 3.14 compat issue) ─

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── JWT helpers ────────────────────────────────────────────────────────────────

def create_access_token(username: str, role: str,
                        expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    """Raises JWTError on invalid / expired token."""
    raw = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return TokenPayload(**raw)


# ── Neo4j user store ───────────────────────────────────────────────────────────

def _user_query(session, username: str) -> Optional[UserInDB]:
    result = session.run(
        "MATCH (u:User {username: $username}) RETURN u",
        username=username,
    ).single()
    if result is None:
        return None
    node = dict(result["u"])
    return UserInDB(**node)


def get_user(username: str) -> Optional[UserInDB]:
    from db.client import neo4j_session
    with neo4j_session() as session:
        return _user_query(session, username)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user_in_db(username: str, password: str,
                      role: str, full_name: str = "") -> UserInDB:
    """Insert a User node into Neo4j (idempotent by username)."""
    from db.client import neo4j_session
    uid = str(uuid.uuid4())
    hashed = hash_password(password)
    with neo4j_session() as session:
        session.run(
            """
            MERGE (u:User {username: $username})
            ON CREATE SET
                u.id            = $id,
                u.hashed_password = $hashed,
                u.role          = $role,
                u.full_name     = $full_name,
                u.is_active     = true
            """,
            username=username, id=uid, hashed=hashed,
            role=role, full_name=full_name,
        )
    return get_user(username)


def ensure_user_schema():
    """Create constraint on User.username (idempotent)."""
    from db.client import neo4j_session
    with neo4j_session() as session:
        try:
            session.run(
                "CREATE CONSTRAINT user_username_unique IF NOT EXISTS "
                "FOR (u:User) REQUIRE u.username IS UNIQUE"
            )
        except Exception:
            pass


def seed_default_users():
    """
    Create default users if they don't exist.
    Called once at API startup.

      admin  / password  → role: admin   (full access)
      viewer / viewer123 → role: viewer  (aggregated view only)
    """
    ensure_user_schema()
    create_user_in_db("admin",  "password",  "admin",  "System Administrator")
    create_user_in_db("viewer", "viewer123", "viewer", "Read-Only Analyst")
