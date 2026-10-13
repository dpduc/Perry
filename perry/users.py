"""Users: who is logged in, their Profile, and the one Owner.

The only module that knows about passwords and sessions. Everything else gets a `User`.
Passwords are hashed with argon2. A session is a signed token naming the User and their
session version; bumping the version ends all of that User's sessions.

Create the Owner once with:

    uv run python -m perry.users create-owner
"""

import datetime
import getpass
import os
import secrets
import sys
import zoneinfo
from collections.abc import Callable
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from itsdangerous import BadSignature, URLSafeTimedSerializer
from psycopg import errors

from perry import db

MIN_PASSWORD_CHARS = 10
MAX_ABOUT_CHARS = 300
SESSION_DAYS = 14
SESSION_SECONDS = SESSION_DAYS * 24 * 60 * 60
SESSION_SECRET_VARIABLE = "PERRY_SESSION_SECRET"
CREATE_OWNER_COMMAND = "uv run python -m perry.users create-owner"
INVITE_DAYS = 7
INVITE_LIFETIME = datetime.timedelta(days=INVITE_DAYS)

WRONG_LOGIN = "Wrong username or password."
NO_OWNER = f"Perry has no Owner yet. Create one first with: {CREATE_OWNER_COMMAND}"
OWNER_EXISTS = "Perry already has an Owner; there can only be one."
_ASK_FOR_ANOTHER = "Ask the Owner for a new one."
INVITE_UNKNOWN = "That Invite code isn't right. Check it, or ask the Owner for a new one."
INVITE_USED = f"That Invite has already been used. {_ASK_FOR_ANOTHER}"
INVITE_EXPIRED = f"That Invite has expired. {_ASK_FOR_ANOTHER}"
INVITE_WITHDRAWN = f"That Invite was withdrawn by the Owner. {_ASK_FOR_ANOTHER}"
NO_SESSION_SECRET = (
    f"{SESSION_SECRET_VARIABLE} is not set. Add a long random value to .env, e.g. the "
    'output of: uv run python -c "import secrets; print(secrets.token_urlsafe(32))"'
)

_hasher = PasswordHasher()
# Checked against when the username doesn't exist, so a wrong username takes as long
# to refuse as a wrong password.
_DUMMY_HASH = _hasher.hash("not anyone's password")


class Refused(Exception):
    """A request about Users that can't be done; the message says why, for the person."""


@dataclass(frozen=True)
class Profile:
    """What a User tells Perry about themselves."""

    display_name: str
    time_zone: str
    about: str = ""


@dataclass(frozen=True)
class User:
    id: int
    username: str
    is_owner: bool
    disabled: bool
    profile: Profile


def create_user(
    username: str,
    password: str,
    display_name: str,
    time_zone: str,
    *,
    is_owner: bool = False,
) -> User:
    """A new User. Refused if the username is taken (ignoring case), the password is
    shorter than MIN_PASSWORD_CHARS, or the time zone isn't a known one."""
    _check_new_user(username, password, display_name, time_zone)
    with db.pool().connection() as conn:
        return _insert_user(conn, username, password, display_name, time_zone, is_owner)


def _check_new_user(username: str, password: str, display_name: str, time_zone: str) -> None:
    if not username.strip():
        raise Refused("Choose a username.")
    if len(password) < MIN_PASSWORD_CHARS:
        raise Refused(f"The password must be at least {MIN_PASSWORD_CHARS} characters long.")
    if not display_name.strip():
        raise Refused("Give a display name.")
    _check_time_zone(time_zone)


def _insert_user(
    conn, username: str, password: str, display_name: str, time_zone: str, is_owner=False
) -> User:
    """Insert a checked new User on `conn`; a clash rolls back `conn`'s transaction."""
    username, display_name = username.strip(), display_name.strip()
    try:
        with conn.transaction():
            row = conn.execute(
                "INSERT INTO users (username, password_hash, is_owner, display_name, time_zone) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING *",
                (username, _hasher.hash(password), is_owner, display_name, time_zone),
            ).fetchone()
    except errors.UniqueViolation as clash:
        if "users_one_owner_idx" in str(clash):
            raise Refused(OWNER_EXISTS) from None
        raise Refused(f"The username {username!r} is taken.") from None
    return _user(row)


def create_owner(username: str, password: str, display_name: str, time_zone: str) -> User:
    """The Owner, who also gets every Thread from before Users existed. Refused if there
    is already an Owner."""
    if owner() is not None:
        raise Refused(OWNER_EXISTS)
    created = create_user(username, password, display_name, time_zone, is_owner=True)
    with db.pool().connection() as conn:
        conn.execute("UPDATE threads SET user_id = %s WHERE user_id IS NULL", (created.id,))
    return created


def owner() -> User | None:
    with db.pool().connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE is_owner").fetchone()
    return _user(row) if row else None


def require_owner() -> User:
    """The Owner; refused (naming the create-owner command) if there is none yet."""
    found = owner()
    if found is None:
        raise Refused(NO_OWNER)
    return found


def get(user_id: int) -> User | None:
    with db.pool().connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
    return _user(row) if row else None


def update_profile(user: User, display_name: str, time_zone: str, about: str = "") -> User:
    """`user` with their Profile changed. Refused if the display name is empty, the time
    zone isn't a known one, or the about-me is over MAX_ABOUT_CHARS. The about-me may be
    empty; it is kept as one line (the Supervisor reads it)."""
    display_name, time_zone = display_name.strip(), time_zone.strip()
    about = " ".join(about.split())
    if not display_name:
        raise Refused("Give a display name.")
    _check_time_zone(time_zone)
    if len(about) > MAX_ABOUT_CHARS:
        raise Refused(f"Keep the about-me line to {MAX_ABOUT_CHARS} characters.")
    with db.pool().connection() as conn:
        row = conn.execute(
            "UPDATE users SET display_name = %s, time_zone = %s, about = %s "
            "WHERE id = %s RETURNING *",
            (display_name, time_zone, about, user.id),
        ).fetchone()
    return _user(row)


def authenticate(username: str, password: str) -> User:
    """The User with this username and password. Refused with the same WRONG_LOGIN
    message whether the username or the password is wrong, or the User is disabled."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE lower(username) = lower(%s)", (username.strip(),)
        ).fetchone()
    if not _password_matches(row["password_hash"] if row else _DUMMY_HASH, password):
        raise Refused(WRONG_LOGIN)
    if row is None or row["disabled"]:
        raise Refused(WRONG_LOGIN)
    with db.pool().connection() as conn:
        conn.execute("UPDATE users SET last_login_at = now() WHERE id = %s", (row["id"],))
    return _user(row)


@dataclass(frozen=True)
class Invite:
    """A single-use code from the Owner that lets one person register."""

    code: str
    created_at: datetime.datetime
    expires_at: datetime.datetime


def create_invite() -> Invite:
    """A new Invite, valid for INVITE_DAYS. Not tied to a name: the invitee chooses
    their own username."""
    created = _now()
    with db.pool().connection() as conn:
        row = conn.execute(
            "INSERT INTO invites (code, created_at, expires_at) VALUES (%s, %s, %s) "
            "RETURNING *",
            (secrets.token_urlsafe(9), created, created + INVITE_LIFETIME),
        ).fetchone()
    return _invite(row)


def open_invites() -> list[Invite]:
    """The Invites that can still be registered with, soonest to expire first."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT * FROM invites WHERE used_at IS NULL AND withdrawn_at IS NULL "
            "AND expires_at > %s ORDER BY expires_at",
            (_now(),),
        ).fetchall()
    return [_invite(row) for row in rows]


def withdraw_invite(code: str) -> None:
    """Cancel an unused Invite, so nobody can register with it. A used one is kept."""
    with db.pool().connection() as conn:
        conn.execute(
            "UPDATE invites SET withdrawn_at = %s "
            "WHERE code = %s AND used_at IS NULL AND withdrawn_at IS NULL",
            (_now(), code),
        )


def register(
    invite: str, username: str, password: str, display_name: str, time_zone: str
) -> User:
    """A new User, registered with an Invite, which is then used up. Refused, leaving
    the Invite as it was, if the Invite is wrong, used, expired or withdrawn, or the
    User can't be created (see `create_user`)."""
    _check_new_user(username, password, display_name, time_zone)
    with db.pool().connection() as conn, conn.transaction():
        row = conn.execute(
            "SELECT * FROM invites WHERE code = %s FOR UPDATE", (invite.strip(),)
        ).fetchone()
        _check_invite(row)
        created = _insert_user(conn, username, password, display_name, time_zone)
        conn.execute(
            "UPDATE invites SET used_at = %s, used_by = %s WHERE code = %s",
            (_now(), created.id, row["code"]),
        )
    return created


def _check_invite(row: dict | None) -> None:
    if row is None:
        raise Refused(INVITE_UNKNOWN)
    if row["used_at"] is not None:
        raise Refused(INVITE_USED)
    if row["withdrawn_at"] is not None:
        raise Refused(INVITE_WITHDRAWN)
    if row["expires_at"] <= _now():
        raise Refused(INVITE_EXPIRED)


def _invite(row: dict) -> Invite:
    return Invite(row["code"], row["created_at"], row["expires_at"])


def _now() -> datetime.datetime:
    """The time now. Tests move it to see Invites expire."""
    return datetime.datetime.now(datetime.UTC)


def start_session(user: User) -> str:
    """A signed session token for `user`, valid for SESSION_DAYS."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT session_version FROM users WHERE id = %s", (user.id,)
        ).fetchone()
    return _serializer().dumps({"user": user.id, "version": row["session_version"]})


def user_for_session(token: str | None) -> User | None:
    """The User behind a session token; None if the token is missing, forged, older than
    SESSION_DAYS, or its User is disabled or has had their sessions ended since."""
    if not token:
        return None
    try:
        session = _serializer().loads(token, max_age=SESSION_SECONDS)
        user_id, version = int(session["user"]), int(session["version"])
    except (BadSignature, KeyError, TypeError, ValueError):
        return None
    with db.pool().connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
    if row is None or row["disabled"] or row["session_version"] != version:
        return None
    return _user(row)


def session_secret() -> str:
    """The secret sessions are signed with; refused if it isn't configured."""
    secret = os.getenv(SESSION_SECRET_VARIABLE, "").strip()
    if not secret:
        raise Refused(NO_SESSION_SECRET)
    return secret


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(session_secret(), salt="perry-session")


def _password_matches(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def _check_time_zone(time_zone: str) -> None:
    try:
        zoneinfo.ZoneInfo(time_zone)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise Refused(
            f"{time_zone!r} isn't a known time zone; use a name like Europe/London."
        ) from None


def _user(row: dict) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        is_owner=row["is_owner"],
        disabled=row["disabled"],
        profile=Profile(row["display_name"], row["time_zone"], row["about"]),
    )


USAGE = f"Usage: {CREATE_OWNER_COMMAND}"


def main(
    argv: list[str] | None = None,
    ask: Callable[[str], str] = input,
    ask_secret: Callable[[str], str] = getpass.getpass,
) -> int:
    """The create-owner command: asks for the Owner's username, password, display name
    and time zone. Returns the exit code."""
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["create-owner"]:
        print(USAGE, file=sys.stderr)
        return 2
    db.init_schema()
    if owner() is not None:
        print(OWNER_EXISTS, file=sys.stderr)
        return 1
    username = ask("Username: ")
    password = ask_secret(f"Password (at least {MIN_PASSWORD_CHARS} characters): ")
    if ask_secret("Password again: ") != password:
        print("The passwords don't match. Nothing was created.", file=sys.stderr)
        return 1
    display_name = ask("Display name: ")
    time_zone = ask("Time zone (e.g. Europe/London): ").strip()
    try:
        created = create_owner(username, password, display_name, time_zone)
    except Refused as refused:
        print(f"{refused} Nothing was created.", file=sys.stderr)
        return 1
    print(f"Created the Owner {created.username!r}. Log in to Perry with it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
