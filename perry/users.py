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
RESET_CODE_HOURS = 24
RESET_CODE_LIFETIME = datetime.timedelta(hours=RESET_CODE_HOURS)

MAX_FAILED_LOGINS = 5
FAILED_LOGIN_MINUTES = 15
FAILED_LOGIN_WINDOW = datetime.timedelta(minutes=FAILED_LOGIN_MINUTES)

WRONG_LOGIN = "Wrong username or password."
TOO_MANY_LOGINS = "Too many failed logins. Try again later."
WRONG_CURRENT_PASSWORD = "That isn't your current password."
NO_OWNER = f"Perry has no Owner yet. Create one first with: {CREATE_OWNER_COMMAND}"
OWNER_EXISTS = "Perry already has an Owner; there can only be one."
OWNER_CANT_BE_DISABLED = "The Owner can't be disabled."
_ASK_FOR_ANOTHER = "Ask the Owner for a new one."
INVITE_UNKNOWN = "That Invite code isn't right. Check it, or ask the Owner for a new one."
INVITE_USED = f"That Invite has already been used. {_ASK_FOR_ANOTHER}"
INVITE_EXPIRED = f"That Invite has expired. {_ASK_FOR_ANOTHER}"
INVITE_WITHDRAWN = f"That Invite was withdrawn by the Owner. {_ASK_FOR_ANOTHER}"
RESET_CODE_UNKNOWN = "That Reset Code isn't right. Check it, or ask the Owner for a new one."
RESET_CODE_USED = f"That Reset Code has already been used. {_ASK_FOR_ANOTHER}"
RESET_CODE_EXPIRED = f"That Reset Code has expired. {_ASK_FOR_ANOTHER}"
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


class TooManyLogins(Refused):
    """A login refused because of too many recent failed logins for that username."""


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
    _check_new_password(password)
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
    message whether the username or the password is wrong, or the User is disabled.
    After MAX_FAILED_LOGINS failures for a username within FAILED_LOGIN_WINDOW, every
    login with it is refused with TOO_MANY_LOGINS (a TooManyLogins) until the window
    has passed, even with the right password."""
    name = username.strip().lower()
    now = _now()
    with db.pool().connection() as conn:
        failures = conn.execute(
            "SELECT count(*) AS n FROM failed_logins WHERE username = %s AND failed_at > %s",
            (name, now - FAILED_LOGIN_WINDOW),
        ).fetchone()["n"]
        if failures >= MAX_FAILED_LOGINS:
            raise TooManyLogins(TOO_MANY_LOGINS)
        row = conn.execute("SELECT * FROM users WHERE lower(username) = %s", (name,)).fetchone()
    matches = _password_matches(row["password_hash"] if row else _DUMMY_HASH, password)
    if not matches or row is None or row["disabled"]:
        with db.pool().connection() as conn:
            conn.execute(
                "INSERT INTO failed_logins (username, failed_at) VALUES (%s, %s)", (name, now)
            )
        raise Refused(WRONG_LOGIN)
    with db.pool().connection() as conn:
        conn.execute("UPDATE users SET last_login_at = %s WHERE id = %s", (now, row["id"]))
    return _user(row)


@dataclass(frozen=True)
class Member:
    """A User as the Owner panel lists them: when they joined and last logged in."""

    user: User
    joined_at: datetime.datetime
    last_login_at: datetime.datetime | None


def all_users() -> list[Member]:
    """Every User, the Owner first, then in the order they joined."""
    with db.pool().connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY is_owner DESC, id").fetchall()
    return [Member(_user(row), row["created_at"], row["last_login_at"]) for row in rows]


def disable(user_id: int) -> None:
    """Stop a User from logging in, and end every session they have open at once.
    Their Threads and Usage Ledger rows are kept. Refused for the Owner."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "UPDATE users SET disabled = true, session_version = session_version + 1 "
            "WHERE id = %s AND NOT is_owner RETURNING id",
            (user_id,),
        ).fetchone()
    if row is None and (found := get(user_id)) is not None and found.is_owner:
        raise Refused(OWNER_CANT_BE_DISABLED)


def enable(user_id: int) -> None:
    """Let a disabled User log in again. Sessions that disabling ended stay ended."""
    with db.pool().connection() as conn:
        conn.execute("UPDATE users SET disabled = false WHERE id = %s", (user_id,))


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


@dataclass(frozen=True)
class ResetCode:
    """A single-use code from the Owner that lets one User choose a new password."""

    code: str
    username: str
    expires_at: datetime.datetime


def create_reset_code(user_id: int) -> ResetCode:
    """A new Reset Code for the User, valid for RESET_CODE_HOURS."""
    created = _now()
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO password_resets (code, user_id, created_at, expires_at) "
            "VALUES (%s, %s, %s, %s)",
            (secrets.token_urlsafe(9), user_id, created, created + RESET_CODE_LIFETIME),
        )
    return open_reset_codes()[-1]


def open_reset_codes() -> list[ResetCode]:
    """The Reset Codes that can still be used, soonest to expire first."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT r.code, u.username, r.expires_at FROM password_resets r "
            "JOIN users u ON u.id = r.user_id "
            "WHERE r.used_at IS NULL AND r.expires_at > %s ORDER BY r.expires_at",
            (_now(),),
        ).fetchall()
    return [ResetCode(row["code"], row["username"], row["expires_at"]) for row in rows]


def reset_password(code: str, password: str) -> User:
    """The User the Reset Code is for, with `password` as their new password. The code
    is then used up, every session they had is ended, and their failed logins are
    cleared (the Owner vouched for them). Refused, changing nothing, if the code is
    wrong, used or expired, or the password is shorter than MIN_PASSWORD_CHARS."""
    _check_new_password(password)
    with db.pool().connection() as conn, conn.transaction():
        reset = conn.execute(
            "SELECT * FROM password_resets WHERE code = %s FOR UPDATE", (code.strip(),)
        ).fetchone()
        if reset is None:
            raise Refused(RESET_CODE_UNKNOWN)
        if reset["used_at"] is not None:
            raise Refused(RESET_CODE_USED)
        if reset["expires_at"] <= _now():
            raise Refused(RESET_CODE_EXPIRED)
        conn.execute(
            "UPDATE password_resets SET used_at = %s WHERE code = %s", (_now(), reset["code"])
        )
        row = _set_password(conn, reset["user_id"], password)
        conn.execute(
            "DELETE FROM failed_logins WHERE username = lower(%s)", (row["username"],)
        )
    return _user(row)


def change_password(user: User, current_password: str, new_password: str) -> None:
    """Give `user` a new password, if `current_password` is their password now. Ends
    every session they have, so start a new one for the browser they changed it in.
    Refused if the current password is wrong or the new one is shorter than
    MIN_PASSWORD_CHARS."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = %s", (user.id,)
        ).fetchone()
    if row is None or not _password_matches(row["password_hash"], current_password):
        raise Refused(WRONG_CURRENT_PASSWORD)
    _check_new_password(new_password)
    with db.pool().connection() as conn:
        _set_password(conn, user.id, new_password)


def _check_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_CHARS:
        raise Refused(f"The password must be at least {MIN_PASSWORD_CHARS} characters long.")


def _set_password(conn, user_id: int, password: str) -> dict:
    """Give the User a new password and end all their sessions; returns their row."""
    return conn.execute(
        "UPDATE users SET password_hash = %s, session_version = session_version + 1 "
        "WHERE id = %s RETURNING *",
        (_hasher.hash(password), user_id),
    ).fetchone()


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


def end_sessions(user: User) -> None:
    """Log `user` out everywhere: every session token they have stops working."""
    with db.pool().connection() as conn:
        _end_sessions(conn, user.id)


def _end_sessions(conn, user_id: int) -> None:
    conn.execute(
        "UPDATE users SET session_version = session_version + 1 WHERE id = %s", (user_id,)
    )


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
