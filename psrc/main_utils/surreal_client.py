from dataclasses import dataclass
from surrealdb.connections.blocking_ws import BlockingWsSurrealConnection
from surrealdb.data.cbor import decode
from surrealdb.errors import ConnectionUnavailableError, UnexpectedResponseError
from surrealdb.request_message.message import RequestMessage
from tqdm import tqdm
from websockets.exceptions import WebSocketException
import websockets
import websockets.sync.client as ws_sync
import json
from typing import Any, Callable, Literal, Mapping, Optional, Tuple
from urllib.parse import urlsplit
from functools import lru_cache
from pathlib import Path
import time
import random
from types import TracebackType


@dataclass(frozen=True, slots=True)
class SurrealClientConfig:
    """Timeout and retry policy for read-only dataset extraction queries.

    ``max_attempts`` includes the initial attempt. Setting a timeout to ``None``
    disables that particular timeout, matching the websockets API.
    """

    open_timeout_seconds: float | None = 30.0
    query_timeout_seconds: float | None = 300.0
    close_timeout_seconds: float | None = 10.0
    ping_interval_seconds: float | None = 20.0
    ping_timeout_seconds: float | None = 20.0
    max_attempts: int = 8
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_seconds: float = 0.5

    def validate(self) -> None:
        for name in (
            "open_timeout_seconds",
            "query_timeout_seconds",
            "close_timeout_seconds",
            "ping_interval_seconds",
            "ping_timeout_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be an integer of at least 1")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be non-negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds must be at least initial_backoff_seconds"
            )
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be non-negative")


class _TimeoutBlockingWsSurrealConnection(BlockingWsSurrealConnection):
    """SurrealDB's blocking WebSocket connection with configurable timeouts."""

    def __init__(self, url: str, config: SurrealClientConfig) -> None:
        super().__init__(url)
        self.config = config

    def _open_socket(self):
        return ws_sync.connect(
            self.raw_url,
            max_size=None,
            subprotocols=[websockets.Subprotocol("cbor")],
            open_timeout=self.config.open_timeout_seconds,
            close_timeout=self.config.close_timeout_seconds,
            ping_interval=self.config.ping_interval_seconds,
            ping_timeout=self.config.ping_timeout_seconds,
        )

    def _send(
        self,
        message: RequestMessage,
        process: str,
        bypass: bool = False,
    ) -> dict[str, Any]:
        # This mirrors surrealdb's BlockingWsSurrealConnection._send while adding
        # the receive timeout that its public constructor currently doesn't expose.
        with self._lock:
            if self.socket is None:
                self.socket = self._open_socket()
            self.socket.send(message.WS_CBOR_DESCRIPTOR)
            data = self.socket.recv(timeout=self.config.query_timeout_seconds)
            response = decode(data if isinstance(data, bytes) else data.encode())

            response_id = response.get("id")
            if response_id is not None and response_id != message.id:
                raise UnexpectedResponseError(
                    f"Response ID mismatch: expected {message.id}, got "
                    f"{response_id}."
                )
            if not bypass:
                self.check_response_for_error(response, process)
            return response

    def __enter__(self) -> "_TimeoutBlockingWsSurrealConnection":
        if self.socket is None:
            self.socket = self._open_socket()
        return self

SURREAL_CLIENT_CONFIG = SurrealClientConfig()
_TRANSIENT_SURREAL_ERRORS = (
    TimeoutError,
    OSError,
    ConnectionUnavailableError,
    UnexpectedResponseError,
    WebSocketException,
)


class RetryingSurrealClient:
    """Reconnect and retry idempotent SurrealDB reads after transient outages."""

    def __init__(
        self,
        login_config: dict[str, Any],
        config: SurrealClientConfig | None = None,
        *,
        connection_factory: Callable[
            [str, SurrealClientConfig], BlockingWsSurrealConnection
        ] = _TimeoutBlockingWsSurrealConnection,
    ) -> None:
        self.login_config = dict(login_config)
        self.config = SURREAL_CLIENT_CONFIG if config is None else config
        self.config.validate()
        required = {"url", "user", "pwd", "ns", "db"}
        missing = sorted(required - self.login_config.keys())
        if missing:
            raise ValueError(
                "SurrealDB login configuration is missing: " + ", ".join(missing)
            )
        scheme = urlsplit(str(self.login_config["url"])).scheme.lower()
        if scheme not in {"ws", "wss"}:
            raise ValueError(
                "RetryingSurrealClient requires a ws:// or wss:// SurrealDB URL"
            )
        self._connection_factory = connection_factory
        self._connection: BlockingWsSurrealConnection | None = None

    def _discard_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.close()
        except Exception:
            pass

    def close(self) -> None:
        self._discard_connection()

    def _connect_once(self) -> BlockingWsSurrealConnection:
        connection = self._connection_factory(
            str(self.login_config["url"]),
            self.config,
        )
        try:
            connection.__enter__()
            connection.signin(
                {
                    "username": self.login_config["user"],
                    "password": self.login_config["pwd"],
                }
            )
            connection.use(
                str(self.login_config["ns"]),
                str(self.login_config["db"]),
            )
        except BaseException:
            try:
                connection.close()
            except Exception:
                pass
            raise
        self._connection = connection
        return connection

    def _delay_seconds(self, failed_attempt: int) -> float:
        base_delay = min(
            self.config.max_backoff_seconds,
            self.config.initial_backoff_seconds
            * self.config.backoff_multiplier ** (failed_attempt - 1),
        )
        return base_delay + random.uniform(0.0, self.config.jitter_seconds)

    def _run_with_retry(
        self,
        operation: Callable[[BlockingWsSurrealConnection], Any],
        description: str,
    ) -> Any:
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                connection = self._connection or self._connect_once()
                return operation(connection)
            except _TRANSIENT_SURREAL_ERRORS as error:
                self._discard_connection()
                if attempt == self.config.max_attempts:
                    error.add_note(
                        f"SurrealDB {description} failed after "
                        f"{self.config.max_attempts} attempts"
                    )
                    raise
                delay = self._delay_seconds(attempt)
                print(
                    f"SurrealDB {description} attempt {attempt}/"
                    f"{self.config.max_attempts} failed with "
                    f"{type(error).__name__}: {error}. Retrying in "
                    f"{delay:.1f}s ...",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable SurrealDB retry state")

    def connect(self) -> None:
        self._run_with_retry(lambda connection: None, "connection")

    def query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Run and retry an idempotent query after reconnecting and signing in."""

        return self._run_with_retry(
            lambda connection: connection.query(query, variables),
            "read query",
        )

    def __enter__(self) -> "RetryingSurrealClient":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@lru_cache(maxsize=1)
def sdb_login():
    login_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data/sdb/login.json"
    )
    with open(login_path, "r") as f:
        config = json.load(f)
    return config


def surreal_client(
    config: SurrealClientConfig | None = None,
) -> RetryingSurrealClient:
    """Create an authenticated, retrying client from the local login file."""

    return RetryingSurrealClient(sdb_login(), config)