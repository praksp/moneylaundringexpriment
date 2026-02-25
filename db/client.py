from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator
from neo4j import GraphDatabase, Driver, Session, AsyncGraphDatabase, AsyncDriver, AsyncSession
from config.settings import settings

_driver: Driver | None = None
_async_driver: AsyncDriver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )
        _driver.verify_connectivity()
    return _driver


async def get_async_driver() -> AsyncDriver:
    global _async_driver
    if _async_driver is None:
        _async_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )
        await _async_driver.verify_connectivity()
    return _async_driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


async def close_async_driver() -> None:
    global _async_driver
    if _async_driver is not None:
        await _async_driver.close()
        _async_driver = None


@contextmanager
def neo4j_session() -> Generator[Session, None, None]:
    driver = get_driver()
    session = driver.session(database=settings.neo4j_database)
    try:
        yield session
    finally:
        session.close()


@asynccontextmanager
async def async_neo4j_session() -> AsyncGenerator[AsyncSession, None]:
    driver = await get_async_driver()
    session = driver.session(database=settings.neo4j_database)
    try:
        yield session
    finally:
        await session.close()
