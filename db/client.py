from contextlib import contextmanager
from typing import Generator
from neo4j import GraphDatabase, Driver, Session
from config.settings import settings

_driver: Driver | None = None


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


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


@contextmanager
def neo4j_session() -> Generator[Session, None, None]:
    driver = get_driver()
    session = driver.session(database=settings.neo4j_database)
    try:
        yield session
    finally:
        session.close()
