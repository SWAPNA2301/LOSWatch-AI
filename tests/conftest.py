import pytest

from loswatch.database import initialize, make_engine


@pytest.fixture
def factory(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    factory = initialize(engine)
    yield factory
    engine.dispose()
