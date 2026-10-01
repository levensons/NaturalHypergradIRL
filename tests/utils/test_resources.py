import pytest

from src.utils.resources import RecordTime


def test_record_time_sets_attribute():
    class Dummy:
        pass

    obj = Dummy()

    with RecordTime(obj, "elapsed"):
        pass

    assert obj.elapsed is not None
    assert obj.elapsed >= 0.0


def test_record_time_records_on_exception():
    timer = RecordTime()

    with pytest.raises(RuntimeError):
        with timer:
            raise RuntimeError("test")

    assert timer.elapsed is not None
    assert timer.elapsed >= 0.0
