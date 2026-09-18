import time

from src.utils.resources import RecordTime


def test_record_time():
    with RecordTime() as timer:
        time.sleep(0.01)

    assert timer.elapsed is not None
    assert timer.elapsed >= 0.01


def test_record_time_sets_attribute():
    class Dummy:
        pass

    obj = Dummy()

    with RecordTime(obj, "elapsed"):
        time.sleep(0.01)

    assert obj.elapsed >= 0.01
