from dementia_project.data.name_normalize import normalize_person_name


def test_normalize_person_name_basic() -> None:
    assert normalize_person_name("Robin Williams") == "robinwilliams"
    assert normalize_person_name("Robin_Williams__01.wav") == "robinwilliams01wav"
    assert normalize_person_name("  Terry O'Rahilly  ") == "terryorahilly"


