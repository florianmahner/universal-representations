from src import factorization


def test_get_snmf_class_returns_the_vendored_estimator():
    import snmf

    assert factorization.get_snmf_class() is snmf.SNMF
