from src import factorization


def test_get_srf_class_returns_the_upstream_estimator():
    import pysrf

    assert factorization.get_srf_class() is pysrf.SRF
