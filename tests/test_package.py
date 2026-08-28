def test_package_version() -> None:
    import trustsr

    assert trustsr.__version__ == "0.1.0"


def test_package_has_description() -> None:
    import trustsr

    assert trustsr.__doc__
