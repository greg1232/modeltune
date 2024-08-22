import pytest
import mlcdocker


def test_image_uri():
    uri = mlcdocker.image_uri(
        image="someimage",
    )
    assert uri == "ghcr.io/mlcommons/someimage:latest"

    uri = mlcdocker.image_uri(image="someimage", tag="1.2.3")
    assert uri == "ghcr.io/mlcommons/someimage:1.2.3"

    uri = mlcdocker.image_uri(image="someimage", tag="1.2.3", namespace="meta")
    assert uri == "ghcr.io/meta/someimage:1.2.3"

    uri = mlcdocker.image_uri(
        image="someimage", tag="1.2.3", namespace="meta", registry="dockerhub.io"
    )
    assert uri == "dockerhub.io/meta/someimage:1.2.3"
