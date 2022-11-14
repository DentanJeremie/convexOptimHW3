import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='{relativeCreated:8.0f}ms {levelname:s} [{filename:s}] {message:s}',
    style='{'
)