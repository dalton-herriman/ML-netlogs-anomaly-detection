from __future__ import annotations

import io
import json
import logging

from src.logging_setup import JsonFormatter, configure_logging, get_request_id, set_request_id


def test_json_formatter_emits_valid_json():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    set_request_id("abc123")
    out = JsonFormatter().format(record)
    payload = json.loads(out)
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "abc123"


def test_configure_logging_attaches_json_handler(capsys):
    configure_logging("INFO")
    log = logging.getLogger("test-logger")
    log.info("structured", extra={"custom_field": "v"})
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "expected a log line"
    payload = json.loads(captured[-1])
    assert payload["message"] == "structured"
    assert payload["custom_field"] == "v"


def test_request_id_default_and_custom():
    set_request_id("xyz")
    assert get_request_id() == "xyz"
    rid = set_request_id()
    assert rid
    assert get_request_id() == rid


def test_set_request_id_generates_unique_ids():
    a = set_request_id()
    b = set_request_id()
    assert a != b


def test_formatter_with_stream_handler():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("streamed")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        logger.info("stream msg")
    finally:
        logger.handlers = []
    payload = json.loads(stream.getvalue().strip())
    assert payload["message"] == "stream msg"
