import logging

from agy_acp.log import SecretMaskingFilter


def _make_record(msg, args=()):
    return logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_secret_masking_filter_standard_dict_keys():
    filter_instance = SecretMaskingFilter()

    record = _make_record("{'name': 'API_KEY', 'value': 'supersecret'}")
    filter_instance.filter(record)

    assert "supersecret" not in record.msg
    assert "***REDACTED***" in record.msg


def test_secret_masking_filter_env_var_model():
    filter_instance = SecretMaskingFilter()

    record = _make_record("EnvVar(name='GEMINI_API_KEY', value='my-top-secret-token')")
    filter_instance.filter(record)

    assert "my-top-secret-token" not in record.msg
    assert "***REDACTED***" in record.msg


def test_secret_masking_filter_args_scrubbing():
    filter_instance = SecretMaskingFilter()

    record = _make_record(
        "Server info: %s",
        args=("EnvVar(name='auth_token', value='12345')",),
    )
    filter_instance.filter(record)

    assert "12345" not in record.args[0]
    assert "***REDACTED***" in record.args[0]
