import logging
from unittest.mock import MagicMock
import pytest
from hellp import SecretMaskingFilter

def test_secret_masking_filter_standard_dict_keys():
    filter_instance = SecretMaskingFilter()
    
    # Simulate a log message with secrets inside a repr
    msg = "{'name': 'API_KEY', 'value': 'supersecret'}"
    record = logging.LogRecord(name="test", level=logging.DEBUG, pathname="test.py", lineno=1, msg=msg, args=(), exc_info=None)
    
    filter_instance.filter(record)
    
    assert "supersecret" not in record.msg
    assert "***REDACTED***" in record.msg

def test_secret_masking_filter_env_var_model():
    filter_instance = SecretMaskingFilter()
    
    # Simulate a log message coming from an EnvVar repr
    msg = "EnvVar(name='GEMINI_API_KEY', value='my-top-secret-token')"
    record = logging.LogRecord(name="test", level=logging.DEBUG, pathname="test.py", lineno=1, msg=msg, args=(), exc_info=None)
    
    filter_instance.filter(record)
    
    assert "my-top-secret-token" not in record.msg
    assert "***REDACTED***" in record.msg

def test_secret_masking_filter_args_scrubbing():
    filter_instance = SecretMaskingFilter()
    
    # Simulate a log message where the secret is in the format args
    record = logging.LogRecord(name="test", level=logging.DEBUG, pathname="test.py", lineno=1, msg="Server info: %s", args=("EnvVar(name='auth_token', value='12345')",), exc_info=None)
    
    filter_instance.filter(record)
    
    assert "12345" not in record.args[0]
    assert "***REDACTED***" in record.args[0]
