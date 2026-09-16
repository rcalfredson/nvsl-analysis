import pytest

from scripts.protocol_audit import command_for, extract_video_lists, parse_audit_record
from src.analysis.training import Training


def test_predicate_is_quiet_and_report_is_structured(capsys):
    training = object.__new__(Training)
    training.n, training.tp, training.sym = 1, Training.TP.circle, False
    assert not training.hasSymCtrl()
    assert capsys.readouterr().out == ""
    training.reportProtocolAudit()
    record = parse_audit_record(capsys.readouterr().out)
    assert record == {"training": 1, "type": str(Training.TP.circle), "hasSymCtrl": False}


def test_parser_ignores_normal_logs_and_rejects_invalid_records():
    assert parse_audit_record("ordinary log line") is None
    with pytest.raises(ValueError):
        parse_audit_record('PROTOCOL_AUDIT {"training": 1}')


def test_command_requests_opt_in_reporting():
    command = command_for("/tmp/video.avi", "HTL")
    assert "--protocol-audit-report" in command
    assert command[1].endswith("/analyze.py")


def test_extract_video_lists():
    entries = extract_video_lists(
        '<p><em>video lists referenced</em></p><ul>'
        '<li>a | b | c | d | HTL | f: /media/videos/*.avi</li></ul>'
    )
    assert entries[0]["chamber"] == "HTL"
    assert entries[0]["patterns"] == ["/media/videos/*.avi"]
