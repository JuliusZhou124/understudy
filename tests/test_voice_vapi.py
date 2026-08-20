from understudy.voice.vapi import handle_webhook


def test_transcript_event_is_translated():
    events = handle_webhook({"message": {"type": "transcript", "role": "assistant",
                                         "transcript": "Hi there", "transcriptType": "final"}})
    assert events == [{"type": "transcript", "speaker": "buyer", "text": "Hi there", "final": True}]


def test_seller_side_transcript_maps_to_seller():
    events = handle_webhook({"message": {"type": "transcript", "role": "user",
                                         "transcript": "It's five hundred",
                                         "transcriptType": "partial"}})
    assert events[0]["speaker"] == "seller"
    assert events[0]["final"] is False


def test_tool_calls_become_offer_and_deal_events():
    body = {"message": {"type": "tool-calls", "toolCallList": [
        {"id": "1", "name": "log_offer", "arguments": {"price": 480}},
        {"id": "2", "name": "accept_offer", "arguments": '{"price": 455}'},
    ]}}
    events = handle_webhook(body)
    assert {"type": "offer", "price": 480.0} in events
    assert {"type": "deal", "price": 455.0} in events


def test_walk_away_becomes_a_call_ended_event():
    body = {"message": {"type": "tool-calls", "toolCallList": [
        {"id": "1", "function": {"name": "walk_away", "arguments": '{"reason": "too high"}'}},
    ]}}
    assert handle_webhook(body) == [{"type": "call-ended", "outcome": "no_deal", "reason": "too high"}]


def test_malformed_tool_arguments_do_not_raise():
    body = {"message": {"type": "tool-calls", "toolCallList": [
        {"id": "1", "name": "log_offer", "arguments": "{not json"},
    ]}}
    assert handle_webhook(body) == []


def test_status_and_report_events():
    assert handle_webhook({"message": {"type": "status-update", "status": "in-progress"}}) == \
        [{"type": "status", "status": "in-progress"}]
    out = handle_webhook({"message": {"type": "end-of-call-report",
                                      "endedReason": "hangup", "summary": "done"}})
    assert out[0]["type"] == "report"


def test_unknown_event_types_are_ignored():
    assert handle_webhook({"message": {"type": "something-else"}}) == []
    assert handle_webhook({}) == []
