from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import agent_runtime
from agent_runtime import AgentTool, VAPAgentRuntime, trim_history
from agent_store import AgentStore


def text_chunks(text: str, finish: str = "stop") -> list:
    return [
        NS(choices=[NS(delta=NS(content=text, tool_calls=None), finish_reason=None)]),
        NS(choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason=finish)]),
    ]


def tool_chunks(*calls: tuple[str, str, str]) -> list:
    """Each call is streamed in two fragments, like the real API does."""
    chunks = []
    for index, (call_id, name, arguments) in enumerate(calls):
        head, tail = arguments[:3], arguments[3:]
        chunks.append(
            NS(
                choices=[
                    NS(
                        delta=NS(
                            content=None,
                            tool_calls=[
                                NS(
                                    index=index,
                                    id=call_id,
                                    function=NS(name=name, arguments=head),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            )
        )
        chunks.append(
            NS(
                choices=[
                    NS(
                        delta=NS(
                            content=None,
                            tool_calls=[
                                NS(
                                    index=index,
                                    id=None,
                                    function=NS(name=None, arguments=tail),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            )
        )
    chunks.append(
        NS(
            choices=[
                NS(delta=NS(content=None, tool_calls=None), finish_reason="tool_calls")
            ]
        )
    )
    return chunks


class FakeLLM:
    def __init__(self, *responses: list) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"])))
        return iter(self.responses.pop(0))


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = AgentStore(Path(tmp.name) / "agent.sqlite3")
        self.runtime = VAPAgentRuntime(store=self.store)
        self.runtime._subscription_key = "key"
        self.started: list[dict] = []
        tools = [
            AgentTool(
                "echo",
                "Echo",
                {"type": "object"},
                "read_only",
                lambda args: {"echo": args},
            ),
            AgentTool(
                "big",
                "Big",
                {"type": "object"},
                "read_only",
                lambda args: {"blob": "x" * 50000},
            ),
            AgentTool(
                "start_run",
                "Start",
                {"type": "object"},
                "requires_approval",
                self.start,
            ),
            AgentTool(
                "dl",
                "Download",
                {"type": "object"},
                "safe",
                lambda args: {
                    "downloads": [
                        {
                            "label": "report",
                            "download_url": "/api/attribution/file?name=r.md",
                        },
                        {"label": "evil", "download_url": "https://example.com/x"},
                    ]
                },
            ),
        ]
        for tool in tools:
            self.runtime.register_tool(tool)

    def start(self, args: dict) -> dict:
        self.started.append(args)
        return {"started": True}

    def drive(self, method, payload: dict, llm: FakeLLM) -> list[dict]:
        with patch.object(self.runtime, "_create_client", return_value=llm):
            return list(method(payload))

    def tool_messages(self, conversation_id: str) -> dict[str, dict]:
        return {
            message["tool_call_id"]: json.loads(message["content"])
            for message in self.store.load_messages(conversation_id)
            if message["role"] == "tool"
        }

    def test_tool_results_stay_in_history_for_the_next_turn(self) -> None:
        llm = FakeLLM(
            tool_chunks(("c1", "echo", '{"q": 1}')),
            text_chunks("first"),
            text_chunks("second"),
        )
        events = self.drive(self.runtime.stream_chat, {"message": "hi"}, llm)
        conversation_id = events[0]["conversation_id"]
        self.assertEqual(
            [event["type"] for event in events],
            ["conversation", "tool_event", "delta", "done"],
        )
        self.drive(
            self.runtime.stream_chat,
            {"message": "again", "conversation_id": conversation_id},
            llm,
        )

        roles = [message["role"] for message in llm.requests[2]]
        self.assertEqual(
            roles, ["system", "user", "assistant", "tool", "assistant", "user"]
        )
        self.assertEqual(llm.requests[2][3]["tool_call_id"], "c1")
        self.assertEqual(
            json.loads(llm.requests[2][3]["content"])["data"], {"echo": {"q": 1}}
        )

    def test_approval_suspends_and_the_decision_resumes_the_run(self) -> None:
        llm = FakeLLM(
            tool_chunks(
                ("c1", "echo", "{}"),
                ("c2", "start_run", '{"tp": 4}'),
                ("c3", "echo", '{"y": 2}'),
            ),
            text_chunks("started"),
        )
        events = self.drive(self.runtime.stream_chat, {"message": "run it"}, llm)
        self.assertEqual(events[-1]["type"], "approval_required")
        self.assertEqual(self.started, [])
        approval_id = events[-1]["approval"]["approval_id"]
        conversation_id = events[0]["conversation_id"]

        resumed = self.drive(
            self.runtime.stream_decision,
            {"approval_id": approval_id, "approved": True},
            llm,
        )

        self.assertEqual(self.started, [{"tp": 4}])
        self.assertEqual(
            [
                event.get("tool_event", {}).get("tool_name")
                for event in resumed
                if event["type"] == "tool_event"
            ],
            ["start_run", "echo"],
        )
        self.assertEqual(resumed[-1]["message"]["content"], "started")
        sent = [message for message in llm.requests[1] if message["role"] == "tool"]
        self.assertEqual(
            [message["tool_call_id"] for message in sent], ["c1", "c2", "c3"]
        )
        self.assertEqual(json.loads(sent[1]["content"])["data"], {"started": True})
        kinds = [event["kind"] for event in self.store.events(conversation_id)]
        self.assertEqual(kinds.count("approval"), 1)
        self.assertIn("llm_call", kinds)
        with self.assertRaises(ValueError):
            self.drive(
                self.runtime.stream_decision,
                {"approval_id": approval_id, "approved": True},
                llm,
            )

    def test_rejection_is_reported_to_the_model_without_running_the_tool(self) -> None:
        llm = FakeLLM(
            tool_chunks(("c2", "start_run", "{}")), text_chunks("ok, not started")
        )
        events = self.drive(self.runtime.stream_chat, {"message": "run it"}, llm)
        resumed = self.drive(
            self.runtime.stream_decision,
            {"approval_id": events[-1]["approval"]["approval_id"], "approved": False},
            llm,
        )
        self.assertEqual(self.started, [])
        self.assertFalse(resumed[1]["tool_event"]["ok"])
        result = self.tool_messages(events[0]["conversation_id"])["c2"]
        self.assertIn("rejected", result["message"])

    def test_a_new_message_closes_a_pending_approval(self) -> None:
        llm = FakeLLM(tool_chunks(("c2", "start_run", "{}")), text_chunks("never mind"))
        events = self.drive(self.runtime.stream_chat, {"message": "run it"}, llm)
        conversation_id = events[0]["conversation_id"]
        self.drive(
            self.runtime.stream_chat,
            {"message": "skip it", "conversation_id": conversation_id},
            llm,
        )
        self.assertIn(
            "Not executed", self.tool_messages(conversation_id)["c2"]["message"]
        )
        self.assertEqual(
            [message["role"] for message in llm.requests[1]][-2:], ["tool", "user"]
        )
        with self.assertRaises(ValueError):
            self.drive(
                self.runtime.stream_decision,
                {
                    "approval_id": events[-1]["approval"]["approval_id"],
                    "approved": True,
                },
                llm,
            )
        self.assertEqual(self.started, [])

    def test_invalid_tool_arguments_go_back_to_the_model(self) -> None:
        llm = FakeLLM(tool_chunks(("c1", "echo", '{"bad"')), text_chunks("retried"))
        events = self.drive(self.runtime.stream_chat, {"message": "hi"}, llm)
        tool_event = events[1]["tool_event"]
        self.assertFalse(tool_event["ok"])
        self.assertIn("not valid JSON", tool_event["message"])
        self.assertEqual(events[-1]["type"], "done")

    def test_large_tool_results_are_truncated_in_the_context(self) -> None:
        llm = FakeLLM(tool_chunks(("c1", "big", "{}")), text_chunks("done"))
        events = self.drive(self.runtime.stream_chat, {"message": "hi"}, llm)
        sent = next(message for message in llm.requests[1] if message["role"] == "tool")
        self.assertLess(len(sent["content"]), agent_runtime.TOOL_RESULT_MAX_CHARS + 200)
        self.assertIn("[truncated:", sent["content"])
        tool_event = next(
            event
            for event in self.store.events(events[0]["conversation_id"])
            if event["kind"] == "tool_call"
        )
        self.assertTrue(tool_event["truncated"])

    def test_length_finish_reason_is_reported(self) -> None:
        events = self.drive(
            self.runtime.stream_chat,
            {"message": "hi"},
            FakeLLM(text_chunks("partial", finish="length")),
        )
        self.assertTrue(events[-1]["truncated"])
        self.assertTrue(events[-1]["message"]["content"].startswith("partial"))
        self.assertIn("token limit", events[-1]["message"]["content"])

    def test_unknown_conversation_starts_a_new_one(self) -> None:
        events = self.drive(
            self.runtime.stream_chat,
            {"message": "hi", "conversation_id": "gone"},
            FakeLLM(text_chunks("hello")),
        )
        self.assertTrue(events[0]["reset"])
        self.assertNotEqual(events[0]["conversation_id"], "gone")

    def test_requires_a_single_message_not_client_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "message is required"):
            self.drive(
                self.runtime.stream_chat,
                {"messages": [{"role": "system", "content": "x"}]},
                FakeLLM(),
            )

    def test_artifacts_only_expose_local_api_urls(self) -> None:
        llm = FakeLLM(tool_chunks(("c1", "dl", "{}")), text_chunks("done"))
        events = self.drive(self.runtime.stream_chat, {"message": "hi"}, llm)
        self.assertEqual(
            events[1]["tool_event"]["artifacts"],
            [{"label": "report", "download_url": "/api/attribution/file?name=r.md"}],
        )

    def test_chat_returns_the_aggregated_stream(self) -> None:
        llm = FakeLLM(tool_chunks(("c1", "echo", "{}")), text_chunks("done"))
        with patch.object(self.runtime, "_create_client", return_value=llm):
            result = self.runtime.chat({"message": "hi"})
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["message"]["content"], "done")
        self.assertEqual(
            [event["tool_name"] for event in result["tool_events"]], ["echo"]
        )

    def test_history_trimming_keeps_whole_recent_turns(self) -> None:
        messages = [
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t"}]},
            {"role": "tool", "tool_call_id": "t", "content": "d" * 100},
        ]
        with patch.object(agent_runtime, "HISTORY_MAX_CHARS", 250):
            self.assertEqual(trim_history(messages), messages[2:])
        with patch.object(agent_runtime, "HISTORY_MAX_CHARS", 10):
            self.assertEqual(trim_history(messages), messages[2:])
        self.assertEqual(trim_history(messages), messages)


if __name__ == "__main__":
    unittest.main()
