import pytest
from unittest.mock import patch, MagicMock, call
from colorama import Fore, Back, Style
from friday.debuggers.debug_chat import debug_chat


class TestDebugChat:
    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_single_user_message(self, mock_system, mock_print):
        messages = [{"role": "user", "content": "Hello"}]

        debug_chat(messages)

        mock_system.assert_called_once()
        mock_print.assert_called_once()

        printed_output = mock_print.call_args[0][0]
        assert "user" in printed_output
        assert "Hello" in printed_output

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_multiple_messages(self, mock_system, mock_print):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        debug_chat(messages)

        assert mock_print.call_count == 2

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_string_content(self, mock_system, mock_print):
        messages = [{"role": "user", "content": "test message"}]

        debug_chat(messages)

        mock_print.assert_called_once()
        printed = mock_print.call_args[0][0]
        assert "test message" in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_list_content(self, mock_system, mock_print):
        messages = [
            {
                "role": "user",
                "content": [{"text": "part1"}, {"text": "part2"}],
            }
        ]

        debug_chat(messages)

        mock_print.assert_called_once()
        printed = mock_print.call_args[0][0]
        assert "part1" in printed
        assert "part2" in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_dict_content(self, mock_system, mock_print):
        messages = [{"role": "assistant", "content": {"text": "response text"}}]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert "response text" in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_user_role_uses_green(self, mock_system, mock_print):
        messages = [{"role": "user", "content": "test"}]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert Back.GREEN in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_assistant_role_uses_red(self, mock_system, mock_print):
        messages = [{"role": "assistant", "content": "response"}]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert Back.RED in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_system_role_uses_blue(self, mock_system, mock_print):
        messages = [{"role": "system", "content": "system message"}]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert Back.BLUE in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_function_call_type_uses_magenta(self, mock_system, mock_print):
        messages = [{"type": "function_call", "content": "calling function"}]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert Back.MAGENTA in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_in_progress_status_uses_yellow(self, mock_system, mock_print):
        messages = [
            {
                "role": "assistant",
                "content": "working",
                "status": "in_progress",
            }
        ]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert Fore.YELLOW in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_uses_name_if_no_content(self, mock_system, mock_print):
        messages = [{"role": "assistant", "name": "function_output"}]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert "function_output" in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_empty_content_returns_empty_string(
        self, mock_system, mock_print
    ):
        messages = [{"role": "user"}]

        debug_chat(messages)

        mock_print.assert_called_once()

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_clears_screen_windows(self, mock_system, mock_print):
        messages = [{"role": "user", "content": "test"}]

        with patch("friday.debuggers.chat_debuggers.os.name", "nt"):
            debug_chat(messages)

        mock_system.assert_called_with("cls")

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_clears_screen_unix(self, mock_system, mock_print):
        messages = [{"role": "user", "content": "test"}]

        with patch("friday.debuggers.chat_debuggers.os.name", "posix"):
            debug_chat(messages)

        mock_system.assert_called_with("clear")

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_nested_list_content(self, mock_system, mock_print):
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "first"},
                    "second",
                    {"text": "third"},
                ],
            }
        ]

        debug_chat(messages)

        printed = mock_print.call_args_list
        assert len(printed) >= 1

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_preserves_color_reset(self, mock_system, mock_print):
        messages = [{"role": "user", "content": "test"}]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert Style.RESET_ALL in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_empty_messages_list(self, mock_system, mock_print):
        messages = []

        debug_chat(messages)

        mock_system.assert_called_once()
        mock_print.assert_not_called()

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_function_call_output_type(self, mock_system, mock_print):
        messages = [
            {
                "type": "function_call_output",
                "content": "output from function",
            }
        ]

        debug_chat(messages)

        printed = mock_print.call_args[0][0]
        assert Back.MAGENTA in printed
        assert "output from function" in printed

    @patch("friday.debuggers.chat_debuggers.print")
    @patch("friday.debuggers.chat_debuggers.os.system")
    def test_debug_chat_message_order_preserved(self, mock_system, mock_print):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]

        debug_chat(messages)

        assert mock_print.call_count == 3
        # Check order of calls
        calls = mock_print.call_args_list
        assert "first" in calls[0][0][0]
        assert "second" in calls[1][0][0]
        assert "third" in calls[2][0][0]
