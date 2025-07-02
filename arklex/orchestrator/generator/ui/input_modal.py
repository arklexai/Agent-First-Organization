"""Input modal UI component for the Arklex task graph generator.

This module provides a modal dialog interface for editing task and step descriptions.
It includes input validation and callback handling for user interactions. The modal
is designed to work within the Textual framework and provides a simple, focused
interface for text input with submit and cancel options.
"""

from collections.abc import Callable
from typing import Union

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Static
from textual.widgets.tree import TreeNode

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class InputModal(Screen):
    """A simple input modal for editing or adding tasks/steps.

    This class provides a modal dialog interface for editing task and step descriptions.
    It includes input validation and callback handling for user interactions.
    The modal displays a title, input field, and action buttons (Submit/Cancel).

    The modal is designed to be non-blocking and uses callbacks to communicate
    results back to the parent application. It automatically closes itself after
    processing button events.

    Attributes:
        title (str): The title of the modal dialog displayed to the user
        default (str): Default value for the input field, shown when modal opens
        result (str): The final result after user interaction (submit value or default)
        node (Union[TreeNode, None]): The tree node being edited, passed to callback
        callback (Union[Callable[[str, Union[TreeNode, None]], None], None]): Function to call after user interaction
        app (Union[App, None]): The parent app instance for screen management

    Methods:
        compose(): Creates the modal UI components with title, input, and buttons
        on_button_pressed(): Handles button press events and processes user input
    """

    def __init__(
        self,
        title: str,
        default: str = "",
        node: Union["TreeNode", None] = None,
        callback: Callable[[str, Union["TreeNode", None]], None] | None = None,
    ) -> None:
        """Initialize the InputModal instance.

        Args:
            title (str): The title of the modal dialog displayed at the top
            default (str): Default value for the input field. Defaults to "".
            node (Union[TreeNode, None]): The tree node being edited. Passed to callback function.
                                   Defaults to None.
            callback (Union[Callable[[str, Union[TreeNode, None]], None], None]): Function to call after
                user interaction. Receives the result string and node as parameters.
                Defaults to None.
        """
        super().__init__()
        self.title = title
        self.default = default
        self.result = default
        self.node = node
        self.callback = callback
        self._app = None

    @property
    def app(self) -> Union["App", None]:
        """Get the app instance.

        Returns the parent app instance for screen management operations.
        """
        return self._app

    @app.setter
    def app(self, value: "App") -> None:
        """Set the app instance.

        Sets the parent app instance, typically called by the Textual framework
        during screen initialization.
        """
        self._app = value

    def compose(self) -> ComposeResult:
        """Create the modal UI components.

        Creates the visual structure of the modal dialog including title, input field,
        and action buttons. The layout uses a vertical container with the title at
        the top, input field in the middle, and horizontal button layout at the bottom.

        The input field is pre-populated with the default value and has the ID
        "input-field" for easy querying. Buttons have IDs "submit" and "cancel"
        for event handling.

        Yields:
            ComposeResult: The composed UI elements in a vertical layout
        """
        yield Vertical(
            Static(self.title, classes="title"),
            Input(value=self.default, id="input-field"),
            Horizontal(
                Button("Submit", id="submit"),
                Button("Cancel", id="cancel"),
                id="buttons",
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events.

        Processes submit and cancel button clicks, updating the result and
        calling the callback function if provided. The method handles both
        submit and cancel actions, always calling the callback with the
        appropriate result value.

        Submit button behavior:
        - Updates result with current input field value
        - Calls callback with the input value and node
        - Closes the modal

        Cancel button behavior:
        - Keeps result as default value
        - Calls callback with default value and node
        - Closes the modal

        Args:
            event (Button.Pressed): The button press event
        """
        if getattr(event.button, "id", None) == "submit":
            # Update result with current input field value on submit
            self.result = self.query_one("#input-field", Input).value
        if self.callback:
            # Always call callback with current result and node
            self.callback(self.result, self.node)
        if hasattr(self, "app") and self.app and hasattr(self.app, "pop_screen"):
            # Close modal by popping it from the screen stack
            self.app.pop_screen()
