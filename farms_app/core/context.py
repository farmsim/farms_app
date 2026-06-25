""" Context

Maintains the current running context of the entire application, windows, ...

"""


class ApplicationContext:
    """ Application Context """

    def __init__(self):
        super().__init__()

        self.active_extension: str = None
        self.active_window: str = None

        self.show_metrics_window: bool = False

        self._subscribers = {}

    def subscribe(self, topic: str, callback):
        self._subscribers.setdefault(topic, []).append(callback)

    def publish(self, topic: str, data: Any):
        for callback in self._subscribers.get(topic, []):


class WindowContext:
    """ Window Context """


class ExtensionContext:
    """ Extension Context """
