""" Simple signal/callback hook system for extensions. """


class Hook:
    """ A named hook point that holds a list of callbacks. """

    __slots__ = ("_callbacks",)

    def __init__(self):
        self._callbacks = []

    def connect(self, fn):
        """Register a callback."""
        if fn not in self._callbacks:
            self._callbacks.append(fn)

    def disconnect(self, fn):
        """Remove a callback."""
        self._callbacks.remove(fn)

    def fire(self, *args, **kwargs):
        """Call all registered callbacks."""
        for fn in self._callbacks:
            fn(*args, **kwargs)

    def __bool__(self):
        return bool(self._callbacks)

    def __len__(self):
        return len(self._callbacks)


class Hooks:
    """Container of named Hook instances.

    Usage::

        hooks = Hooks("pre_update", "post_update")
        hooks["pre_update"].connect(my_fn)
        hooks["pre_update"].fire(dt)
    """

    def __init__(self, *names):
        self._hooks = {name: Hook() for name in names}

    def __getitem__(self, name):
        return self._hooks[name]

    def __contains__(self, name):
        return name in self._hooks

    def add(self, name):
        """Add a new hook point. No-op if it already exists."""
        if name not in self._hooks:
            self._hooks[name] = Hook()

    def names(self):
        return list(self._hooks.keys())
