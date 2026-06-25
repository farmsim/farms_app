""" Generic data registry: a discoverable catalog of plottable signals.

Extensions populate the registry with DataSource entries. Plot windows
use it to discover what data is available and how to access it.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DataSource:
    """A single plottable signal."""
    name: str               # display path, e.g. "joints/elbow/position"
    group: str              # top-level group, e.g. "joints"
    unit: str               # e.g. "rad", "N", "rad/s", ""
    accessor: Callable      # fn() -> 1-D np.ndarray of shape (buffer_size,)


@dataclass
class DataRegistry:
    """Catalog of plottable signals."""
    sources: dict[str, DataSource] = field(default_factory=dict)

    def get(self, name: str) -> Optional[DataSource]:
        return self.sources.get(name)

    def group(self, group_name: str) -> list[DataSource]:
        """All sources belonging to a group."""
        return [s for s in self.sources.values() if s.group == group_name]

    @property
    def groups(self) -> list[str]:
        """Sorted list of unique group names."""
        return sorted(set(s.group for s in self.sources.values()))

    def add(self, source: DataSource):
        """Register a data source."""
        self.sources[source.name] = source
