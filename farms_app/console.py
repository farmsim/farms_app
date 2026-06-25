from rich.console import Console
from rich.traceback import install


console = Console()
# Make all uncaught exceptions use Rich formatting automatically
install(show_locals=True)
