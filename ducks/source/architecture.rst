============
Architecture
============

This document describes the architecture of FARMS-APP: the definitions,
responsibilities, and relationships between its core components.

Design Principles
=================

**Library first.** ``FARMSApplication`` is a Python class, not a monolithic
executable. The primary usage model is a user script that instantiates the
application, enables extensions, registers windows, and calls ``run()``. The
``farms-app`` CLI entry point is a convenience wrapper around this API.

**The app is a general-purpose ImGui shell.** It provides a windowed environment
with extension discovery, lifecycle management, and layout persistence. It knows
nothing about simulations, physics engines, or neuromechanics.

**Extensions are isolated by default.** Each extension owns its data, its
state, and its windows. Two extensions running side-by-side share nothing
unless they explicitly choose to. A FARMS simulation extension and an
experimental visualizer coexist at the same level — neither is privileged.

**Windows are the primary user-facing API.** Researchers who want to add a
custom plot or analysis panel to an existing extension do not need to write a
full extension. They subclass ``BaseWindow``, register it on the extension they
want to extend, and the window accesses the extension's data through its
reference. Writing a window is the lowest-barrier way to extend the app.

**The framework provides contracts, not behaviour.** Base classes
(``BaseExtension``, ``BaseWindow``) define the interface that the app uses to
manage extensions and windows. What happens inside those interfaces is entirely
up to the extension author.


Core Concepts
=============

Application
-----------

The ``FARMSApplication`` is the outermost shell. It owns:

- The **backend** (platform + renderer — currently GLFW + OpenGL2).
- The **ExtensionManager** — discovery, enable/disable, lifecycle dispatch.
- The **main menu bar** — a single menu bar rendered once per frame.
- The **main loop** — poll events, begin frame, dispatch to extensions, end
  frame.

The application does not own simulation state, domain logic, or
extension-specific data. It is a host.

There are two ways to use it:

**As a library** (primary interface) — users write a script in their own
project:

.. code-block:: python

   # app.py — in the user's research project
   from farms_app.core.application import FARMSApplication
   from farms_app.core.options import ApplicationOptions

   app = FARMSApplication(ApplicationOptions(title="My Experiment"))

   # Enable extensions
   app.extension_manager.enable("farmsim")

   # Get the extension and add custom windows to it
   farmsim = app.extension_manager.get("farmsim")
   farmsim.register_window(MyGaitDiagram(farmsim))
   farmsim.register_window(MyNetworkPlot(farmsim))

   app.run()

**As a CLI** (convenience) — launches with default extensions:

.. code-block:: bash

   farms-app

The CLI is a thin wrapper that calls the same API with default options. The
library interface is where the real power lives — users compose exactly the
environment they need for their research.

Extension
---------

An extension is a self-contained unit of functionality. It is discovered via
Stevedore entry points, instantiated by the ``ExtensionManager``, and
interacted with through a defined lifecycle.

**What an extension owns:**

- Its internal state (simulation data, configuration, buffers — whatever it
  needs).
- Its windows — created via ``register_window()``, each receiving a reference
  to the extension.
- Its menus — rendered inside its ``MainExtensionWindow`` menu bar.

**What an extension does not own:**

- The main menu bar (the app owns this; extensions contribute indirectly
  through their category and visibility state).
- The layout (the app manages window discovery and docking).
- Other extensions' data.

Extension Categories
~~~~~~~~~~~~~~~~~~~~

Categories describe an extension's relationship to the host application:

**UIExtension**
   Modifies core application-level UI: status bars, toolbars, overlays.
   Does not get a ``MainExtensionWindow`` with a dockspace. Has access to the
   app's rendering surface directly (e.g., viewport side bars).

**WorkflowExtension**
   Domain-specific workflows. Automatically gets a ``MainExtensionWindow`` with
   a dockspace and menu bar. Intended for extensions that depend on FARMS
   packages, but the framework does not enforce this — any domain can use this
   category.

**CustomExtension**
   Standalone utilities. Same capabilities as ``WorkflowExtension`` (gets a
   ``MainExtensionWindow`` with dockspace), but semantically signals that the
   extension is independent of FARMS and has no domain dependencies.

The distinction between ``WorkflowExtension`` and ``CustomExtension`` is
organisational, not structural. Both get a dockspace and follow the same
lifecycle. The category is used for menu grouping and to signal intent to
users browsing available extensions.

Extension Lifecycle
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Discovery (Stevedore)
       │
       ▼
   ExtensionManager.enable(name)
       │
       ├── Instantiate extension class
       ├── Call on_enable()
       │       └── Extension may register initial windows here
       │
       ▼
   ┌─── Per frame (while enabled) ───┐
   │                                  │
   │   on_update()   ← state changes, simulation stepping     │
   │       │                          │
   │   on_event()    ← input handling │
   │       │                          │
   │   render()      ← draw UI       │
   │       ├── on_render()            │
   │       └── iterate windows → window._render()             │
   │                                  │
   └──────────────────────────────────┘
       │
   ExtensionManager.disable(name)
       │
       ├── Call on_disable()
       ├── Call cleanup()
       └── Remove from enabled registry

Extensions may register additional windows at any point during their lifetime
— not only during ``on_enable()``. For example, a simulation extension might
register analysis windows only after a model has been loaded. Windows registered
later are picked up by the rendering loop on the next frame.

Window
------

A window is a view into an extension's data. It is the **primary extension
point for end users** — researchers who want to add a custom plot, analysis
panel, or interaction widget to an existing extension.

Writing a window requires no knowledge of the extension system, Stevedore, or
the application lifecycle. The user subclasses ``BaseWindow``, implements
``on_render()``, and registers it on the extension whose data they want to
visualise.

.. code-block:: python

   from farms_app.core.window import BaseWindow
   from imgui_bundle import implot

   class GaitDiagramWindow(BaseWindow):
       """User-written window — added to the FARMS extension."""

       def __init__(self, extension):
           super().__init__("Gait Diagram", extension)

       def on_initialize(self):
           pass  # called once when the extension decides data is ready

       def on_render(self):
           sensors = self._extension.farms_data.sensors
           # ... render gait diagram using implot

The window gets its data through ``self._extension`` — the reference to the
parent extension passed at construction. This is the only coupling: the window
knows the extension's public attributes, nothing else.

A user's ``app.py`` then wires it up:

.. code-block:: python

   app.extension_manager.enable("farmsim")
   farmsim = app.extension_manager.get("farmsim")
   farmsim.register_window(GaitDiagramWindow(farmsim))
   app.run()

The window docks into the extension's dockspace, appears in the extension's
Window menu, and participates in layout persistence — all automatically through
the ``BaseWindow`` contract.

**BaseWindow contract:**

- ``on_initialize()`` — called once, after the extension decides the window is
  ready. Not called by the framework automatically — the extension controls
  when initialization happens.
- ``on_render()`` — called every frame (guarded by ``_initialized`` and
  ``_visible``).
- Unique ID: ``{window_name}##{extension_name}`` — guarantees no collisions
  between extensions.
- Visibility: managed via ``_visible`` flag, togglable from the View menu.
- Docking: windows can dock into their extension's dockspace.

This two-tier model means:

- **Extension authors** write full extensions (lifecycle, state, menus) when
  they need a new domain or data source.
- **Researchers** write windows when they want to add a view of existing data
  — a much lower barrier.


Reusable Windows
~~~~~~~~~~~~~~~~

Windows should depend on **data attributes**, not on a specific extension
class. A window that renders a MuJoCo scene needs ``self._extension.model``
and ``self._extension.data`` — it does not need to know whether the extension
is a live simulation, a replay, or an optimisation run.

.. code-block:: python

   class ViewerWindow(BaseWindow):
       """Reusable — works with any extension that has .model and .data."""

       def on_render(self):
           render_scene(self._extension.model, self._extension.data)

   class JointPlotWindow(BaseWindow):
       """Reusable — works with any extension that has .farms_data."""

       def on_render(self):
           sensors = self._extension.farms_data.sensors
           # ... plot joint positions

Any extension that exposes the expected attributes can use these windows:

.. code-block:: python

   class FarmsSimExtension(Extension):
       """Live simulation."""

       def enable(self, ctx):
           self.add_window(ViewerWindow("Viewer", self))
           self.add_window(JointPlotWindow("Joints", self))

       def update(self):
           self.sim.step()


   class ReplayExtension(Extension):
       """Replay from saved data."""

       def enable(self, ctx):
           self.add_window(ViewerWindow("Viewer", self))
           self.add_window(JointPlotWindow("Joints", self))

       def load(self, path):
           self.model, self.data, self.farms_data = load_from_hdf5(path)
           self.init_windows()

       def update(self):
           self.frame_index += 1
           self.data = self.recording[self.frame_index]

Same window classes, different extensions, different data sources. A user can
run both side by side from their ``app.py``:

.. code-block:: python

   app = App("Compare Runs")

   sim = FarmsSimExtension()
   replay = ReplayExtension()

   app.add(sim)
   app.add(replay)
   app.run()

With a global dockspace, the live viewer and replay viewer sit next to each
other. Same window class, two instances, two data sources.

This leads to a natural separation in the codebase:

- **Reusable windows** live in shared modules (e.g., ``farms_app/views/``).
  They depend on data attributes, not on specific extension classes. Any
  extension that exposes the right attributes can use them.
- **Extension-specific logic** (simulation stepping, model loading, remote job
  management) lives in the extension. The extension is the data source and
  controller. Windows are the lenses.

Extensions that need specialised, one-off windows still define them alongside
the extension code. The shared views module is for windows that are useful
across multiple extensions.


Background Work
~~~~~~~~~~~~~~~

Extensions that perform long-running or remote operations (optimisation jobs,
data loading, remote simulation) must not block the main loop. The ``update()``
method is called every frame — it must return quickly.

The extension manages its own concurrency. A common pattern:

.. code-block:: python

   class OptimizationExtension(Extension):

       def enable(self, ctx):
           self.results = queue.Queue()
           self.history = []
           self.worker = None

       def launch(self, config, remote_host):
           self.worker = threading.Thread(
               target=poll_remote, args=(remote_host, self.results)
           )
           self.worker.start()

       def update(self):
           # Non-blocking drain of results arrived since last frame
           while not self.results.empty():
               result = self.results.get_nowait()
               self.history.append(result)

       def disable(self):
           if self.worker:
               self.worker.join()

Windows render ``self.history`` — they don't know or care that the data came
from a remote machine. The framework does not provide async primitives; the
extension owns its threading. This keeps the framework simple and avoids
imposing a concurrency model.


Open Questions
==============

The following concerns need to be resolved before implementation. They affect
the ``BaseExtension`` and ``BaseWindow`` contracts that everything else builds
on.

1. Window initialisation timing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a user registers a window from ``app.py`` before ``run()``, the
extension's data is not yet available — e.g., ``farmsim.farms_data`` is
``None`` until an experiment is loaded through the GUI. Every user-written
window would need to guard against missing data in ``on_render()``.

The framework should make this easy. Options:

- Windows are not rendered until ``initialize()`` is called, and the extension
  defers ``initialize()`` until data is ready.
- Windows registered from ``app.py`` are queued and initialised by the
  extension at the appropriate time (e.g., after model load).
- ``on_render()`` is simply not called when ``_initialized`` is ``False`` (this
  is already the case — but who calls ``initialize()`` for user-registered
  windows?).

2. Window initialisation ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently ``ExtensionManager.enable()`` iterates ``extension.windows`` and
calls ``initialize()`` on each. But:

- Windows registered from ``app.py`` after ``enable()`` miss this.
- Windows registered late by the extension itself (after loading a model) also
  require manual ``initialize()`` calls.

The architecture should clarify: **the extension always owns initialisation of
its windows.** The manager should not iterate and initialise — the extension
decides when each window is ready. This shifts responsibility clearly but
needs to be documented as part of the contract.

3. WorkflowExtension vs CustomExtension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The distinction is described as organisational, not structural — both have
identical capabilities. But ``pendulum.py`` (which runs a full MuJoCo + FARMS
simulation) subclasses ``CustomExtension``. If the author of the most
FARMS-heavy extension chose the "wrong" category, the distinction is confusing.

Options:

- **Merge into one class** with a ``category`` metadata field (string or enum)
  used for menu grouping. One base class, one set of capabilities.
- **Make the distinction structural** — give ``WorkflowExtension`` something
  ``CustomExtension`` does not (e.g., access to a shared FARMS session). But
  this conflicts with the isolation principle.
- **Keep both** but sharpen the documentation so the choice is obvious.

4. Per-extension dockspaces vs shared dockspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ``WorkflowExtension`` / ``CustomExtension`` gets its own
``MainExtensionWindow`` with its own dockspace. With multiple extensions
enabled, the UI fragments into isolated workspaces. A user cannot easily
arrange a MuJoCo viewer from one extension next to an analysis panel from
another — they live in separate dockspaces.

Options:

- **Keep per-extension dockspaces** — each extension is a self-contained
  workspace. Simple model, but limits cross-extension layout flexibility.
- **Single global dockspace** — all windows dock into one shared space with
  extension-scoped IDs. More flexible layout, but extensions lose their visual
  boundary.
- **Hybrid** — extensions get a default dockspace, but windows can opt out and
  dock globally.

5. Extension back-reference for dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dependency mechanism requires ``self.get_extension("farmsim")``, but
``BaseExtension`` has no reference to the ``ExtensionManager``.

Options:

- The manager injects itself (or a lookup function) into extensions at enable
  time.
- Dependencies are resolved by the manager and passed as arguments to
  ``on_enable()``.
- Extensions receive a lightweight context object with a ``get_extension()``
  method.

6. Lifecycle uniformity across categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``UIExtension`` gets only ``render()``. ``WorkflowExtension`` and
``CustomExtension`` get ``on_update() → on_event() → render()``. If the
categories are organisational (not structural), the lifecycle difference is
inconsistent.

Options:

- **Uniform lifecycle** — all extensions get ``on_update() → on_event() →
  render()``. ``UIExtension`` simply has no-op defaults for ``on_update()``
  and ``on_event()``.
- **Keep the split** — but acknowledge that the categories are structural,
  not just organisational, and document why UI extensions have a simpler
  lifecycle.

MainExtensionWindow
~~~~~~~~~~~~~~~~~~~

``WorkflowExtension`` and ``CustomExtension`` automatically create a
``MainExtensionWindow`` — a top-level window that provides:

- A **dockspace** where child windows can dock.
- A **menu bar** where the extension renders its own menus (File, Simulation,
  Window, etc.) via ``render_menu()``.

This means each extension with a ``MainExtensionWindow`` is visually a
self-contained workspace. A FARMS simulation extension has its own dockspace
with a MuJoCo viewer, analysis panels, and property editors docked inside it.
A pendulum extension has its own separate dockspace with its own windows.

``UIExtension`` does not get a ``MainExtensionWindow`` because it operates on
the application-level surface (status bars, overlays) rather than in its own
workspace.

Backend
-------

The backend abstracts platform and rendering concerns:

**Platform backend** (e.g., GLFW)
   Window creation, event polling, swap buffers.

**Renderer backend** (e.g., OpenGL2)
   Frame begin/end, draw data rendering.

The ``BackendManager`` uses a factory pattern to select and initialize the
appropriate backend combination. Currently only GLFW + OpenGL2 is implemented.

To add a new backend: implement the ``BaseBackend`` interface and register it
in the ``BackendManager``.


Menu System
===========

The application renders a **single main menu bar** per frame. Its structure:

**View**
   - Per-category submenus (UI, Workflow, Custom) listing enabled extensions
     with visibility toggles.
   - Layout actions (dock all windows to extension, theme selection).

**Extensions**
   - Lists all discovered extensions (enabled and disabled) with toggles to
     enable/disable at runtime.

**Debug**
   - Metrics window toggle.
   - Log level selection.

Extension-specific menus (File > Open, Simulation > Run, etc.) live inside the
extension's ``MainExtensionWindow`` menu bar — not in the application's main
menu bar. This keeps the main menu bar stable regardless of which extensions
are loaded.


Input System
============

The input system (``farms_app/core/inputs.py``) manages global shortcuts and
provides helpers for extensions that need focus-aware input handling.

Current Design
--------------

**InputManager** is created by the application and processed once per frame,
before any extension's ``on_event()`` is called.

**Global shortcuts** are registered with a key, optional modifiers, and a
callback. They fire regardless of which window is focused. They are skipped
when ImGui is capturing text input (e.g., the user is typing in a text field).

.. code-block:: python

   app.input_manager.register(
       imgui.Key.space, self.toggle_playback,
       description="Play/Pause", mod_ctrl=False,
   )

**Extension input** is handled in ``on_event()``, called on every enabled
extension every frame. Extensions check ImGui input state directly. The
framework provides helpers to check focus:

.. code-block:: python

   def on_event(self):
       if not InputManager.is_extension_focused(self):
           return
       # Only process input when one of our windows is focused
       ...

**Current dispatch order:**

1. ``input_manager.process()`` — global shortcuts
2. ``extension_manager.tick(dt)`` — calls ``on_event()`` on all enabled extensions

Future: Event Layers
--------------------

The input system is designed to evolve toward prioritized dispatch with
propagation/consumption semantics:

**Layer 1 — Global shortcuts** (current)
   App-level shortcuts that always fire. Already implemented.

**Layer 2 — Focused extension dispatch**
   Only the extension that owns the currently focused window receives
   ``on_event()``. Other extensions are skipped. This avoids multiple
   extensions competing for the same keyboard input.

**Layer 3 — Focused window dispatch**
   Input is delivered to the specific focused window, not just the
   extension. Windows can consume events (stop propagation) or pass them
   up to the extension.

**Layer 4 — Named action system**
   Instead of checking raw keys, extensions register named actions
   (``"play"``, ``"step_forward"``, ``"open_file"``) with default key
   bindings. The framework dispatches actions, not keys. This enables:

   - Key rebinding by the user
   - Conflict detection between extensions
   - Automatic shortcut display in menus

Each layer builds on the previous one. The current implementation (Layer 1 +
all-extensions ``on_event()``) is the minimal useful foundation.


Rendering Flow
==============

.. code-block:: text

   FARMSApplication.run()
   │
   └── while not should_close():
       │
       ├── compute dt
       ├── fps_idling()            ← throttle when idle
       ├── backend.poll_events()
       ├── backend.begin_frame()   ← includes dock_space_over_viewport()
       │
       ├── render_menu()           ← extension.menu() + app menus (View, Debug, Extensions)
       │
       ├── input_manager.process() ← global shortcuts
       │
       ├── ExtensionManager.tick(dt)
       │   │
       │   └── for each enabled extension:
       │       │
       │       ├── on_update(dt)   ← simulation stepping (decoupled from frame rate)
       │       ├── on_event()      ← input handling
       │       └── render()
       │           ├── on_render()
       │           └── for each window:
       │               └── window._render()
       │
       ├── backend.end_frame()
       │
       └── [on exit] cleanup all extensions, cleanup backend


Extension Data Isolation
========================

Extensions do not share state. Each extension owns its data and decides what
to expose to its windows.

.. code-block:: text

   ┌─────────────────────────┐     ┌─────────────────────────┐
   │  FarmsSimExtension      │     │  PendulumExtension      │
   │                         │     │                         │
   │  ┌─ model ─┐            │     │  ┌─ model ─┐            │
   │  │ MuJoCo  │            │     │  │ MuJoCo  │            │
   │  │ model   │            │     │  │ model   │            │
   │  │ data    │            │     │  │ data    │            │
   │  └─────────┘            │     │  └─────────┘            │
   │  ┌─ farms ─┐            │     │                         │
   │  │ sensors │            │     │  ┌─ windows ──────────┐ │
   │  │ network │            │     │  │ ViewerWindow       │ │
   │  └─────────┘            │     │  │  → self._extension │ │
   │                         │     │  └────────────────────┘ │
   │  ┌─ windows ──────────┐ │     └─────────────────────────┘
   │  │ MuJoCoWindow       │ │
   │  │  → self._extension │ │     Windows access data through
   │  │ AnalysisWindow     │ │     their extension reference.
   │  │  → self._extension │ │     No cross-extension access.
   │  └────────────────────┘ │
   └─────────────────────────┘

Cross-extension data access is possible through **extension dependencies**.
An extension can declare dependencies on other extensions via
``get_dependencies()``. The framework ensures dependencies are enabled first
and provides a reference through ``get_extension(name)``.

.. code-block:: python

   class MyAnalysisExtension(CustomExtension):
       def get_dependencies(self):
           return ["farmsim"]

       def on_enable(self):
           farmsim = self.get_extension("farmsim")
           self.register_window(MyPlotWindow(self, farmsim.farms_data))

However, the more common pattern for users who just want to add a window to
an existing extension is the simpler approach: register the window directly on
the extension from their ``app.py`` script (see `Window`_). Dependencies are
for cases where a separate extension needs its own lifecycle and state while
reading data from another extension.


Shared Utilities
================

Code that is reusable across extensions but not part of the framework contract
lives in shared utility modules. These are imported by extensions that need
them — the framework does not depend on them.

Examples:

- **MuJoCo viewport**: Framebuffer creation, scene rendering, camera controls,
  keyboard/mouse interaction. Used by any extension that embeds a MuJoCo
  viewer.
- **Color utilities**: Color manipulation helpers.
- **Path utilities**: Project root resolution, asset paths.

Shared utilities are not extensions and not windows. They are plain Python
modules that extensions import and use as building blocks.


Registration and Discovery
==========================

Extensions are registered as Stevedore entry points in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."farms.app.extension"]
   status_bar = "farms_app.extensions.ui.status_bar:StatusBarExtension"
   pendulum = "farms_app.extensions.custom.pendulum:PendulumExtension"
   farmsim = "farms_app.extensions.workflow.farmsim:FarmsSimulatorExtension"

The ``ExtensionManager`` discovers all entry points at startup, validates that
each points to a ``BaseExtension`` subclass, and makes them available for
enabling. Extensions are not instantiated until explicitly enabled.

Auto-enabling of default extensions (e.g., ``status_bar``) is configured in
``ApplicationOptions``, not hardcoded in the application loop.
