=========
Overview
=========

## Overview

FARMS-app is an extensible immediate-mode graphical user interface application built on
top of imgui and imgui_bundle to provide researchers with an interactive environment for
designing, running, and analyzing neuromechanical simulations of animals and robots.
There will be direct support for **FARMS (Framework for Animal and Robot Modeling and
Simulation)** framework but is not dependency.

While FARMS itself is a powerful command-line framework for neuromechanical modeling, FARMS-app brings this capability into an accessible GUI environment with a particular focus on:

- **Interactive simulation and analysis**: Real-time visualization and control of simulations
- **Extensibility**: A extension architecture that allows users to customize the interface to their specific research needs
- **FARMS pipeline integration**: Seamless access to the full FARMS workflow (modeling, simulation, and analysis)
- **Flexibility**: Framework-agnostic design that doesn't restrict users to FARMS-only
  workflows

## What is FARMS?

FARMS is an open-source, interdisciplinary framework designed to facilitate neuromechanical simulations for studying animal locomotion and bio-inspired robotic systems. The framework provides:

- **Modeling tools**: Design animal and robot bodies with realistic dynamics, including musculoskeletal systems
- **Simulation capabilities**: Physics-based simulations using engines like MuJoCo and Bullet
- **Neural controllers**: Support for CPG networks, reinforcement learning, and other control strategies
- **Analysis tools**: Post-processing, visualization, and comparison with experimental data

For more information about the core FARMS framework, see the [FARMS paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011026).

## Why FARMS-app?

While FARMS excels at programmatic simulation workflows, many researchers benefit from a graphical interface for:

1. **Rapid prototyping**: Quickly test parameters and visualize results without writing code
2. **Interactive analysis**: Explore simulation data with custom visualization tools
3. **Workflow customization**: Extend the interface with domain-specific tools through plugins
4. **Accessibility**: Lower the barrier to entry for researchers new to neuromechanical simulation

FARMS-app achieves this through a modern, extensible architecture built on **imgui_bundle**, providing a responsive, cross-platform GUI experience.

## Key Features

### Extension System

FARMS-app's core strength is its **extension framework**, which allows users to add functionality without modifying the core application. Extensions can:

- Add custom GUI windows and panels
- Integrate data analysis and visualization tools
- Create domain-specific workflows
- Interface with external tools and data sources

The framework supports three types of extensions:

- **UIExtension**: Modify core application UI elements (menus, toolbars, status bars)
- **WorkflowExtension**: Create FARMS-specific workflow tools with access to simulation data
- **CustomExtension**: Build general-purpose utilities independent of FARMS data

### FARMS Integration

FARMS-app provides seamless access to the FARMS pipeline:

- Load and configure FARMS models
- Run simulations interactively
- Access simulation data for analysis
- Visualize results in real-time

Importantly, **the app never restricts users to FARMS** – you can integrate your own data sources, models, and analysis tools through the extension system.

### Cross-Platform Support

Built with modern, cross-platform technologies, FARMS-app runs on Linux, macOS, and Windows, with support for multiple rendering backends where available.

## Who Should Use FARMS-app?

FARMS-app is designed for:

- **Neuroscientists** studying locomotor control and neural circuits
- **Biomechanists** analyzing animal movement and musculoskeletal dynamics
- **Roboticists** developing bio-inspired control systems
- **Researchers** needing interactive tools for neuromechanical simulation

Whether you're an experienced FARMS user looking for a GUI, or a researcher new to neuromechanical simulation, FARMS-app provides an accessible entry point.

## Getting Started

New to FARMS-app? Start with:

1. **Installation Guide**: Get FARMS-app up and running on your system
2. **Quick Start Tutorial**: Run your first simulation in minutes
3. **Extension Development**: Learn how to create custom extensions
4. **API Reference**: Detailed documentation for developers

## Design Philosophy

FARMS-app is built on several core principles:

1. **Modularity**: Extension system allows functionality to be added without core modifications
2. **Flexibility**: Never force users into a single workflow or data format
3. **Developer-Friendly**: Clear APIs and examples make extension development straightforward
4. **Performance**: Built on efficient technologies for responsive interaction
5. **Community-Driven**: Open-source development welcoming contributions

## Project Status

FARMS-app is under active development. Version 0.1 focuses on:

- ✅ Extension discovery and loading
- ✅ Basic GUI framework with imgui_bundle
- 🔧 Window management for extensions
- 🔧 Developer documentation and examples
- 📋 Data access patterns (planned)
- 📋 Advanced extension features (planned)

Check the [roadmap](#roadmap) for details on upcoming features.

## Contributing

FARMS-app is an open-source project and welcomes contributions! Whether you're:

- Reporting bugs
- Suggesting features
- Writing documentation
- Creating extensions
- Contributing code

...your involvement helps make FARMS-app better for the research community.

See the [Contributing Guide](#contributing-guide) to get started.

## License

FARMS-app is released under the [Apache 2.0 License](LICENSE), consistent with the FARMS framework.
