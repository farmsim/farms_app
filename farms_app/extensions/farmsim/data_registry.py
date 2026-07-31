""" FARMS-specific data registry builder.

Walks a FARMS simulation and populates a DataRegistry with all
plottable signals from sensors and network.
"""
import numpy as np
from farms_app.plots.data_registry import DataRegistry, DataSource
from farms_core import pylog
from farms_core.sensors.sensor_convention import sc


def build_registry(sim, network=None) -> DataRegistry:
    """Walk the simulation data and build a registry of all plottable signals.

    Args:
        sim: The FARMS simulation object.
        network: Optional resolved network object. If provided, network signals
            are registered. Pass ``None`` when no network is present.
    """
    registry = DataRegistry()

    farms_data = sim.task.data.animats[0]
    sensors = farms_data.sensors

    _register_joints(registry, sensors.joints)
    _register_links(registry, sensors.links)
    _register_muscles(registry, sensors.muscles)
    _register_contacts(registry, sensors.contacts)

    if network is not None:
        _register_network(network, registry)

    pylog.info(f"Data registry: {len(registry.sources)} signals in {len(registry.groups)} groups")
    return registry


# Joints
_JOINT_CHANNELS = [
    ("position",      sc.joint_position,     "rad"),
    ("velocity",      sc.joint_velocity,     "rad/s"),
    ("torque",        sc.joint_torque,       "N·m"),
    ("torque_x",      sc.joint_torque_x,     "N·m"),
    ("torque_y",      sc.joint_torque_y,     "N·m"),
    ("torque_z",      sc.joint_torque_z,     "N·m"),
    ("cmd_position",  sc.joint_cmd_position, "rad"),
    ("cmd_velocity",  sc.joint_cmd_velocity, "rad/s"),
    ("cmd_torque",    sc.joint_cmd_torque,   "N·m"),
    ("torque_active", sc.joint_torque_active, "N·m"),
]


def _register_joints(registry, joints):
    if joints is None or len(joints.names) == 0:
        return
    arr = joints.array
    for i, name in enumerate(joints.names):
        for ch_name, ch_idx, unit in _JOINT_CHANNELS:
            source_name = f"joints/{name}/{ch_name}"
            _idx, _ch = i, ch_idx
            registry.add(DataSource(
                name=source_name,
                group="joints",
                unit=unit,
                accessor=lambda a=arr, j=_idx, c=_ch: a[:, j, c],
            ))


# Links
_LINK_CHANNELS = [
    ("com_pos_x", sc.link_com_position_x, "m"),
    ("com_pos_y", sc.link_com_position_y, "m"),
    ("com_pos_z", sc.link_com_position_z, "m"),
    ("com_vel_x", sc.link_com_velocity_lin_x, "m/s"),
    ("com_vel_y", sc.link_com_velocity_lin_y, "m/s"),
    ("com_vel_z", sc.link_com_velocity_lin_z, "m/s"),
]


def _register_links(registry, links):
    if links is None or len(links.names) == 0:
        return
    arr = links.array
    for i, name in enumerate(links.names):
        for ch_name, ch_idx, unit in _LINK_CHANNELS:
            source_name = f"links/{name}/{ch_name}"
            _idx, _ch = i, ch_idx
            registry.add(DataSource(
                name=source_name,
                group="links",
                unit=unit,
                accessor=lambda a=arr, j=_idx, c=_ch: a[:, j, c],
            ))


# Muscles
_MUSCLE_CHANNELS = [
    ("excitation",      sc.muscle_excitation,     ""),
    ("activation",      sc.muscle_activation,     ""),
    ("fiber_length",    sc.muscle_fiber_length,   "m"),
    ("fiber_velocity",  sc.muscle_fiber_velocity, "m/s"),
    ("active_force",    sc.muscle_active_force,   "N"),
    ("passive_force",   sc.muscle_passive_force,  "N"),
    ("mtu_length",      sc.muscle_tendon_unit_length, "m"),
    ("mtu_force",       sc.muscle_tendon_unit_force,  "N"),
    ("force_length",    sc.muscle_force_length,   ""),
    ("force_velocity",  sc.muscle_force_velocity, ""),
    ("Ia_feedback",     sc.muscle_Ia_feedback,    ""),
    ("II_feedback",     sc.muscle_II_feedback,    ""),
    ("Ib_feedback",     sc.muscle_Ib_feedback,    ""),
]


def _register_muscles(registry, muscles):
    if muscles is None or len(muscles.names) == 0:
        return
    arr = muscles.array
    for i, name in enumerate(muscles.names):
        for ch_name, ch_idx, unit in _MUSCLE_CHANNELS:
            source_name = f"muscles/{name}/{ch_name}"
            _idx, _ch = i, ch_idx
            registry.add(DataSource(
                name=source_name,
                group="muscles",
                unit=unit,
                accessor=lambda a=arr, j=_idx, c=_ch: a[:, j, c],
            ))


# Contacts
_CONTACT_CHANNELS = [
    ("reaction_x", sc.contact_reaction_x, "N"),
    ("reaction_y", sc.contact_reaction_y, "N"),
    ("reaction_z", sc.contact_reaction_z, "N"),
    ("total_x",    sc.contact_total_x,    "N"),
    ("total_y",    sc.contact_total_y,    "N"),
    ("total_z",    sc.contact_total_z,    "N"),
]


def _register_contacts(registry, contacts):
    if contacts is None or len(contacts.names) == 0:
        return
    arr = contacts.array
    for i, name in enumerate(contacts.names):
        for ch_name, ch_idx, unit in _CONTACT_CHANNELS:
            source_name = f"contacts/{name}/{ch_name}"
            _idx, _ch = i, ch_idx
            registry.add(DataSource(
                name=source_name,
                group="contacts",
                unit=unit,
                accessor=lambda a=arr, j=_idx, c=_ch: a[:, j, c],
            ))


# Network
def _register_network(network, registry: DataRegistry) -> None:
    log = network.log

    # Outputs
    if hasattr(log, 'outputs') and log.outputs is not None:
        arr = log.outputs.array
        for name, idx in log.nodes._name_to_index.items():
            source_name = f"network/outputs/{name}"
            _idx = idx
            registry.add(DataSource(
                name=source_name,
                group="network",
                unit="",
                accessor=lambda a=arr, j=_idx: a[:, j],
            ))

    # States (variable number per node, indexed via log.states.indices)
    if hasattr(log, 'states') and log.states is not None:
        arr = log.states.array
        indices = log.states.indices
        node_options = network.options.nodes
        for name, node_idx in log.nodes._name_to_index.items():
            start = int(indices[node_idx])
            end = int(indices[node_idx + 1])
            if start == end:
                continue
            node_opt = node_options[node_idx]
            state_names = (node_opt.state.STATE_NAMES
                           if node_opt.state is not None else [])
            for s in range(end - start):
                col = start + s
                sname = state_names[s] if s < len(state_names) else str(s)
                source_name = f"network/states/{name}_{sname}"
                registry.add(DataSource(
                    name=source_name,
                    group="network",
                    unit="",
                    accessor=lambda a=arr, c=col: a[:, c],
                ))
