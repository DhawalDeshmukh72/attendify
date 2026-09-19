"""
Simulated Wi-Fi / MAC-address presence layer for Attendify.

Simulates the classroom Wi-Fi router / Access Point (AP) environment.
During lecture hours, when a student connects to the campus Wi-Fi network,
their device MAC address is registered in the active AP connection pool.
"""
import random

# In-memory active connected devices pool (MAC -> metadata)
_active_connected_macs = set()


def _random_mac() -> str:
    return ":".join(f"{random.randint(0, 255):02X}" for _ in range(6))


def register_connected_device(mac_address: str):
    """Mark a student device as currently connected to the classroom Wi-Fi AP."""
    if mac_address:
        _active_connected_macs.add(mac_address.strip().upper())


def disconnect_device(mac_address: str):
    if mac_address and mac_address.strip().upper() in _active_connected_macs:
        _active_connected_macs.remove(mac_address.strip().upper())


def get_connected_devices():
    return list(_active_connected_macs)


def simulate_connected_macs(registered_macs):
    """Return all explicitly connected devices + random ambient campus devices."""
    connected = set(_active_connected_macs)
    # Also include previously registered macs probabilistically
    for mac in registered_macs:
        if random.random() < 0.8:
            connected.add(mac.upper())
    
    # Add a few random noise devices
    for _ in range(random.randint(1, 3)):
        connected.add(_random_mac())
    return list(connected)


def verify_mac(student_mac: str, connected_macs) -> bool:
    """Check if the student's MAC is present in the classroom AP connection list."""
    if not student_mac:
        return False
    norm_student = student_mac.strip().upper()
    return norm_student in {m.strip().upper() for m in connected_macs} or norm_student in _active_connected_macs


def generate_mock_mac() -> str:
    """Generate a valid format MAC address for demo/test purposes."""
    return _random_mac()

