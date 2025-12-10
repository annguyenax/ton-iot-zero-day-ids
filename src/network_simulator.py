"""
IoT Network Simulator - Generate realistic IoT device traffic
Simulates different types of IoT devices with normal and attack behaviors
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random


class IoTDevice:
    """Base class for IoT devices"""

    def __init__(self, device_id, device_type, ip_address):
        self.device_id = device_id
        self.device_type = device_type
        self.ip_address = ip_address
        self.state = "normal"  # normal, compromised, attacking
        self.packets_sent = 0

    def set_state(self, state):
        """Change device state (normal, compromised, attacking)"""
        self.state = state

    def generate_traffic(self):
        """Generate network traffic - to be overridden by subclasses"""
        raise NotImplementedError


class SmartThermostat(IoTDevice):
    """Smart home thermostat - sends periodic temperature updates"""

    def __init__(self, device_id, ip_address):
        super().__init__(device_id, "thermostat", ip_address)

    def generate_traffic(self):
        """Generate thermostat traffic"""
        if self.state == "normal":
            # Normal: Small packets, periodic, HTTP
            return {
                'src_ip': self.ip_address,
                'dst_ip': '192.168.1.1',  # Gateway
                'src_port': random.randint(49152, 65535),
                'dst_port': 80,
                'protocol': 'TCP',
                'service': 'http',
                'duration': random.uniform(0.05, 0.2),
                'src_bytes': random.randint(50, 200),
                'dst_bytes': random.randint(50, 150),
                'conn_state': 'SF',  # Successful
                'device_type': self.device_type,
                'device_id': self.device_id
            }
        else:  # compromised/attacking
            # Attack: Larger packets, more frequent, unusual ports
            return {
                'src_ip': self.ip_address,
                'dst_ip': f"192.168.1.{random.randint(2, 254)}",  # Scanning
                'src_port': random.randint(1024, 65535),
                'dst_port': random.choice([22, 23, 3389, 445]),  # Attack ports
                'protocol': 'TCP',
                'service': 'other',
                'duration': random.uniform(0.5, 2.0),
                'src_bytes': random.randint(500, 2000),
                'dst_bytes': random.randint(100, 500),
                'conn_state': random.choice(['S0', 'REJ', 'RSTO']),  # Failed connections
                'device_type': self.device_type,
                'device_id': self.device_id
            }


class SmartCamera(IoTDevice):
    """Smart camera - streams video data"""

    def __init__(self, device_id, ip_address):
        super().__init__(device_id, "camera", ip_address)

    def generate_traffic(self):
        """Generate camera traffic"""
        if self.state == "normal":
            # Normal: Large packets (video stream), continuous
            return {
                'src_ip': self.ip_address,
                'dst_ip': '192.168.1.100',  # NVR/Cloud
                'src_port': random.randint(49152, 65535),
                'dst_port': 554,  # RTSP
                'protocol': 'UDP',
                'service': 'rtsp',
                'duration': random.uniform(1.0, 5.0),
                'src_bytes': random.randint(5000, 20000),
                'dst_bytes': random.randint(100, 500),
                'conn_state': 'SF',
                'device_type': self.device_type,
                'device_id': self.device_id
            }
        else:  # DDoS attack
            return {
                'src_ip': self.ip_address,
                'dst_ip': '8.8.8.8',  # External target
                'src_port': random.randint(1024, 65535),
                'dst_port': 80,
                'protocol': 'TCP',
                'service': 'http',
                'duration': random.uniform(0.01, 0.05),  # Very short
                'src_bytes': random.randint(50, 100),  # SYN flood
                'dst_bytes': 0,
                'conn_state': 'S0',  # No response
                'device_type': self.device_type,
                'device_id': self.device_id
            }


class ModbusPLC(IoTDevice):
    """Industrial Modbus PLC"""

    def __init__(self, device_id, ip_address):
        super().__init__(device_id, "plc", ip_address)

    def generate_traffic(self):
        """Generate Modbus traffic"""
        if self.state == "normal":
            # Normal: Read coils/registers
            return {
                'src_ip': self.ip_address,
                'dst_ip': '192.168.1.50',  # SCADA server
                'src_port': random.randint(49152, 65535),
                'dst_port': 502,  # Modbus
                'protocol': 'TCP',
                'service': 'modbus',
                'duration': random.uniform(0.05, 0.15),
                'src_bytes': random.randint(100, 300),
                'dst_bytes': random.randint(100, 300),
                'conn_state': 'SF',
                'device_type': self.device_type,
                'device_id': self.device_id
            }
        else:  # Modbus attack (write to critical registers)
            return {
                'src_ip': self.ip_address,
                'dst_ip': '192.168.1.50',
                'src_port': random.randint(49152, 65535),
                'dst_port': 502,
                'protocol': 'TCP',
                'service': 'modbus',
                'duration': random.uniform(0.1, 0.3),
                'src_bytes': random.randint(300, 1000),  # Larger writes
                'dst_bytes': random.randint(50, 100),
                'conn_state': 'SF',
                'device_type': self.device_type,
                'device_id': self.device_id
            }


class SmartDoorLock(IoTDevice):
    """Smart door lock"""

    def __init__(self, device_id, ip_address):
        super().__init__(device_id, "doorlock", ip_address)

    def generate_traffic(self):
        """Generate door lock traffic"""
        if self.state == "normal":
            # Normal: Infrequent, authentication requests
            return {
                'src_ip': self.ip_address,
                'dst_ip': '192.168.1.1',
                'src_port': random.randint(49152, 65535),
                'dst_port': 443,  # HTTPS
                'protocol': 'TCP',
                'service': 'ssl',
                'duration': random.uniform(0.1, 0.5),
                'src_bytes': random.randint(200, 500),
                'dst_bytes': random.randint(200, 500),
                'conn_state': 'SF',
                'device_type': self.device_type,
                'device_id': self.device_id
            }
        else:  # Brute force attack
            return {
                'src_ip': self.ip_address,
                'dst_ip': '192.168.1.1',
                'src_port': random.randint(49152, 65535),
                'dst_port': 443,
                'protocol': 'TCP',
                'service': 'ssl',
                'duration': random.uniform(0.05, 0.1),  # Fast attempts
                'src_bytes': random.randint(300, 800),
                'dst_bytes': random.randint(100, 300),
                'conn_state': random.choice(['SF', 'RSTO']),
                'device_type': self.device_type,
                'device_id': self.device_id
            }


class IoTNetworkSimulator:
    """Simulates a complete IoT network"""

    def __init__(self):
        self.devices = []
        self.setup_network()

    def setup_network(self):
        """Create IoT devices"""
        # Smart Home
        self.devices.append(SmartThermostat('thermo-001', '192.168.1.101'))
        self.devices.append(SmartThermostat('thermo-002', '192.168.1.102'))
        self.devices.append(SmartCamera('camera-001', '192.168.1.111'))
        self.devices.append(SmartCamera('camera-002', '192.168.1.112'))
        self.devices.append(SmartDoorLock('lock-001', '192.168.1.121'))

        # Industrial
        self.devices.append(ModbusPLC('plc-001', '192.168.1.201'))
        self.devices.append(ModbusPLC('plc-002', '192.168.1.202'))

    def get_device_by_id(self, device_id):
        """Get device by ID"""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None

    def compromise_device(self, device_id):
        """Compromise a device (simulate attack)"""
        device = self.get_device_by_id(device_id)
        if device:
            device.set_state("compromised")
            return True
        return False

    def restore_device(self, device_id):
        """Restore device to normal state"""
        device = self.get_device_by_id(device_id)
        if device:
            device.set_state("normal")
            return True
        return False

    def generate_packet(self, device_id=None):
        """Generate a single packet from a device"""
        if device_id:
            device = self.get_device_by_id(device_id)
            if not device:
                return None
        else:
            # Random device
            device = random.choice(self.devices)

        traffic = device.generate_traffic()
        traffic['timestamp'] = datetime.now()
        device.packets_sent += 1

        return traffic

    def generate_batch(self, n_packets=100, attack_ratio=0.0):
        """
        Generate a batch of packets

        Args:
            n_packets: Number of packets to generate
            attack_ratio: Ratio of attack packets (0.0-1.0)

        Returns:
            DataFrame with generated packets
        """
        packets = []

        # Decide which devices to compromise
        n_compromised = int(len(self.devices) * attack_ratio)
        if n_compromised > 0:
            compromised_devices = random.sample(self.devices, n_compromised)
            for device in compromised_devices:
                device.set_state("compromised")

        # Generate packets
        for _ in range(n_packets):
            packet = self.generate_packet()
            packets.append(packet)

        # Restore all devices
        for device in self.devices:
            device.set_state("normal")

        return pd.DataFrame(packets)

    def get_network_status(self):
        """Get current network status"""
        status = {
            'total_devices': len(self.devices),
            'normal_devices': sum([1 for d in self.devices if d.state == 'normal']),
            'compromised_devices': sum([1 for d in self.devices if d.state == 'compromised']),
            'devices': []
        }

        for device in self.devices:
            status['devices'].append({
                'id': device.device_id,
                'type': device.device_type,
                'ip': device.ip_address,
                'state': device.state,
                'packets_sent': device.packets_sent
            })

        return status


# For testing
if __name__ == "__main__":
    print("IoT Network Simulator Test\n")

    # Create simulator
    sim = IoTNetworkSimulator()

    print(f"Network setup complete: {len(sim.devices)} devices\n")

    # Generate normal traffic
    print("Generating 10 normal packets...")
    for i in range(10):
        packet = sim.generate_packet()
        print(f"{i+1}. {packet['device_type']:10s} | {packet['src_ip']:15s} -> {packet['dst_ip']:15s} | {packet['service']:10s} | {packet['src_bytes']:5d} bytes")

    print("\n" + "="*80)

    # Compromise a device
    print("\nCompromising camera-001...")
    sim.compromise_device('camera-001')

    print("Generating 10 packets from compromised camera...")
    for i in range(10):
        packet = sim.generate_packet('camera-001')
        print(f"{i+1}. {packet['device_type']:10s} | {packet['src_ip']:15s} -> {packet['dst_ip']:15s} | {packet['service']:10s} | {packet['src_bytes']:5d} bytes | {packet['conn_state']}")

    print("\n" + "="*80)

    # Batch generation
    print("\nGenerating batch with 20% attack ratio...")
    df_batch = sim.generate_batch(n_packets=20, attack_ratio=0.2)
    print(df_batch[['device_type', 'src_ip', 'dst_ip', 'service', 'src_bytes', 'conn_state']])

    print("\nNetwork Status:")
    status = sim.get_network_status()
    print(f"Total: {status['total_devices']}, Normal: {status['normal_devices']}, Compromised: {status['compromised_devices']}")
