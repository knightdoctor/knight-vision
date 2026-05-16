"""Minimal Polar PMD ACC stream byte dump.

Connects, requests accelerometer stream at 50 Hz / 8G / 16-bit / 3 axes,
prints the first 8 data packets as raw hex so we can see the actual
frame format and build a parser.
"""
import asyncio
from bleak import BleakClient, BleakScanner

PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA    = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

# CTRL: 0x02 (start) + 0x02 (ACC) + settings as type/len/value bytes
START_ACC = bytes([
    0x02, 0x02,
    0x00, 0x01, 0x34, 0x00,   # sample rate 52 Hz  (0x0034 LE)
    0x01, 0x01, 0x10, 0x00,   # resolution 16 bit
    0x02, 0x01, 0x08, 0x00,   # range 8 G
    0x04, 0x01, 0x03, 0x00,   # 3 channels (X,Y,Z)
])

data_packets = []

def on_ctrl(_s, data):
    print(f"  [CTRL] {data.hex()}")

def on_data(_s, data):
    if len(data_packets) < 8:
        data_packets.append(bytes(data))
        print(f"  [DATA #{len(data_packets)}] len={len(data)} {data[:24].hex()}{'...' if len(data) > 24 else ''}")

async def main():
    devs = await BleakScanner.discover(timeout=8.0)
    polar = next((d for d in devs if d.name and "Polar" in d.name), None)
    if not polar:
        print("no Polar")
        return
    print(f"connecting to {polar.name}")
    async with BleakClient(polar.address) as c:
        print("subscribing data point first")
        await c.start_notify(PMD_DATA, on_data)
        print(f"writing START ACC (no response): {START_ACC.hex()}")
        await c.write_gatt_char(PMD_CONTROL, START_ACC, response=False)
        print("collecting for 6s...")
        await asyncio.sleep(6.0)
        print("stopping")
        try:
            await c.write_gatt_char(PMD_CONTROL, bytes([0x03, 0x02]), response=False)
        except Exception as e:
            print(f"  stop write error: {e}")
        await c.stop_notify(PMD_DATA)
    print(f"\n{len(data_packets)} packets captured")

asyncio.run(main())
