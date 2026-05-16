import asyncio
from bleak import BleakClient, BleakScanner

PMD_SVC = "fb005c80-02e7-f387-1cad-8acd2d8df0c8"

async def main():
    print("scanning…")
    devs = await BleakScanner.discover(timeout=8.0)
    polar = next((d for d in devs if d.name and "Polar" in d.name), None)
    if not polar:
        print("no Polar found"); return
    print(f"connecting to {polar.name}")
    async with BleakClient(polar.address) as c:
        svcs = c.services
        pmd_found = any(s.uuid.lower() == PMD_SVC for s in svcs)
        print("services:")
        for s in svcs:
            tag = "  *PMD*" if s.uuid.lower() == PMD_SVC else ""
            print(f"  {s.uuid}{tag}")
            for ch in s.characteristics:
                props = ",".join(ch.properties)
                print(f"      {ch.uuid}  ({props})")
        print(f"\nPMD service present: {pmd_found}")

asyncio.run(main())
