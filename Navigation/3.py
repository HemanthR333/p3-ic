import bluetooth

nearby_devices = bluetooth.discover_devices(lookup_names=True)

for addr, name in nearby_devices:
    print(f"Address: {addr}, Name: {name}")