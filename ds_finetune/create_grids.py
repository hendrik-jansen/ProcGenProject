import random
import json

# Anzahl der Beispiele, die du generieren möchtest
anzahl_grids = 100000

# Ziel-Dateipfad
dateipfad = "random_grids.json"

def generate_random_grid():
    """Erstellt ein 25x25 Grid mit 0 und 1 als String"""
    grid = ''.join(random.choice('01') for _ in range(625))
    start = f" starting position: ({random.randint(0, 24)},{random.randint(0,24)})"
    end = f" and ending position: ({random.randint(0, 24)},{random.randint(0,24)})"
    grid += start + end
    return grid

# Schreiben in die Datei
with open(dateipfad, 'w', encoding='utf-8') as f:
    for i in range(anzahl_grids):
        if i % 10000 == 0:
            print(f"iteration: {i}")
        sample = {
            "instruction": "Create a 25x25 grid via returning 625 0's or 1's. Also specify a starting position and a goal as a tuple of two integers.",
            "output": generate_random_grid()
        }
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"{anzahl_grids} Grids wurden in {dateipfad} gespeichert.")
