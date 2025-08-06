import random 
import json
import numpy as np 

dataset_size = 1000
grid_size = 10
num_grids = int(grid_size**2)

data = []

for i in range(dataset_size):
    number = random.randint(0, num_grids)
    list = [(random.randint(0, grid_size-1), random.randint(0, grid_size-1)) for _ in range(number)]
    data_point = {
        "prompt": "Create a maze by returning a list of 2d coordinates. This list indicates where we want to place walls. Also specify a starting position and a goal via coordinates.", 
        "completion": f"{list}\nstart: {(random.randint(0, grid_size-1), random.randint(0, grid_size-1))}\ngoal: {(random.randint(0, grid_size-1), random.randint(0, grid_size-1))}"
    }

    data.append(data_point)

with open("random_grids2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
