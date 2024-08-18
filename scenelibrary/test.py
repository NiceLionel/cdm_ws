import csv

path = "scenelibrary/select_scene.csv"


with open(path, "r") as f:
    reader = csv.reader(f)
    print(list(reader))
    # for i, rows in enumerate(reader):
    #     if len(rows) == 4:
    #         print(i, rows)
