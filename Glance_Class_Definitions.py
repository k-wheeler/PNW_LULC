class1_dict = {1: "Water",
               2: "Ice/Snow",
               3: "Developed",
               4: "Barren/Sparse",
               5: "Trees",
               6: "Shrubs",
               7: "Herbaceous"}


class2_dict = {0: "Unknown",
               1: "Water",
               2: "Ice/Snow",
               3: "Developed",
               4: "Soil",
               5: "Rock",
               6: "Beach/Sand",
               7: "Deciduous",
               8: "Evergreen",
               9: "Mixed",
               10: "Shrub",
               11: "Grassland",
               12: "Agriculture",
               13: "Moss/Lichen"}


# Classes averaged for the "Subset balanced accuracy" metric in Evaluation.compare_models:
# the average of per-class recall over these classes only, ignoring the rest.
class1_subset = [3, 4, 5, 6, 7]  # Developed, Barren/Sparse, Trees, Shrubs, Herbaceous

class2_subset = [3, 4, 7, 8, 9, 10, 11]  # Developed, Soil, Deciduous, Evergreen, Mixed, Shrub, Grassland