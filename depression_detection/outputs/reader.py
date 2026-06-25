# Source - https://stackoverflow.com/a/51465553
# Posted by N.S
# Retrieved 2026-06-17, License - CC BY-SA 4.0

from numpy import load
import os


data = load(r"E:\research\depression_detection\outputs\embeddings.npz")
lst = data.files
for item in lst:
    print(item)
    print(data[item])
