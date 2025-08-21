import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('points_line.csv')
print(data.head())
studytime = data['studytime'].to_numpy() # Convert to NumPy array
score = data['score'].to_numpy() # Convert to NumPy array
print(studytime)
print(score)