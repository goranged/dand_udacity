import pandas as pd
import csv
import matplotlib.pyplot as plt
node_users = pd.read_csv('node_users.csv', delimiter="|")
fig, ax = plt.subplots()
plt.hist(node_users['quantity'],bins=250)
plt.xlabel('Collaboration to nodes')
plt.ylabel('Users')
plt.title('User collaboration distribution by Nodes')
plt.xlim([0,5774])
plt.grid(True)
plt.subplots_adjust(top=0.5)
plt.show()