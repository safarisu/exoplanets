import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('C:/Users/acer/Desktop/PSC.xlsx')
import matplotlib.pyplot as plt

plt.scatter(data['sy_dist'], data['pl_bmasse'])
plt.xlabel('Odległość od Ziemi (pc)')
plt.ylabel('Masa planety (Masa Ziemi)')
plt.xscale('log')
plt.show()

plt.scatter(data['sy_snum'], data['sy_pnum'])
plt.xlabel('Ilość gwiazd')
plt.ylabel('ilość planet')
plt.show()


plt.scatter(data['disc_year'], data['sy_dist'])
plt.xlabel('Rok odkrycia')
plt.ylabel('Odległość do planety (1 pc ≈ 206265 j.a. ≈ 3,086×10^16 m)')
plt.show()