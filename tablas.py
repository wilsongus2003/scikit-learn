from tabulate import tabulate
import pandas as pd
df = pd.read_csv('./data/dset.csv')
grupo = df.groupby(['Marital', 'Course'])
#ventas_por_mes_y_año = grupo.agg({'ventas': 'sum'})
tabla_ventas = pd.pivot_table(values='Marital')
print(tabla_ventas)