#Laboratorio No.3
# Autores: Johan Alejanro Espitia, Juan Esteban Villamil

# Importar librerias
from dbfread import DBF
from mlxtend import frequent_patterns
import pandas as pd

# Ruta al archivo .DBF
file_path1 = 'C:/Users/johan/OneDrive/Mayor/Programming/Major/Materias/Analitica Datos/Analitica/martes/fc02042024.DBF'
file_path2 = 'C:/Users/johan/OneDrive/Mayor/Programming/Major/Materias/Analitica Datos/Analitica/martes/fc12032024.DBF'

# Leer el archivo .DBF
table1 = DBF(file_path1, load=True)
table2 = DBF(file_path2, load=True)

# Crear un DataFrame
df1 = pd.DataFrame(iter(table1))
df2 = pd.DataFrame(iter(table2))

# Mostrar las primeras 5 filas
print(df1.head())
print(df2.head())

# Mostrar las columnas
print(df1.columns)
print(df2.columns)

#Usando mlxtend encontramos la relacion entre los productos vendidos
#y la cantidad de veces que se vendieron juntos
# Convert the DataFrame to a transaction list
transactions = df1.values.tolist()

# Apply one-hot encoding to the transaction list
one_hot_encoded = frequent_patterns.transactionlist_to_dataframe(transactions, colnames=df1.columns)

# Find frequent itemsets using the Apriori algorithm
frequent_itemsets = frequent_patterns.apriori(one_hot_encoded, min_support=0.1, use_colnames=True)

# Print the frequent itemsets
print(frequent_itemsets)