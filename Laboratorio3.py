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
print(df1.head(20))
print(df2.head(20))

# Mostrar las columnas
print(df1.columns)
print(df2.columns)

#Convertirmos los datos a booleanos
df1 = df1.astype(bool)
df2 = df2.astype(bool)

# Mostrar las primeras 5 filas
print(df1.head(20))
print(df2.head(20))

#Usando mlxtend encontramos la relacion entre los productos vendidos entre los df1 y df2
print("Relacion entre los productos vendidos entre los df1 y df2")
print("df1------------------------------------")
frequent_itemsets = frequent_patterns.apriori(df1, min_support=0.6, use_colnames=True)
print(frequent_itemsets)

print("df2------------------------------------")
frequent_itemsets = frequent_patterns.apriori(df2, min_support=0.6, use_colnames=True)
print(frequent_itemsets)
print("Fin del programa")