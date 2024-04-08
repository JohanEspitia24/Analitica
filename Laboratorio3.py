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

