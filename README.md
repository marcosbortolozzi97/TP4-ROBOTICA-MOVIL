# TP4 – Robótica Móvil  
## Graph SLAM

---

### **Descripción general**
Este repositorio contiene el desarrollo completo del Trabajo Práctico de SLAM 2D y 3D utilizando GTSAM, incluyendo:  
- Lectura y parseo de archivos .g2o
- Lectura y parseo de archivos .g2o
- Construcción de grafos de factores
- Solución incremental y por lotes
- Visualización de trayectorias
- Scripts para los casos 2D y 3D

---

### **Requisitos previos**
Antes de comenzar, es necesario contar con:  
- Python 3.10
- pip actualizado
  
Se recomienda crear un entorno virtual para realizar la ejecución, 
```bash
python3 -m venv venv
source venv/bin/activate
```
  
Se intslan las dependencias para ejecutar el trabajo, al descargar la carpeta del repositorio dentro se encontrará con el archivo requirements.txt que contiene las dependencias necesarias, ubica el directorio en la carpeta y ejecuta  
```bash
pip install -r requirements.txt
```
  
Si desea salir del entorno virtual
```bash
deactivate
```
  
---

### **Ejecución**
Todos los scripts deben ejecutarse desde la carpeta raíz del trabajo:  
- Batch Solution 2D
```bash
python3 grafo_2D.py
```
Genera como salida imagen 2D_Batch.png
  
  
- Incremental Solution 2D
```bash
python3 grafo_2D_incremental.py
```
Genera como salida imagen 2D_isam.png
  
  
  - Batch Solution 3D
```bash
python3 grafo_3D.py
```
Genera como salida imagen 3D_Batch.png
  
  
  - Incremental Solution 3D
```bash
python3 grafo_3D_incremental.py
```
Genera como salida imagen 3D_isam.png
  
  

