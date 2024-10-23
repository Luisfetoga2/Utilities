from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

origins = [
    'http://localhost:8000',
    "http://127.0.0.1:8000",
    'http://localhost:5173',
    'http://localhost:5174',
    'http://localhost:3000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Modelo(BaseModel):
    id: int
    nombre: str
    variable: str
    filename: str 
    tipo: str  # Regresión o Clasificación
    algoritmos: List[str]
    entrenado: bool = False  

modelos = []

opcionesAlgoritmos = {
    'Regresión': [
        'Regresión Lineal', 
        'Regresión Ridge', 
        'Regresión Lasso', 
        'Regresión Polinómica', 
        'ElasticNet', 
        'Bayesian Ridge', 
        'SVR (Máquina de Soporte Vectorial)', 
        'Árbol de Decisión para Regresión', 
        'Random Forest Regressor', 
        'Gradient Boosting Regressor', 
        'K-Neighbors Regressor'
    ],
    'Clasificación': [
        'Regresión Logística', 
        'Máquina de Soporte Vectorial (SVC)', 
        'Árbol de Decisión', 
        'Random Forest Classifier', 
        'Gradient Boosting Classifier', 
        'Naive Bayes', 
        'K-Neighbors Classifier', 
        'Perceptrón', 
        'Red Neuronal (MLPClassifier)'
    ],
    'Agrupación': [
        'K-Means', 
        'DBSCAN', 
        'Mean Shift', 
        'Agglomerative Clustering', 
        'Birch', 
        'Affinity Propagation', 
        'Spectral Clustering', 
        'Gaussian Mixture Models (GMM)'
    ]
}


@app.get("/modelos", response_model=List[Modelo])
def obtener_modelos():
    return modelos

@app.post("/modelos")
async def crear_modelo(
    nombre: str = Form(...), 
    dataset: UploadFile = File(...),
    variable: str = Form(...),
    tipo: str = Form(...),
    algoritmos: str = Form(...)
):
    print("Creando modelo...")
    
    try:
        id = len(modelos) + 1
        # Extract file extension
        file_extension = os.path.splitext(dataset.filename)[1]
        # Create folder under /models: /models/model{id}/
        folder_path = f'models/model{id}'
        os.makedirs(folder_path, exist_ok=True)
        # Construct the filename with the appropriate extension
        filename = f'{folder_path}/dataset{file_extension}'
        # Save file to disk
        with open(filename, 'wb') as file_object:
            file_object.write(dataset.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    # Create a new model with metadata about the file
    nuevo_modelo = Modelo(
        id=len(modelos) + 1, 
        nombre=nombre, 
        variable=variable, 
        filename=dataset.filename, 
        tipo=tipo,
        algoritmos=algoritmos.split(','),
    )
    modelos.append(nuevo_modelo)
    return nuevo_modelo

@app.get("/modelos/{id}", response_model=Modelo)
def obtener_modelo(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return modelo

@app.post("/variables", response_model=List[List[str]])
async def obtener_variables(dataset: UploadFile = File(...)):
    # Determine the type of file and read it into a DataFrame
    if dataset.content_type == 'text/csv' or dataset.filename.endswith('.csv'):
        df = pd.read_csv(dataset.file)
    elif dataset.content_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] or dataset.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(dataset.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Determine whether columns are categorical or numerical
    variables = []
    for col in df.columns:
        tipo = 'categórica' if df[col].nunique() <= 10 else 'numérica'
        variables.append([col, tipo])
    
    return variables

# Retornar todos los algoritmos disponibles
@app.get("/algoritmos", response_model=Dict[str, List[str]])
def obtener_algoritmos():
    return opcionesAlgoritmos
