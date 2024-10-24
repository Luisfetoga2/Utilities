from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn import neural_network
from sklearn import cluster
from sklearn import mixture
from sklearn import naive_bayes
from sklearn import model_selection
import numpy as np
import joblib
from unidecode import unidecode

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
    fecha_creacion: str = None

    mejor_modelo: str = None
    score: float = None

modelos = []

# Load models from models.txt
# Check if the file exists
if not os.path.exists('models/models.txt'):
    with open('models/models.txt', 'w') as file:
        file.write('')
with open('models/models.txt', 'r') as file:
    line = file.readline()
    while line:
        id, nombre, variable, filename, tipo, algoritmos, entrenado, fecha_creacion, mejor_modelo, score = line.strip().split(';')
        modelos.append(Modelo(id=int(id), nombre=nombre, variable=variable, filename=filename, tipo=tipo, algoritmos=algoritmos.split(','), entrenado=entrenado=='True', fecha_creacion=fecha_creacion, mejor_modelo=mejor_modelo, score=float(score) if score else None))
        line = file.readline()

opcionesAlgoritmos = {
    'Regresión': [
        'Regresión Lineal', 
        'Regresión Ridge', 
        'Regresión Lasso', 
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
    #'Agrupación': [
    #    'K-Means', 
    #    'DBSCAN', 
    #    'Mean Shift', 
    #    'Agglomerative Clustering', 
    #    'Birch', 
    #    'Affinity Propagation', 
    #    'Spectral Clustering', 
    #    'Gaussian Mixture Models (GMM)'
    #]
}

funcionesAlgoritmos = {
    'Regresión Lineal': sklearn.linear_model.LinearRegression(),
    'Regresión Ridge': sklearn.linear_model.Ridge(),
    'Regresión Lasso': sklearn.linear_model.Lasso(),
    'ElasticNet': sklearn.linear_model.ElasticNet(),
    'Bayesian Ridge': sklearn.linear_model.BayesianRidge(),
    'SVR (Máquina de Soporte Vectorial)': sklearn.svm.SVR(),
    'Árbol de Decisión para Regresión': sklearn.tree.DecisionTreeRegressor(),
    'Random Forest Regressor': sklearn.ensemble.RandomForestRegressor(),
    'Gradient Boosting Regressor': sklearn.ensemble.GradientBoostingRegressor(),
    'K-Neighbors Regressor': sklearn.neighbors.KNeighborsRegressor(),
    'Regresión Logística': sklearn.linear_model.LogisticRegression(),
    'Máquina de Soporte Vectorial (SVC)': sklearn.svm.SVC(),
    'Árbol de Decisión': sklearn.tree.DecisionTreeClassifier(),
    'Random Forest Classifier': sklearn.ensemble.RandomForestClassifier(),
    'Gradient Boosting Classifier': sklearn.ensemble.GradientBoostingClassifier(),
    'Naive Bayes': sklearn.naive_bayes.GaussianNB(),
    'K-Neighbors Classifier': sklearn.neighbors.KNeighborsClassifier(),
    'Perceptrón': sklearn.linear_model.Perceptron(),
    'Red Neuronal (MLPClassifier)': sklearn.neural_network.MLPClassifier(),
    #'K-Means': sklearn.cluster.KMeans(),
    #'DBSCAN': sklearn.cluster.DBSCAN(),
    #'Mean Shift': sklearn.cluster.MeanShift(),
    #'Agglomerative Clustering': sklearn.cluster.AgglomerativeClustering(),
    #'Birch': sklearn.cluster.Birch(),
    #'Affinity Propagation': sklearn.cluster.AffinityPropagation(),
    #'Spectral Clustering': sklearn.cluster.SpectralClustering(),
    #'Gaussian Mixture Models (GMM)': sklearn.mixture.GaussianMixture()
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
    algoritmos: str = Form(...),
):    
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
        entrenado=False,
        fecha_creacion=time.strftime('%d-%m-%Y')
    )
    modelos.append(nuevo_modelo)

    # Write on models/models.txt
    with open('models/models.txt', 'a') as file:
        file.write(f'{nuevo_modelo.id};{nuevo_modelo.nombre};{nuevo_modelo.variable};{nuevo_modelo.filename};{nuevo_modelo.tipo};{",".join(nuevo_modelo.algoritmos)};{nuevo_modelo.entrenado};{nuevo_modelo.fecha_creacion};;\n')

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
        if df[col].dtype == 'object':
            tipo = 'categórica'
        else:
            tipo = 'categórica' if df[col].nunique() <= 10 else 'numérica'
        variables.append([col, tipo])
    
    return variables

# Retornar todos los algoritmos disponibles
@app.get("/algoritmos", response_model=Dict[str, List[str]])
def obtener_algoritmos():
    return opcionesAlgoritmos

# Entrenar un modelo
@app.post("/modelos/{id}/entrenar")
def entrenar_modelo(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    if modelo.entrenado:
        raise HTTPException(status_code=400, detail="El modelo ya ha sido entrenado")

    # Cargar el dataset
    file_extension = os.path.splitext(modelo.filename)[1]
    dataset_path = f'models/model{id}/dataset{file_extension}'
    
    try:
        if file_extension == '.csv':
            dataset = pd.read_csv(dataset_path)
        elif file_extension in ['.xls', '.xlsx']:
            dataset = pd.read_excel(dataset_path)
        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo: {str(e)}")

    # Eliminar filas con valores faltantes en la variable objetivo
    dataset = dataset.dropna(subset=[modelo.variable])

    # Separar características y variable objetivo
    X = dataset.drop(columns=modelo.variable)
    y = dataset[modelo.variable]

    # Identificar columnas categóricas y numéricas
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=[np.number]).columns

    # Guardar las columnas para predicciones futuras
    joblib.dump(list(numerical_columns), f'models/model{id}/numerical_columns.joblib')
    joblib.dump(list(categorical_columns), f'models/model{id}/categorical_columns.joblib')

    # Imputar valores faltantes
    X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].median())
    X[categorical_columns] = X[categorical_columns].fillna("Desconocido")

    # Preprocesar variables categóricas
    X[categorical_columns] = X[categorical_columns].applymap(
        lambda x: unidecode(str(x).lower()) if isinstance(x, str) else x
    )

    # Aplicar OneHotEncoder a las columnas categóricas
    encoder = sklearn.preprocessing.OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(X[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    # Guardar el encoder y las columnas codificadas
    joblib.dump(encoder, f'models/model{id}/encoder.joblib')
    joblib.dump(encoder.get_feature_names_out(categorical_columns), f'models/model{id}/encoder_feature_names.joblib')

    # Concatenar datos numéricos y codificados
    X = pd.concat([X[numerical_columns].reset_index(drop=True), encoded_df], axis=1)

    # Escalar las columnas numéricas
    scaler = sklearn.preprocessing.StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Guardar el escalador y las columnas finales
    joblib.dump(scaler, f'models/model{id}/scaler.joblib')
    joblib.dump(list(X.columns), f'models/model{id}/feature_columns.joblib')

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar el mejor modelo
    best_score = 0
    best_model = None
    mejor_modelo = None
    for algoritmo in modelo.algoritmos:
        model = funcionesAlgoritmos[algoritmo]
        
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Evaluar el modelo
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = model
            mejor_modelo = algoritmo

    if best_model is None:
        raise HTTPException(status_code=500, detail="No se pudo entrenar ningún modelo")

    # Guardar el modelo entrenado
    joblib.dump(best_model, f'models/model{id}/trained_model.joblib')

    # Marcar el modelo como entrenado
    modelo.entrenado = True
    modelo.mejor_modelo = mejor_modelo
    modelo.score = best_score

    # Actualizar y guardar el archivo de modelos
    with open('models/models.txt', 'w') as file:
        for m in modelos:
            file.write(f"{m.id};{m.nombre};{m.variable};{m.filename};"
                       f"{m.tipo};{','.join(m.algoritmos)};"
                       f"{m.entrenado};{m.fecha_creacion};{m.mejor_modelo};{m.score}\n")

    return {"message": "Modelo entrenado exitosamente", "mejor_modelo": modelo.nombre}

@app.post("/modelos/{id}/predecir")
def predecir(id: int, datos: dict):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None or not modelo.entrenado:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no entrenado")

    # Load the trained model, scaler, encoder, and feature columns
    model = joblib.load(f'models/model{id}/trained_model.joblib')
    scaler = joblib.load(f'models/model{id}/scaler.joblib')
    encoder = joblib.load(f'models/model{id}/encoder.joblib')
    encoder_feature_names = joblib.load(f'models/model{id}/encoder_feature_names.joblib')
    categorical_columns = joblib.load(f'models/model{id}/categorical_columns.joblib')
    numerical_columns = joblib.load(f'models/model{id}/numerical_columns.joblib')
    feature_columns = joblib.load(f'models/model{id}/feature_columns.joblib')

    # Convert input data to DataFrame
    input_data = pd.DataFrame([datos])

    # Asegurar que las columnas numéricas estén presentes y convertir a 0 si no son numéricas
    numerical_data = input_data[numerical_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Convertir las columnas categóricas a minúsculas y eliminar tildes
    categorical_data = input_data[categorical_columns].astype(str).applymap(
        lambda x: unidecode(x.lower())
    )

    # Aplicar el encoder a los datos categóricos
    encoded_data = encoder.transform(categorical_data)
    encoded_df = pd.DataFrame(encoded_data, columns=encoder_feature_names)

    # Concatenar los datos numéricos y categóricos codificados
    input_data = pd.concat([numerical_data, encoded_df], axis=1)

    # Alinear con las columnas de características del entrenamiento y llenar con 0 si faltan
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # Escalar las columnas numéricas
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

    # Realizar la predicción
    prediction = model.predict(input_data)

    return {"prediccion": prediction.tolist()}

@app.get("/modelos/{id}/parametros")
def obtener_parametros(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None or not modelo.entrenado:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no entrenado")

    numerical_columns = joblib.load(f'models/model{id}/numerical_columns.joblib')
    categorical_columns = joblib.load(f'models/model{id}/categorical_columns.joblib')

    return numerical_columns+categorical_columns
    
   