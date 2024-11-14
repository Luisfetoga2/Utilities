from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from scipy import stats
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)
import numpy as np
import joblib
from unidecode import unidecode
from io import BytesIO

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
    parametros: List[str] # Parámetros del modelo (columnas)
    algoritmos: List[str]
    entrenado: bool = False 
    fecha_creacion: str = None

    mejor_modelo: str = None
    score: float = None

modelos = []

# Load models from models.txt
# Chek if folder exists
if not os.path.exists('models'):
    os.makedirs('models')
# Check if the file exists
if not os.path.exists('models/models.txt'):
    with open('models/models.txt', 'w') as file:
        file.write('')
with open('models/models.txt', 'r') as file:
    line = file.readline()
    while line:
        id, nombre, variable, filename, tipo, parametros, algoritmos, entrenado, fecha_creacion, mejor_modelo, score = line.strip().split(';')
        modelos.append(Modelo(id=int(id), nombre=nombre, variable=variable, filename=filename, tipo=tipo, parametros=parametros.split(','), algoritmos=algoritmos.split(','), entrenado=entrenado=='True', fecha_creacion=fecha_creacion, mejor_modelo=mejor_modelo, score=float(score) if score else None))
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
    'Regresión Lineal': sklearn.linear_model.LinearRegression,
    'Regresión Ridge': sklearn.linear_model.Ridge,
    'Regresión Lasso': sklearn.linear_model.Lasso,
    'ElasticNet': sklearn.linear_model.ElasticNet,
    'Bayesian Ridge': sklearn.linear_model.BayesianRidge,
    'SVR (Máquina de Soporte Vectorial)': sklearn.svm.SVR,
    'Árbol de Decisión para Regresión': sklearn.tree.DecisionTreeRegressor,
    'Random Forest Regressor': sklearn.ensemble.RandomForestRegressor,
    'Gradient Boosting Regressor': sklearn.ensemble.GradientBoostingRegressor,
    'K-Neighbors Regressor': sklearn.neighbors.KNeighborsRegressor,
    'Regresión Logística': sklearn.linear_model.LogisticRegression,
    'Máquina de Soporte Vectorial (SVC)': sklearn.svm.SVC,
    'Árbol de Decisión': sklearn.tree.DecisionTreeClassifier,
    'Random Forest Classifier': sklearn.ensemble.RandomForestClassifier,
    'Gradient Boosting Classifier': sklearn.ensemble.GradientBoostingClassifier,
    'Naive Bayes': sklearn.naive_bayes.GaussianNB,
    'K-Neighbors Classifier': sklearn.neighbors.KNeighborsClassifier,
    'Perceptrón': sklearn.linear_model.Perceptron,
    'Red Neuronal (MLPClassifier)': sklearn.neural_network.MLPClassifier,
    #'K-Means': sklearn.cluster.KMeans,
    #'DBSCAN': sklearn.cluster.DBSCAN,
    #'Mean Shift': sklearn.cluster.MeanShift,
    #'Agglomerative Clustering': sklearn.cluster.AgglomerativeClustering,
    #'Birch': sklearn.cluster.Birch,
    #'Affinity Propagation': sklearn.cluster.AffinityPropagation,
    #'Spectral Clustering': sklearn.cluster.SpectralClustering,
    #'Gaussian Mixture Models (GMM)': sklearn.mixture.GaussianMixture()
}

parametros_algoritmos = {
    'Regresión Lineal': {},
    'Regresión Ridge': {'alpha': {"type": "number", "minimum": 0.0, "default": 1.0}},
    'Regresión Lasso': {'alpha': {"type": "number", "minimum": 0.0, "default": 1.0}},
    'ElasticNet': {'alpha': {"type": "number", "minimum": 0.0, "default": 1.0}, 'l1_ratio': {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5}},
    'Bayesian Ridge': {'alpha_1': {"type": "number", "minimum": 0.0, "default": 1e-6}, 'alpha_2': {"type": "number", "minimum": 0.0, "default": 1e-6}, 'lambda_1': {"type": "number", "minimum": 0.0, "default": 1e-6}, 'lambda_2': {"type": "number", "minimum": 0.0, "default": 1e-6}},
    'SVR (Máquina de Soporte Vectorial)': {'C': {"type": "number", "minimum": 0.0, "default": 1.0}, 'epsilon': {"type": "number", "minimum": 0.0, "default": 0.1}},
    'Árbol de Decisión para Regresión': {'max_depth': {"type": "integer", "minimum": 1, "default": None}},
    'Random Forest Regressor': {'n_estimators': {"type": "integer", "minimum": 1, "default": 100}, 'max_depth': {"type": "integer", "minimum": 1, "default": None}},
    'Gradient Boosting Regressor': {'n_estimators': {"type": "integer", "minimum": 1, "default": 100}, 'max_depth': {"type": "integer", "minimum": 1, "default": 3}},
    'K-Neighbors Regressor': {'n_neighbors': {"type": "integer", "minimum": 1, "default": 5}},
    'Regresión Logística': {'C': {"type": "number", "minimum": 0.0, "default": 1.0}},
    'Máquina de Soporte Vectorial (SVC)': {'C': {"type": "number", "minimum": 0.0, "default": 1.0}, 'gamma': {"type": "number", "minimum": 0.0, "default": 'scale'}},
    'Árbol de Decisión': {'max_depth': {"type": "integer", "minimum": 1, "default": None}},
    'Random Forest Classifier': {'n_estimators': {"type": "integer", "minimum": 1, "default": 100}, 'max_depth': {"type": "integer", "minimum": 1, "default": None}},
    'Gradient Boosting Classifier': {'n_estimators': {"type": "integer", "minimum": 1, "default": 100}, 'max_depth': {"type": "integer", "minimum": 1, "default": 3}},
    'Naive Bayes': {},
    'K-Neighbors Classifier': {'n_neighbors': {"type": "integer", "minimum": 1, "default": 5}},
    'Perceptrón': {'alpha': {"type": "number", "minimum": 0.0, "default": 0.0001}},
    'Red Neuronal (MLPClassifier)': {'hidden_layer_sizes': {"type": "array", "items": {"type": "integer", "minimum": 1}, "default": [100]}, 'alpha': {"type": "number", "minimum": 0.0, "default": 0.0001}},
}
def actualizar_modelos():
    with open('models/models.txt', 'w') as file:
        for m in modelos:
            file.write(f"{m.id};{m.nombre};{m.variable};{m.filename};"
                       f"{m.tipo};{','.join(m.parametros)};{','.join(m.algoritmos)};"
                       f"{m.entrenado};{m.fecha_creacion};{m.mejor_modelo};{m.score}\n")

normalizar = False

@app.get("/state")
def obtener_estado():
    return {"message": "API en funcionamiento"}

@app.get("/modelos", response_model=List[Modelo])
def obtener_modelos():
    return modelos

@app.post("/modelos")
async def crear_modelo(
    nombre: str = Form(...), 
    dataset: UploadFile = File(...),
    variable: str = Form(...),
    tipo: str = Form(...),
    parametros: str = Form(...),
    variables_numericas: str = Form(...),
    variables_categoricas: str = Form(...),
    algoritmos: str = Form(...),
):  
    try:
        # Save parameters type:

        # Remove objective variable from parameters
        if variables_numericas == "No variables":
            variables_numericas = []
        else:
            variables_numericas = variables_numericas.split(',')
        if variables_categoricas == "No variables":
            variables_categoricas = []
        else:
            variables_categoricas = variables_categoricas.split(',')

        if variable in variables_numericas:
            variables_numericas.remove(variable)

        if variable in variables_categoricas:
            variables_categoricas.remove(variable)
        
        variables_numericas = [col for col in variables_numericas if col in parametros.split(',')]
        variables_categoricas = [col for col in variables_categoricas if col in parametros.split(',')]

        id = len(modelos) + 1
        # Extract file extension
        file_extension = os.path.splitext(dataset.filename)[1]
        # Create folder under /models: /models/model{id}/
        folder_path = f'models/model{id}'
        os.makedirs(folder_path, exist_ok=True)
        # Construct the filename with the appropriate extension
        filename = f'{folder_path}/dataset{file_extension}'
        # Save file to disk
        contents = await dataset.read()
        if dataset.content_type == 'text/csv' or dataset.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        elif dataset.content_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] or dataset.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Transform categorical columns to str
        df[variables_categoricas] = df[variables_categoricas].astype(str)

        # Save the dataset to disk
        if file_extension == '.csv':
            df.to_csv(filename, index=False)
        elif file_extension in ['.xls', '.xlsx']:
            df.to_excel(filename, index=False)

        joblib.dump(variables_numericas, f'{folder_path}/numerical_columns.joblib')
        joblib.dump(variables_categoricas, f'{folder_path}/categorical_columns.joblib')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Create a new model with metadata about the file
    nuevo_modelo = Modelo(
        id=len(modelos) + 1, 
        nombre=nombre, 
        variable=variable, 
        filename=dataset.filename, 
        tipo=tipo,
        parametros=parametros.split(','),
        algoritmos=algoritmos.split(','),
        entrenado=False,
        fecha_creacion=time.strftime('%d-%m-%Y')
    )
    modelos.append(nuevo_modelo)

    # Write on models/models.txt
    with open('models/models.txt', 'a') as file:
        file.write(f'{nuevo_modelo.id};{nuevo_modelo.nombre};{nuevo_modelo.variable};{nuevo_modelo.filename};{nuevo_modelo.tipo};{",".join(nuevo_modelo.parametros)};{",".join(nuevo_modelo.algoritmos)};{nuevo_modelo.entrenado};{nuevo_modelo.fecha_creacion};;\n')

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
    try:
        contents = await dataset.read()
        if dataset.content_type == 'text/csv' or dataset.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        elif dataset.content_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] or dataset.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Determine whether columns are categorical or numerical
        variables = []
        for col in df.columns:
            
            # If the values are all NaN, skip the column
            if df[col].isnull().all():
                continue

            if df[col].dtype == 'object':
                tipo = 'categórica'
            elif df[col].nunique() <= 10:
                tipo = 'unknown'
            else:
                tipo = 'numérica'
            variables.append([col, tipo])
        
        return variables
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retornar todos los algoritmos disponibles
@app.get("/algoritmos", response_model=Dict[str, List[str]])
def obtener_algoritmos():
    return opcionesAlgoritmos

# Entrenar un modelo
@app.post("/modelos/{id}/entrenar")
async def entrenar_modelo(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    if modelo.entrenado:
        raise HTTPException(status_code=400, detail="El modelo ya ha sido entrenado")

    # Load the dataset
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

    # Drop rows with missing values in the target variable
    dataset = dataset.dropna(subset=[modelo.variable])

    X = dataset.drop(columns=modelo.variable)[modelo.parametros]
    y = dataset[modelo.variable]

    numerical_columns = joblib.load(f'models/model{id}/numerical_columns.joblib')
    categorical_columns = joblib.load(f'models/model{id}/categorical_columns.joblib')

    numerical_columns = [col for col in numerical_columns if col in X.columns]
    categorical_columns = [col for col in categorical_columns if col in X.columns]

    if set(modelo.parametros) != set(numerical_columns + categorical_columns):
        raise HTTPException(status_code=407, detail="Los parámetros del modelo no coinciden con las columnas del dataset")

    # Guardar los posibles valores de las columnas categóricas
    for col in categorical_columns:
        # If there is a number in the unique values, convert it to string
        if any(isinstance(x, (int, float)) for x in X[col].unique()):
            X[col] = X[col].astype(str)
    joblib.dump({col: sorted(X[col].unique().tolist()) for col in categorical_columns}, f'models/model{id}/categorical_values.joblib')

    # Imputar valores faltantes
    X[categorical_columns] = X[categorical_columns].applymap(lambda x: unidecode(str(x).lower()) if isinstance(x, str) else x)
    X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].median())
    X[categorical_columns] = X[categorical_columns].fillna("Desconocido")

    # Sacar valores p de las columnas, y quitar las que son < 0.05
    columnas_eliminadas = []
    if modelo.tipo == "Regresión":
        while True:
            p_values = []
            for col in X.columns:
                if col in numerical_columns:
                    _, p = stats.pearsonr(X[col], y)
                else:
                    _, p = stats.f_oneway(*[X.loc[X[col] == val, y.name] for val in X[col].unique()])
                p_values.append(p)
            if max(p_values) <= 0.05:
                break
            columna_a_eliminar = X.columns[p_values.index(max(p_values))]
            columnas_eliminadas.append(['"'+columna_a_eliminar+'"', round(max(p_values), 3)])
            X = X.drop(columns=columna_a_eliminar)
    
    joblib.dump(columnas_eliminadas, f'models/model{id}/columnas_eliminadas.joblib')


    # Actualizar las columnas numéricas y categóricas
    numerical_columns = [col for col in numerical_columns if col in X.columns]
    categorical_columns = [col for col in categorical_columns if col in X.columns]

    joblib.dump(numerical_columns, f'models/model{id}/numerical_columns.joblib')
    joblib.dump(categorical_columns, f'models/model{id}/categorical_columns.joblib')

    encoder = preprocessing.OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore', max_categories=10)
    encoded_data = encoder.fit_transform(X[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    # Guardar el encoder y las columnas codificadas
    joblib.dump(encoder, f'models/model{id}/encoder.joblib')
    joblib.dump(encoder.get_feature_names_out(categorical_columns), f'models/model{id}/encoder_feature_names.joblib')

    # Concatenar datos numéricos y codificados
    X = pd.concat([X[numerical_columns].reset_index(drop=True), encoded_df], axis=1)

    if normalizar:
        scaler = preprocessing.StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

        # Guardar el escalador y las columnas finales
        joblib.dump(scaler, f'models/model{id}/scaler.joblib')
    joblib.dump(list(X.columns), f'models/model{id}/feature_columns.joblib')

    if modelo.tipo == 'Clasificación':
    # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    else:
        X_train = X
        y_train = y

    best_score = 0
    best_model = None
    mejor_modelo = None
    scores = {}
    all_metrics = {}

    for algoritmo in modelo.algoritmos:
        model = funcionesAlgoritmos[algoritmo]()
        model.fit(X_train, y_train)

        if modelo.tipo == 'Regresión':
            y_pred = model.predict(X_train)
            r2 = r2_score(y_train, y_pred)
            mae = mean_absolute_error(y_train, y_pred)
            mse = mean_squared_error(y_train, y_pred)
            rmse = np.sqrt(mse)

            metrics = {
                'R²': r2,
                'Error Absoluto Medio (MAE)': mae,
                'Error Cuadrático Medio (MSE)': mse,
                'Raíz del Error Cuadrático Medio (RMSE)': rmse
            }
            score = r2

        elif modelo.tipo == 'Clasificación':
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            score = accuracy

        if score > best_score:
            best_score = score
            best_model = model
            mejor_modelo = algoritmo

        scores[algoritmo] = score
        all_metrics[algoritmo] = metrics

    if best_model is None:
        raise HTTPException(status_code=500, detail="No se pudo entrenar ningún modelo")

    # Save the trained model and metrics
    joblib.dump(best_model, f'models/model{id}/trained_model.joblib')
    joblib.dump(scores, f'models/model{id}/scores.joblib')
    joblib.dump(all_metrics, f'models/model{id}/metrics.joblib')

    modelo.entrenado = True
    modelo.mejor_modelo = mejor_modelo
    modelo.score = best_score
    actualizar_modelos()

@app.get("/modelos/{id}/columnasEliminadas")
def obtener_columnas_eliminadas(id: int):
    try:
        columnas_eliminadas = joblib.load(f'models/model{id}/columnas_eliminadas.joblib')
        return columnas_eliminadas
    except:
        return []

@app.get("/modelos/{id}/entrenamiento")
def obtener_entrenamiento(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    if modelo.entrenado:
        return {"message": "Modelo entrenado exitosamente", "mejor_modelo": modelo.mejor_modelo, "score": modelo.score}
    else:
        return {"message": "Modelo no entrenado"}

@app.post("/modelos/{id}/predecir")
async def predecir(id: int, datos: dict):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None or not modelo.entrenado:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no entrenado")

    # Load the trained model, scaler, encoder, and feature columns
    model = joblib.load(f'models/model{id}/trained_model.joblib')
    if normalizar:
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
    if normalizar:
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

    # Realizar la predicción
    prediction = model.predict(input_data)

    if len(prediction) == 1:
        prediction = round(prediction[0], 4)

    return {"prediccion": prediction.tolist()}

@app.get("/modelos/{id}/parametros")
def obtener_parametros(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None or not modelo.entrenado:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no entrenado")

    numerical_columns = joblib.load(f'models/model{id}/numerical_columns.joblib')
    categorical_columns = joblib.load(f'models/model{id}/categorical_columns.joblib')

    return numerical_columns+categorical_columns
    
@app.get("/modelos/{id}/parametrosLabels")
async def obtener_parametros_valores(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None or not modelo.entrenado:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no entrenado")
    
    parametrosModelo = obtener_parametros(id)

    categorical_values = joblib.load(f'models/model{id}/categorical_values.joblib')

    valores = {}
    for parametro in parametrosModelo:
        if parametro in categorical_values:
            valores[parametro] = categorical_values[parametro]

    return valores

@app.delete("/modelos/{id}")
async def eliminar_modelo(id: int):
    global modelos
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    modelos = [m for m in modelos if m.id != id]

    # Eliminar el directorio del modelo
    folder_path = f'models/model{id}'
    if os.path.exists(folder_path):
        import shutil
        shutil.rmtree(folder_path)

    # Actualizar y guardar el archivo de modelos
    actualizar_modelos()

    return {"message": "Modelo eliminado exitosamente"}

@app.get("/modelos/{id}/scores")
def obtener_scores(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None or not modelo.entrenado:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no entrenado")

    scores = joblib.load(f'models/model{id}/scores.joblib')

    return scores

@app.get("/modelos/{id}/metrics")
def obtener_metricas(id: int):
    modelo = next((m for m in modelos if m.id == id), None)
    if modelo is None or not modelo.entrenado:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no entrenado")

    # Load the metrics from the joblib file
    metrics = joblib.load(f'models/model{id}/metrics.joblib')
    best_model = joblib.load(f'models/model{id}/trained_model.joblib')

    respuesta = {"metrics": metrics[modelo.mejor_modelo]}

    if modelo.tipo == 'Regresión':
        if modelo.mejor_modelo in ['Regresión Lineal', 'Regresión Ridge', 'Regresión Lasso', 'ElasticNet', 'Bayesian Ridge', 'SVR (Máquina de Soporte Vectorial)']:
            respuesta['coeficientes'] = dict(sorted(zip(best_model.feature_names_in_, best_model.coef_), key=lambda item: item[1], reverse=True))
            respuesta['intercepto'] = best_model.intercept_
        elif modelo.mejor_modelo in ['Árbol de Decisión para Regresión', 'Random Forest Regressor', 'Gradient Boosting Regressor']:
            respuesta['caracteristicas'] = list(best_model.feature_importances_)
    elif modelo.tipo == 'Clasificación':
        if modelo.mejor_modelo in ['Regresión Logística', 'Máquina de Soporte Vectorial (SVC)', 'Perceptrón', 'Red Neuronal (MLPClassifier)']:
            respuesta['coeficientes'] = dict(sorted(zip(best_model.feature_names_in_, best_model.coef_), key=lambda item: item[1], reverse=True))
            respuesta['intercepto'] = best_model.intercept_
        elif modelo.mejor_modelo in ['Árbol de Decisión', 'Random Forest Classifier', 'Gradient Boosting Classifier']:
            respuesta['caracteristicas'] = list(best_model.feature_importances_)

    return respuesta