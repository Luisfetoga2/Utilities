import React, { useEffect, useState } from "react";
import { useNavigate  } from "react-router-dom";
import { Modal, Container, Form, Row, Col, Button, Spinner } from 'react-bootstrap';
import './modelosML.css';

function ModelosML() {
    /*

    Tener un listado de modelos:
        - Si no hay, mostrar mensaje grande de crear. (Si sí hay, mostrar listado de modelos y botón de crear)

    Creando un modelo:
        - Mostrar formulario con campos:
            - Nombre del modelo (tener un por defecto (Modelo 1, Modelo 2, etc))
            - Dataset (subir archivo)
            - Variable a predecir (seleccionar de un listado, que se obtiene del dataset)
            - Mensaje indicando que se hara regresion o clasificacion (dependiendo de la variable a predecir)
            - Checklist de algoritmos (seleccionar uno o más) (dependiendo de la variable a predecir) (se crearan modelos para cada algoritmo seleccionado)
            - Botón de crear modelo
        - Al dar clic en crear modelo, se abre el detalle de modelo
    
    Detalle de modelo:
        - Mostrar información del modelo (nombre, dataset, variable a predecir, algoritmos, etc)
        - Mostrar botón de entrenar modelo
        - Mostrar botón de hacer predicción
        - Mostrar botón de eliminar modelo
        - Mostrar gráficas de entrenamiento (si ya se entreno)
        - Mostrar gráficas de predicción (si ya se predijo)

    */
    const navigate = useNavigate();
    const [modelos, setModelos] = useState([]);
    const [showNewModelForm, setShowNewModelForm] = useState(false);
    const [nombre, setNombre] = useState("");
    const [dataset, setDataset] = useState(null);
    const [variable, setVariable] = useState("");
    const [posiblesParametros, setPosiblesParametros] = useState([]);
    const [parametros, setParametros] = useState([]);
    const [tipoModelo, setTipoModelo] = useState("");

    const [fetchedPossibleVariables, setFetchedPossibleVariables] = useState(0);
    const [possibleVariables, setPossibleVariables] = useState([]);
    const [possibleVariableTypes, setPossibleVariableTypes] = useState([]);
    const [loadingVariables, setLoadingVariables] = useState(false);

    const [algorithmOptions, setAlgorithmOptions] = useState({});
    const [selectedAlgorithms, setSelectedAlgorithms] = useState([]);

    const [apiStatus, setApiStatus] = useState("checking");

    const handleClose = () => {
        setShowNewModelForm(false);
        // Reset form fields
        setNombre("");
        setDataset(null);
        setVariable("");
        setTipoModelo("");
        setFetchedPossibleVariables(0);
        setPossibleVariables([]);
        setPossibleVariableTypes([]);
        setPosiblesParametros([]);
        setParametros([]);
        setLoadingVariables(false);
        setSelectedAlgorithms([]);
    }
    const handleShow = () => {
        setShowNewModelForm(true);
        setNombre("Modelo "+(modelos.length+1));
    }

    const handleAlgorithmChange = (algorithm) => {
        setSelectedAlgorithms((prev) =>
            prev.includes(algorithm)
                ? prev.filter((a) => a !== algorithm)
                : [...prev, algorithm]
        );
    };

    const handleParametroChange = (parametro) => {
        setParametros((prev) =>
            prev.includes(parametro)
                ? prev.filter((a) => a !== parametro)
                : [...prev, parametro]
        );
    };

    const onFileChange = async (event) => {

        // Reset variables
        setVariable("");
        setTipoModelo("");
        setFetchedPossibleVariables(0);
        setPossibleVariables([]);
        setPossibleVariableTypes([]);
        setPosiblesParametros([]);
        setParametros([]);
        setSelectedAlgorithms([]);
        setLoadingVariables(false);

        const file = event.target.files[0];
        setDataset(file);

        if (file) {
            setLoadingVariables(true); // Mostrar indicador de carga
            try {
                const formData = new FormData();
                formData.append("dataset", file);

                console.log(formData);

                const response = await fetch("http://127.0.0.1:8000/variables", {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error("Error al obtener las variables");
                }

                const data = await response.json();

                var possibleVariables = [];
                var possibleVariableTypes = [];
                
                for (var i = 0; i < data.length; i++) {
                    var variable = data[i][0];
                    var tipo = data[i][1];
                    
                    possibleVariables.push(variable);
                    possibleVariableTypes.push(tipo);
                }

                setPossibleVariables(possibleVariables);
                setPossibleVariableTypes(possibleVariableTypes);

                if (possibleVariableTypes.includes("unknown")) {
                    setFetchedPossibleVariables(1);
                } else {
                    setFetchedPossibleVariables(2);
                }
            } catch (error) {
                console.error("Error al obtener las variables:", error);
                setPossibleVariables([]);
                setFetchedPossibleVariables(3);
            } finally {
                setLoadingVariables(false); // Ocultar indicador de carga
            }
        }
    };

    useEffect(() => {
        const checkApiStatus = async () => {
            try {
                const response = await fetch("http://127.0.0.1:8000/state");
                const data = await response.json();
                if (data.message === "API en funcionamiento") {
                    setApiStatus("running");
                    fetch("http://127.0.0.1:8000/modelos") // FastAPI URL
                        .then(response => response.json())
                        .then(data => setModelos(data))
                        .catch(error => console.error("Error al obtener los modelos:", error));

                    fetch("http://127.0.0.1:8000/algoritmos") // FastAPI URL
                        .then(response => response.json())
                        .then(data => setAlgorithmOptions(data))
                        .catch(error => console.error("Error al obtener los algoritmos:", error));
                } else {
                    setApiStatus("initializing");
                }
            } catch (error) {
                console.error("Error al verificar el estado de la API:", error);
                setApiStatus("initializing");
            }
        };

        checkApiStatus();

        const interval = setInterval(() => {
            if (apiStatus === "initializing") {
                checkApiStatus();
            }
        }, 5000);

        return () => clearInterval(interval);
    }, [apiStatus]);

    const handleSubmit = async (e) => {
        try{
            const formData = new FormData();
            formData.append("nombre", nombre);
            formData.append("dataset", dataset);
            formData.append("variable", variable);
            formData.append("tipo", tipoModelo);
            formData.append("parametros", parametros.join(","));
            let numericVariables = possibleVariables.filter((variable, index) => possibleVariableTypes[index] === "numérica").join(",");
            let categoricalVariables = possibleVariables.filter((variable, index) => possibleVariableTypes[index] === "categórica").join(",");

            formData.append("variables_numericas", numericVariables || "No variables");
            formData.append("variables_categoricas", categoricalVariables || "No variables");
            formData.append("algoritmos", selectedAlgorithms.join(","));

            console.log(formData);

            const response = await fetch("http://127.0.0.1:8000/modelos", {
                method: "POST",
                body: formData,
            })
            
            if (!response.ok) {
                throw new Error("Error al crear el modelo");
            }

            const data = await response.json();
            // Redirigir al detalle del modelo
            navigate(`/modelos/${data.id}`);
        } catch (error) {
            console.error("Error al crear el modelo:", error);
        }
    };

    const handleVariableTypeChange = (index, newType) => {
        const updatedTypes = [...possibleVariableTypes];
        updatedTypes[index] = newType; // Update the type for the specific variable
        setPossibleVariableTypes(updatedTypes);
    
        // Check if all unknown types are resolved
        if (!updatedTypes.includes("unknown")) {
            setFetchedPossibleVariables(2);
        } else {
            setFetchedPossibleVariables(1);
        }
    };

    const handleVariableChange = (index) => {
        const selectedVariable = possibleVariables[index];
        setVariable(selectedVariable);
        setTipoModelo(possibleVariableTypes[index] === "numérica" ? "Regresión" : "Clasificación");
        // posiblesParametros son todas las variables menos la seleccionada
        setParametros([]);
        setPosiblesParametros(possibleVariables.filter((variable) => variable !== selectedVariable));
    };
    
    const handleRowClick = (id) => {
        navigate(`/modelos/${id}`); // Redirige al detalle del modelo
    };

    if (apiStatus === "initializing" || apiStatus === "checking") {
        return (
            <Container fluid className="d-flex justify-content-center align-items-center" style={{ minHeight: 'calc(100vh - 56px)' }}>
                <div className="text-center">
                    <Spinner animation="border" role="status">
                        <span className="visually-hidden">Loading...</span>
                    </Spinner>
                    <p>Se está inicializando el servidor</p>
                </div>
            </Container>
        );
    }

    return (
        <Container fluid className="d-flex justify-content-center align-items-center" style={{ minHeight: 'calc(100vh - 56px)' }}>
            <div>
                <h1 className="text-center">Predicciones usando modelos de ML</h1>
                <br />
                {modelos.length === 0 && (
                        <p> No has creado ningun modelo</p>
                )}
                {modelos.length > 0 && (
                    <table className="modelos-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Nombre</th>
                                <th>Variable</th>
                                <th>Parametros</th>
                                <th>Archivo</th>
                                <th>Tipo</th>
                                <th>Algoritmos</th>
                                <th>Entrenado</th>
                                <th>Fecha creación</th>
                            </tr>
                        </thead>
                        <tbody>
                            {modelos.map((modelo) => (
                                <tr
                                    key={modelo.id}
                                    className="modelo-row"
                                    onClick={() => handleRowClick(modelo.id)}
                                >
                                    <td>{modelo.id}</td>
                                    <td>{modelo.nombre}</td>
                                    <td>{modelo.variable}</td>
                                    <td>{modelo.parametros.join(', ')}</td>
                                    <td>{modelo.filename}</td>
                                    <td>{modelo.tipo}</td>
                                    <td>{modelo.algoritmos.join(', ')}</td>
                                    <td>{modelo.entrenado ? 'Sí' : 'No'}</td>
                                    <td>{modelo.fecha_creacion}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
                <br />
                <Button variant="primary" onClick={handleShow}>Crear modelo</Button>
            </div>

            <Modal
                onHide={handleClose}
                show={showNewModelForm}
                size="lg"
                aria-labelledby="contained-modal-title-vcenter"
                centered
            >
                <Modal.Header closeButton>
                    <Modal.Title id="contained-modal-title-vcenter">
                        Crear modelo
                    </Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <Form>
                        <Form.Group as={Row} className="mb-3">
                            <Form.Label column sm="3">Nombre del Modelo</Form.Label>
                            <Col sm="9">
                                <Form.Control
                                    type="text"
                                    value={nombre}
                                    onChange={e => setNombre(e.target.value)}
                                />
                            </Col>
                        </Form.Group>

                        <Form.Group controlId="formDataset" as={Row} className="mb-3">
                            <Form.Label column sm="3">Dataset</Form.Label>
                            <Col sm="9">
                                <Form.Control type="file" onChange={onFileChange} />
                            </Col>
                        </Form.Group>

                        {loadingVariables && <p>Cargando variables...</p>}

                        {fetchedPossibleVariables===3 && <p>Error al cargar las variables. Revisa el archivo</p>}

                        {fetchedPossibleVariables>=1 && (
                            <Form.Group as={Row} className="mb-3">
                                <Form.Label column sm="3">Tipos de las variables</Form.Label>
                                <Col sm="9">
                                    {/* Map all the possible variables and their types.
                                    possibleVariables has the possible variables, and possibleVariableType has the type of the variables.
                                    If the type is "numérica" or "categorica" show the variable name and a non interactive select that has the current type.
                                    If the type is "unkown", show the variable name and a interactive select that has "numérica" or "categorica"*/}
                                    {possibleVariables.map((variable, index) => (
                                        <Form.Group as={Row} key={index} className="mb-3">
                                            <Form.Label column sm="7">{variable}</Form.Label>
                                            <Col sm="5">
                                                {possibleVariableTypes[index] === "numérica" || possibleVariableTypes[index] === "categórica" ? (
                                                    <Form.Control
                                                        type="text"
                                                        value={possibleVariableTypes[index]}
                                                        readOnly
                                                        disabled
                                                    />
                                                ) : (
                                                    <Form.Select
                                                        value={possibleVariableTypes[index]}
                                                        onChange={(e) => handleVariableTypeChange(index, e.target.value)}
                                                    >
                                                        <option value="unknown">Seleccione un tipo</option>
                                                        <option value="numérica">Numérica</option>
                                                        <option value="categórica">Categórica</option>
                                                    </Form.Select>
                                                )}

                                            </Col>
                                        </Form.Group>
                                    ))}
                                </Col>
                            </Form.Group>
                        )}

                        {fetchedPossibleVariables===2 && possibleVariables.length > 0 && (
                            <Form.Group as={Row} className="mb-3">
                                <Form.Label column sm="3">Variable a Predecir</Form.Label>
                                <Col sm="9">
                                    <Form.Select
                                        value={possibleVariables.indexOf(variable)} // Map variable back to index
                                        onChange={e => handleVariableChange(e.target.value)}
                                    >
                                        <option value="">Seleccione una variable</option>
                                        {possibleVariables.map((variable, index) => (
                                            <option key={index} value={index}>
                                                {variable} ({possibleVariableTypes[index]})
                                            </option>
                                        ))}
                                    </Form.Select>

                                </Col>
                            </Form.Group>
                        )}

                        {variable && (
                            <Form.Group as={Row} className="mb-3">
                                <Form.Label column sm="3">Tipo de Modelo</Form.Label>
                                <Col sm="9">
                                    {tipoModelo==="Regresión" ? (
                                    <Form.Control
                                        type="text"
                                        value={tipoModelo}
                                        readOnly
                                        disabled
                                    />) : (
                                        <Form.Control
                                            type="text"
                                            value={tipoModelo}
                                            readOnly
                                            disabled
                                        />
                                        )} {/* <Form.Select value = {tipoModelo} onChange={e => setTipoModelo(e.target.value)} >
                                        <option value="">Seleccione un tipo de modelo</option>
                                        <option value="Agrupación">Agrupación</option>
                                        <option value="Clasificación">Clasificación</option>
                                    </Form.Select> */}
                                </Col>
                            </Form.Group>
                        )}

                        {tipoModelo && (
                            <Form.Group as={Row} className="mb-3">
                                <Col sm="3">
                                    <Row>
                                    <Form.Label column sm="3">Parametros</Form.Label>
                                    </Row>
                                    <Button variant = "primary" size="sm" onClick={() => setParametros(posiblesParametros)}>Seleccionar todos</Button>
                                    <div className="mb-3"></div>
                                    <Button
                                        variant="secondary"
                                        size="sm"
                                        onClick={() => setParametros([])}
                                        {...(parametros.length === 0 && { disabled: true })}
                                    > Limpiar selección </Button>
                                </Col>
                                <Col sm="9">
                                    {posiblesParametros?.map((parametro, index) => (
                                        <Form.Check
                                            key={index}
                                            type="checkbox"
                                            label={parametro}
                                            checked={parametros.includes(parametro)}
                                            onChange={() => handleParametroChange(parametro)}
                                        />
                                    ))}
                                </Col>
                            </Form.Group>
                        )}

                        {parametros.length > 0 && (
                            <Form.Group as={Row} className="mb-3">
                                <Col sm="3">
                                    <Row>
                                    <Form.Label column sm="3">Algoritmos</Form.Label>
                                    </Row>
                                    <Button variant = "primary" size="sm" onClick={() => setSelectedAlgorithms(algorithmOptions[tipoModelo])}>Seleccionar todos</Button>
                                    <div className="mb-3"></div>
                                    <Button
                                        variant="secondary"
                                        size="sm"
                                        onClick={() => setSelectedAlgorithms([])}
                                        {...(selectedAlgorithms.length === 0 && { disabled: true })}
                                    > Limpiar selección </Button>
                                </Col>
                                <Col sm="9">
                                    {algorithmOptions[tipoModelo]?.map((algorithm, index) => (
                                        <Form.Check
                                            key={index}
                                            type="checkbox"
                                            label={algorithm}
                                            checked={selectedAlgorithms.includes(algorithm)}
                                            onChange={() => handleAlgorithmChange(algorithm)}
                                        />
                                    ))}
                                </Col>
                            </Form.Group>
                        )}
                    </Form>
                </Modal.Body>

                <Modal.Footer>
                    <Button variant="secondary" onClick={handleClose}>
                        Cerrar
                    </Button>
                    <Button variant="primary" type="submit" onClick={handleSubmit}
                    {...((!nombre || !dataset || !variable || !tipoModelo || parametros.length===0 || selectedAlgorithms.length === 0) && { disabled: true })}
                    >
                        Crear Modelo
                    </Button>
                </Modal.Footer>
            </Modal>
        </Container>
    );
}

export default ModelosML;
