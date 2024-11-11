import React, { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { Container, Button, Form, Table, Col, Row, Card, ProgressBar, Spinner, Badge } from 'react-bootstrap';
import { FaStar, FaChartLine, FaPlay, FaPlus, FaTrash } from 'react-icons/fa';

function DetalleModelo() {
  const { id } = useParams();
  const [modelFound, setModelFound] = useState(0);
  const [modelo, setModelo] = useState(null);
  const [estadoEntrenamiento, setEstadoEntrenamiento] = useState(0); // 0: no entrenado, 1: entrenando, 2: entrenado
  const [parametros, setParametros] = useState([]);
  const [inputLabels, setInputLabels] = useState({}); // Store input labels for prediction
  const [scores, setScores] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [inputTables, setInputTables] = useState([{ id: Date.now(), values: {}}]);
  const [results, setResults] = useState({});

  useEffect(() => {
    async function fetchModelo() {
      try {
        const response = await fetch(`http://127.0.0.1:8000/modelos/${id}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        //console.log("Modelo encontrado");
        //console.log(data)
        setModelo(data);
        setModelFound(1);
        if (data.entrenado) {
          //console.log("Modelo entrenado");
          setEstadoEntrenamiento(2);
          const response = await fetch(`http://127.0.0.1:8000/modelos/${id}/parametros`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const paramsData = await response.json();
          setParametros(paramsData);
          initializeInputValues(paramsData);

          const labelsResponse = await fetch(`http://127.0.0.1:8000/modelos/${id}/parametrosLabels`);
          if (!labelsResponse.ok) {
            throw new Error(`HTTP error! status: ${labelsResponse.status}`);
          }
          const labelsData = await labelsResponse.json();
          setInputLabels(labelsData);
        }
      } catch (error) {
        //console.error("Network error:", error);
        setModelFound(-1);
      }
    }

    fetchModelo();
  }, [id]);

  const fetchScores = useCallback(async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/modelos/${id}/scores`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();

      // Sort scores by dictionary value
      const dict = data;
      const sorted = Object.keys(dict).sort((a, b) => dict[b] - dict[a]).reduce((obj, key) => {
        obj[key] = dict[key];
        return obj;
      }, {});

      //console.log("Scores:", sorted);
      setScores(sorted); // Store the scores in state
    } catch (error) {
      //console.error("Error fetching scores:", error);
    }
  }, [id]);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/modelos/${id}/metrics`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setMetrics(data); // Store metrics in state
    } catch (error) {
      //console.error("Error fetching metrics:", error);
    }
  }, [id]);
  
  useEffect(() => {
    if (estadoEntrenamiento === 2) {
      //console.log("Estado de entrenamiento:", estadoEntrenamiento);
      fetchScores();
      fetchMetrics();
    }
  }, [estadoEntrenamiento, fetchMetrics, fetchScores]);

  useEffect(() => {
    if (parametros.length > 0) {
      initializeInputValues(parametros);
    }
  }, [parametros]);
  
  

  const initializeInputValues = (params) => {
    const initialValues = {};
    params.forEach((param) => {
      initialValues[param] = "";
    });
    setInputTables((prevTables) =>
      prevTables.map((table) => ({
        ...table,
        values: { ...initialValues },
      }))
    );
  };

  const handlePredict = async (tableId) => {
    const table = inputTables.find((t) => t.id === tableId);
    const predictionData = table.values;
  
    try {
      const response = await fetch(`http://127.0.0.1:8000/modelos/${id}/predecir`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(predictionData),
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json();
      return data.prediccion; // Return prediction result
    } catch (error) {
      return "Error en la predicción";
    }
  };
  

  const handleAddTable = () => {
    setInputTables((prevTables) => [
      ...prevTables,
      { id: Date.now(), values: {} },
    ]);
  };

  const handleRemoveTable = (tableId) => {
    setInputTables((prevTables) =>
      prevTables.filter((table) => table.id !== tableId)
    );
  };

  const isTableComplete = useCallback(
    (table) => parametros.every((param) => table.values[param] && table.values[param] !== ""),
    [parametros]
  );

  const handleInputChange = (tableId, name, value) => {
    setInputTables((prevTables) =>
      prevTables.map((table) =>
        table.id === tableId
          ? { ...table, values: { ...table.values, [name]: value } }
          : table
      )
    );
  };
  
  const handleSelectChange = (tableId, name, value) => {
    setInputTables((prevTables) =>
      prevTables.map((table) =>
        table.id === tableId
          ? { ...table, values: { ...table.values, [name]: value } }
          : table
      )
    );
  };

  // Use useEffect to predict when the input tables change
  useEffect(() => {
    const updatePredictions = async () => {
      const updatedResults = {};
      for (const table of inputTables) {
        if (isTableComplete(table) && parametros.length > 0) {
          const tablePrediction = await handlePredict(table.id); // Get predictions
          updatedResults[table.id] = tablePrediction;
        } else {
          updatedResults[table.id] = "-";
        }
      }
      setResults(updatedResults); // Update all predictions at once
    };
  
    updatePredictions(); // Trigger the prediction function
  }, [inputTables, parametros]);

  const handleEliminar = () => {
    //console.log("Eliminar modelo", id);
    fetch(`http://127.0.0.1:8000/modelos/${id}`, {
      method: "DELETE",
    })
      .then((response) => response.json())
      .then((data) => {
        //console.log("Modelo eliminado:", data);
        // Redirect to /modelos
        window.location.href = "/modelos";
      }
    );
  };

  const handleEntrenar = () => {
    setEstadoEntrenamiento(1);

    fetch(`http://127.0.0.1:8000/modelos/${id}/entrenar`, {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        // Reload the page to show the new state
        window.location.reload();
      }
      );
  };

  const maxColumnsPerRow = 4;

  return (
    <Container fluid className="d-flex justify-content-center align-items-center" style={{ minHeight: 'calc(100vh - 56px)' }}>
      <div className="w-80">
        {modelFound === 0 && <p>Cargando...</p>}
        {modelFound === -1 && <p>Modelo no encontrado</p>}
        {modelFound === 1 && (
          <>
            <h1>{modelo.nombre}</h1>
            <br />
            <Table className="modelos-detail" striped bordered>
              <thead>
                <tr>
                  <th>ID</th>
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
                <tr>
                  <td>{modelo.id}</td>
                  <td>{modelo.variable}</td>
                  <td>{modelo.parametros.join(", ")}</td>
                  <td>{modelo.filename}</td>
                  <td>{modelo.tipo}</td>
                  <td>{modelo.algoritmos.join(", ")}</td>
                  <td>{modelo.entrenado ? "Sí" : "No"}</td>
                  <td>{modelo.fecha_creacion}</td>
                </tr>
              </tbody>
            </Table>
            <br />
            <Button variant="danger" onClick={handleEliminar}>Eliminar modelo</Button>

            <Container style={{ paddingTop: '20px' }}>
              <h2>Entrenamiento</h2>
              {estadoEntrenamiento === 0 && (
                <Button variant="success" onClick={handleEntrenar} className="mt-3">
                  <FaPlay style={{ marginRight: '8px' }} />
                  Iniciar Entrenamiento
                </Button>
              )}
              {estadoEntrenamiento === 1 && (
                <>
                  <Spinner animation="border" variant="primary" />
                  <h4 className="mt-3">Entrenando modelo...</h4>
                  <ProgressBar animated now={75} label="75%" className="mt-3" />
                </>
              )}
              {estadoEntrenamiento === 2 && (
              <Card className="mt-4 shadow-sm" border="primary" style={{ maxWidth: '400px', margin: 'auto' }}>
                <Card.Body className="text-center">
                <h5 className="mt-4">Scores por Modelo:</h5>
                <Table bordered style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f2f2f2', textAlign: 'left' }}>
                      <th style={{ padding: '10px', borderBottom: '2px solid #ddd' }}>Modelo</th>
                      <th style={{ padding: '10px', borderBottom: '2px solid #ddd' }}>R² Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scores && Object.entries(scores).map(([modelName, score]) => (
                      <tr key={modelName} style={{ borderBottom: '1px solid #ddd' }}>
                        <td style={{ padding: '10px' }}>{modelName}</td>
                        <td style={{ padding: '10px' }}>{score.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
                  <h3>
                    <FaStar style={{ color: '#FFD700', marginRight: '8px' }} />
                    Mejor Modelo: 
                    <Badge bg="info" className="ms-2">{modelo.mejor_modelo}</Badge>
                  </h3>
                  <h4 className="mt-3">
                    <FaChartLine style={{ marginRight: '8px' }} />
                    Score: <span style={{ fontWeight: 'bold', color: '#007bff' }}>{modelo.score.toFixed(3)}</span>
                  </h4>

                  {metrics && (
                    <>
                      <h5 className="mt-4">Métricas del Mejor Modelo:</h5>
                      <Table bordered>
                        <tbody>
                          {Object.entries(metrics).map(([key, value]) => (
                            <tr key={key}>
                              <td>{key}</td>
                              <td>{typeof value === 'number' ? value.toFixed(3) : JSON.stringify(value)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </>
                  )}
                </Card.Body>
              </Card>
            )}
            </Container>
            
            {estadoEntrenamiento === 2 && (
              <Container style={{ paddingTop: '20px' }}>
                <h2>Predecir</h2>
                {inputTables.map((table) => (
                  <div key={table.id} className="mb-4">
                    <Row>
                    <Col>
                    <Table bordered>
                      <tbody>
                        {Array.from({ length: Math.ceil(parametros.length / maxColumnsPerRow) }).map((_, rowIndex) => (
                          <React.Fragment key={rowIndex}>
                            <tr>
                              {parametros
                                .slice(rowIndex * maxColumnsPerRow, (rowIndex + 1) * maxColumnsPerRow)
                                .map((param) => (
                                  <th key={param}>{param}</th>
                                ))}
                              {Array.from({
                                length: maxColumnsPerRow -
                                  parametros.slice(rowIndex * maxColumnsPerRow, (rowIndex + 1) * maxColumnsPerRow).length,
                              }).map((_, emptyIndex) => (
                                <th key={`empty-header-${rowIndex}-${emptyIndex}`} />
                              ))}
                            </tr>
                            <tr>
                              {parametros
                                .slice(rowIndex * maxColumnsPerRow, (rowIndex + 1) * maxColumnsPerRow)
                                .map((param) => (
                                  <td key={param}>
                                    {inputLabels[param] ? (
                                      <Form.Select
                                        name={param}
                                        value={table.values[param] || ""}
                                        onChange={(e) => handleSelectChange(table.id, param, e.target.value)}
                                      >
                                        <option value="">Seleccionar...</option>
                                        {inputLabels[param].map((label) => (
                                          <option key={label} value={label}>
                                            {label}
                                          </option>
                                        ))}
                                      </Form.Select>
                                    ) : (
                                      <Form.Control
                                        type="number"
                                        name={param}
                                        value={table.values[param] || ""}
                                        onChange={(e) => handleInputChange(table.id, param, e.target.value)}
                                      />
                                    )}
                                  </td>
                                ))}
                              {Array.from({
                                length: maxColumnsPerRow -
                                  parametros.slice(rowIndex * maxColumnsPerRow, (rowIndex + 1) * maxColumnsPerRow).length,
                              }).map((_, emptyIndex) => (
                                <td key={`empty-input-${rowIndex}-${emptyIndex}`} />
                              ))}
                            </tr>
                          </React.Fragment>
                        ))}
                      </tbody>
                    </Table>
                    <div className="d-flex justify-content-between">
                      <Button
                        variant="danger"
                        onClick={() => handleRemoveTable(table.id)}
                        style={{ display: inputTables.length === 1 ? 'none' : 'inline-block' }} // Make invisible if it's the only table
                      >
                        <FaTrash /> Eliminar
                      </Button>
                    </div>
                    </Col>
                    <Col>
                      <h3>Resultado:</h3>
                      <div className="resultado-cuadro">{results[table.id]}</div>
                    </Col>
                    </Row>
                  </div>
                ))}
                <Button variant="success" onClick={handleAddTable} className="mt-4">
                  <FaPlus /> Agregar Tabla
                </Button>
              </Container>
            )}
          </>
        )}
        <div style={{ height: '200px' }}></div>
      </div>
    
    </Container>
  );
}

export default DetalleModelo;
