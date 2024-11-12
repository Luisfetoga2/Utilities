import React, { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "react-router-dom";
import { Container, Button, Form, Table, Row, Card, ProgressBar, Spinner, Badge, Col } from 'react-bootstrap';
import { FaStar, FaChartLine, FaPlay, FaPlus, FaTrash } from 'react-icons/fa';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ChartDataLabels);

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
  const [coeficientes, setCoeficientes] = useState(null);
  const [intercepto, setIntercepto] = useState(null);
  const [caracteristicas, setCaracteristicas] = useState(null);
  const inputRefs = useRef({});

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
      setMetrics(data["metrics"]); // Store metrics in state
      setCoeficientes(data["coeficientes"]); // Store coefficients in state
      setIntercepto(data["intercepto"]); // Store intercept in state
      setCaracteristicas(data["caracteristicas"]); // Store features in state
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

  const handlePredict = useCallback(async (tableId) => {
    const table = inputTables.find((t) => t.id === tableId);
    const predictionData = table.values;

    console.log("Predicción:", predictionData);
  
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
  }, [id, inputTables]);
  

  const handleAddTable = () => {
    setInputTables((prevTables) => [
      ...prevTables,
      { id: Date.now(), values: {} },
    ]);
  };

  const handleKeyDown = (e, tableId, param) => {
    if (e.key === "Tab") {
      e.preventDefault(); // Prevent default tab behavior
  
      // Get the current row and column index
      const tableIndex = inputTables.findIndex((table) => table.id === tableId);
      const paramIndex = parametros.indexOf(param);

      const isShiftPressed = e.shiftKey;

      var nextTableIndex;
      var nextParamIndex;
      
      if (!isShiftPressed) {
        // Calculate the next row
        nextParamIndex = paramIndex + 1;
    
        // If we're at the last row, move to the next column
        if (nextParamIndex >= parametros.length) {
          nextParamIndex = 0; // Wrap to the first row
          nextTableIndex = tableIndex + 1;
          if (nextTableIndex >= inputTables.length) {
            nextTableIndex = 0; // Wrap to the first column 
          }
          tableId = inputTables[nextTableIndex].id;
        }
      } else {
        // Calculate the previous row
        nextParamIndex = paramIndex - 1;

        // If we're at the first row, move to the previous column
        if (nextParamIndex < 0) {
          nextParamIndex = parametros.length - 1; // Wrap to the last row
          nextTableIndex = tableIndex - 1;
          if (nextTableIndex < 0) {
            nextTableIndex = inputTables.length - 1; // Wrap to the last column
          }
          tableId = inputTables[nextTableIndex].id;
        }
      }
  
      // Find the next input ref
      const nextRefKey = `${tableId}-${parametros[nextParamIndex]}`;
      const nextRef = inputRefs.current[nextRefKey];
  
      if (nextRef) {
        nextRef.focus();
      }
    }
  };
  

  const handleRemoveTable = (tableId) => {
    // Remove the table and associated refs
    Object.keys(inputRefs.current).forEach((key) => {
      if (key.startsWith(`${tableId}-`)) {
        delete inputRefs.current[key];
      }
    });
  
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
  }, [inputTables, parametros, handlePredict, isTableComplete]);

  const handleEliminar = () => {
    if (window.confirm("Estas seguro que desea eliminar este modelo?")) {
        fetch(`http://127.0.0.1:8000/modelos/${id}`, {
            method: "DELETE",
        })
        .then((response) => response.json())
        .then((data) => {
            //console.log("Modelo eliminado:", data);
            // Redirect to /modelos
            window.location.href = "/modelos";
        });
    }
};

  const handleEntrenar = () => {
    setEstadoEntrenamiento(1);

    fetch(`http://127.0.0.1:8000/modelos/${id}/entrenar`, {
      method: "POST",
    }).catch((error) => {});
  };

  // If estadoEntrenamiento is 1, fetch "/modelos/{id}/entrenamiento" every second, if "message" is "Modelo entrenado exitosamente", reload the page
  useEffect(() => {
    if (estadoEntrenamiento === 1) {
      const interval = setInterval(() => {
        fetch(`http://127.0.0.1:8000/modelos/${id}/entrenamiento`)
          .then((response) => response.json())
          .then((data) => {
            if (data.message === "Modelo entrenado exitosamente") {
              clearInterval(interval);
              window.location.reload();
            }
          });
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [estadoEntrenamiento, id]);

  const BarGraph = ({ coeficientes }) => {
    const values = Object.values(coeficientes);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const padding = (maxValue - minValue) * 0.25; // 10% padding
  
    const data = {
      labels: Object.keys(coeficientes),
      datasets: [
        {
          data: values,
          backgroundColor: values.map(value => value >= 0 ? 'rgba(75, 192, 192, 0.2)' : 'rgba(255, 99, 132, 0.2)'),
          borderColor: values.map(value => value >= 0 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'),
          borderWidth: 1,
        },
      ],
    };
  
    const options = {
      indexAxis: 'y',
      scales: {
        x: {
          beginAtZero: false,
          min: minValue - padding,
          max: maxValue + padding,
        },
      },
      plugins: {
        legend: {
          display: false,
        },
        datalabels: {
          anchor: 'end',
          align: 'end',
          formatter: (value) => value.toFixed(4),
        },
      },
    };
  
    return <Bar data={data} options={options} />;
  };

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
              <Row>
              <Card className="mt-4 shadow-sm" border="primary" style={{ maxWidth: '400px', margin: 'auto' }}>
                <Card.Body className="text-center">
                <h5>Scores por Modelo:</h5>
                <Table bordered style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f2f2f2', textAlign: 'left' }}>
                      <th style={{ padding: '10px', borderBottom: '2px solid #ddd', width: "75%" }}>Modelo</th>
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
                </Card.Body>
              </Card>
              <Card className="mt-4 shadow-sm" border="primary" style={{ maxWidth: '900px', margin: 'auto' }}>
                <Card.Body className="text-center">
                  <h3>
                    <FaStar style={{ color: '#FFD700', marginRight: '8px' }} />
                    Mejor Modelo: 
                    <Badge bg="info" className="ms-2">{modelo.mejor_modelo}</Badge>
                  </h3>
                  <h4 className="mt-3">
                    <FaChartLine style={{ marginRight: '8px' }} />
                    Score: <span style={{ fontWeight: 'bold', color: '#007bff' }}>{modelo.score.toFixed(3)}</span>
                  </h4>
                  <Row>
                    <Col>
                  {metrics && (
                    <>
                      <h5 className="mt-4">Métricas del Mejor Modelo:</h5>
                      <Table borderless>
                        <tbody>
                          {Object.entries(metrics).map(([key, value]) => (
                            <tr key={key}>
                              <td style={{ textAlign: 'left', paddingLeft: "10px" }}>
                                <strong>{key+": "}</strong> {typeof value === 'number' ? value.toFixed(4) : JSON.stringify(value)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </>
                  )}
                  </Col>
                  {coeficientes && (
                    <Col>
                    <>
                      <h5 className="mt-4">Coeficientes del Mejor Modelo:</h5>
                      <BarGraph coeficientes={coeficientes} />
                      <p style={{ textAlign: 'center', marginTop: '10px' }}>
                        Intercepto: <strong>{intercepto.toFixed(4)}</strong>
                      </p>
                    </>
                    </Col>
                  )}
                  {caracteristicas && (
                    <Col>
                    <>
                      <h5 className="mt-4">Características del Mejor Modelo:</h5>
                      <Table borderless>
                        <tbody>
                          {caracteristicas.map((caracteristica) => (
                            <tr key={caracteristica}>
                              <td style={{ textAlign: 'left', paddingLeft: "10px" }}>
                                {caracteristica}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </>
                    </Col>
                  )}
                  </Row>
                </Card.Body>
              </Card>
              </Row>
            )}
            </Container>
            
            {estadoEntrenamiento === 2 && (
              <Container style={{ paddingTop: '20px' }}>
                <h2>Predecir</h2>
                <div className="d-flex align-items-center">
                  <Table bordered style={{ flex: '1' }}>
                    <thead>
                      <tr>
                        <th>Parámetro</th>
                        {inputTables.map((table, index) => (
                          <th key={table.id}>
                            {`Predicción ${index + 1}`}
                            <Button
                              variant="danger"
                              onClick={() => handleRemoveTable(table.id)}
                              size="sm"
                              style={{
                                marginLeft: '10px',
                                display: inputTables.length === 1 ? 'none' : 'inline-block',
                              }}
                            >
                              <FaTrash />
                            </Button>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {parametros.map((param, rowIndex) => (
                        <tr key={param}>
                          <td>{param}</td>
                          {inputTables.map((table) => (
                            <td key={`${table.id}-${param}`}>
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
                                  onKeyDown={(e) => handleKeyDown(e, table.id, param)}
                                  ref={(el) => {
                                    if (el) {
                                      inputRefs.current[`${table.id}-${param}`] = el;
                                    }
                                  }}
                                />
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                      <tr>
                        <td><strong>{modelo.variable}</strong></td>
                        {inputTables.map((table) => (
                          <td key={`${table.id}-resultado`}>
                            <div className="resultado-cuadro">{ results[table.id] }</div>
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </Table>
                  <div className="ms-4">
                    {inputTables.length < 10 && (
                      <Button
                        variant="success"
                        onClick={handleAddTable}
                        size="lg"
                        className="d-flex align-items-center"
                      >
                        <FaPlus />
                      </Button>
                    )}
                  </div>
                </div>
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
