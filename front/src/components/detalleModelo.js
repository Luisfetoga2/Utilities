import React, { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { Container, Button, Form, Table, Col, Row, Card, ProgressBar, Spinner, Badge } from 'react-bootstrap';
import { FaStar, FaChartLine, FaPlay } from 'react-icons/fa';

function DetalleModelo() {
  const { id } = useParams();
  const [modelFound, setModelFound] = useState(0);
  const [modelo, setModelo] = useState(null);
  const [estadoEntrenamiento, setEstadoEntrenamiento] = useState(0); // 0: no entrenado, 1: entrenando, 2: entrenado
  const [resultado, setResultado] = useState("-");
  const [parametros, setParametros] = useState([]);
  const [inputValues, setInputValues] = useState({}); // Store input values for prediction
  const [inputLabels, setInputLabels] = useState({}); // Store input labels for prediction
  const [scores, setScores] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    async function fetchModelo() {
      try {
        const response = await fetch(`http://127.0.0.1:8000/modelos/${id}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Modelo encontrado");
        console.log(data)
        setModelo(data);
        setModelFound(1);
        if (data.entrenado) {
          console.log("Modelo entrenado");
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
        console.error("Network error:", error);
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

      console.log("Scores:", sorted);
      setScores(sorted); // Store the scores in state
    } catch (error) {
      console.error("Error fetching scores:", error);
    }
  }, [id]);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/modelos/${id}/metrics`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setMetrics(data); // Store metrics in state
    } catch (error) {
      console.error("Error fetching metrics:", error);
    }
  }, [id]);
  
  useEffect(() => {
    if (estadoEntrenamiento === 2) {
      console.log("Estado de entrenamiento:", estadoEntrenamiento);
      fetchScores();
      fetchMetrics();
    }
  }, [estadoEntrenamiento]);
  

  const initializeInputValues = (params) => {
    const initialValues = {};
    params.forEach(param => {
      initialValues[param] = "";
    });
    setInputValues(initialValues);
  };

  const handlePredecir = useCallback(async () => {
    try {
      const jsonBody = {};
      parametros.forEach((param) => {
        jsonBody[param] = inputValues[param];
      });

      const response = await fetch(`http://127.0.0.1:8000/modelos/${id}/predecir`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(jsonBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResultado(data.prediccion);
    } catch (error) {
      console.error("Prediction error:", error);
      setResultado("Error en la predicción");
    }
  }, [id, parametros, inputValues]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
  
    setInputValues((prevValues) => ({
      ...prevValues,
      [name]: value,
    }));

    console.log(inputValues);
  };

  const handleSelectChange = (e) => {
    const { name, value } = e.target;
  
    setInputValues((prevValues) => ({
      ...prevValues,
      [name]: value,
    }));
  };
  
  // Use useEffect to react to changes in inputValues
  const checkAllFilled = useCallback(() => {
    return parametros.every((param) => inputValues[param] !== "");
  }, [parametros, inputValues]);
  
  useEffect(() => {
    if (parametros.length > 0 && checkAllFilled()) {
      handlePredecir();
    } else {
      setResultado("-");
    }
  }, [inputValues, parametros.length, checkAllFilled, handlePredecir]);
  

  const handleEliminar = () => {
    console.log("Eliminar modelo", id);
    fetch(`http://127.0.0.1:8000/modelos/${id}`, {
      method: "DELETE",
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Modelo eliminado:", data);
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
                <Row>
                <h2>Predecir</h2>
                <Col>
                  <Form>
                    <Table bordered>
                      <thead>
                        <tr>
                          <th>Parámetro</th>
                          <th>Valor</th>
                        </tr>
                      </thead>
                      <tbody>
                        {parametros.map((param) => (
                          <tr key={param}>
                            <td>{param}</td>
                            <td>
                              {inputLabels[param] ? (
                                <Form.Select
                                  name={param}
                                  value={inputValues[param] || ""}
                                  onChange={handleSelectChange}
                                >
                                  <option value="">Seleccionar...</option>
                                  {inputLabels[param].map((label) => (
                                    <option key={label} value={label}>{label}</option>
                                  ))}
                                </Form.Select>
                              ) : (
                                <Form.Control
                                  type="number"
                                  name={param}
                                  value={inputValues[param] || ""}
                                  onChange={handleInputChange}
                                />
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </Form>
                </Col>
                <Col>
                  <h3>Resultado:</h3>
                  <div className="resultado-cuadro">{resultado}</div>
                </Col>
                </Row>
              </Container>
            )}
          </>
        )}
      </div>
    </Container>
  );
}

export default DetalleModelo;
