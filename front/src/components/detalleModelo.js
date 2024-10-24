import React, { useEffect, useState  } from "react";
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

  useEffect(() => {
    async function fetchModelo() {
      try {
        const response = await fetch(`http://127.0.0.1:8000/modelos/${id}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Modelo encontrado");
        setModelo(data);
        setModelFound(1);
        if (data.entrenado) {
          setEstadoEntrenamiento(2);
          const response = await fetch(`http://127.0.0.1:8000/modelos/${id}/parametros`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const paramsData = await response.json();
          setParametros(paramsData);
          initializeInputValues(paramsData);
        }
      } catch (error) {
        console.error("Network error:", error);
        setModelFound(-1);
      }
    }

    fetchModelo();
  }, [id]);

  const initializeInputValues = (params) => {
    const initialValues = {};
    params.forEach(param => {
      initialValues[param] = "";
    });
    setInputValues(initialValues);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
  
    setInputValues((prevValues) => ({
      ...prevValues,
      [name]: value,
    }));

    console.log(inputValues);
  };
  
  // Use useEffect to react to changes in inputValues
  function checkAllFilled() {
    console.log(parametros.every((param) => inputValues[param] !== ""));
    return parametros.every((param) => inputValues[param] !== "");
  }
  
  useEffect(() => {
    if (parametros.length > 0 && checkAllFilled()) {
      handlePredecir();
    } else {
      setResultado("-");
    }
  }, [inputValues]);
  

  const handleEliminar = () => {
    console.log("Eliminar modelo", id);
    // TODO
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

  const handlePredecir = async () => {
    try {
      const jsonBody = {}
      parametros.forEach(param => {
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
      console.log("Prediction result:", data.prediccion);
      setResultado(data.prediccion);
    } catch (error) {
      console.error("Prediction error:", error);
      setResultado("Error en la predicción");
    }
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
                        <h3>
                          <FaStar style={{ color: '#FFD700', marginRight: '8px' }} />
                          Mejor Modelo: 
                          <Badge bg="info" className="ms-2">{modelo.mejor_modelo}</Badge>
                        </h3>
                        <h4 className="mt-3">
                          <FaChartLine style={{ marginRight: '8px' }} />
                          R²: <span style={{ fontWeight: 'bold', color: '#007bff' }}>{modelo.score.toFixed(3)}</span>
                        </h4>
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
                              <Form.Control
                                type="text"
                                name={param}
                                value={inputValues[param] || ""}
                                onChange={handleInputChange}
                              />
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
