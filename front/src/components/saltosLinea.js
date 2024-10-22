import React, { useState } from "react";
import Container from "react-bootstrap/esm/Container";
import { FaSyncAlt, FaRegCopy, FaCheck } from "react-icons/fa";

function SaltosLinea() {
    const [textoOriginal, setTextoOriginal] = useState("");
    const [textoTransformado, setTextoTransformado] = useState("");
    const [copiado, setCopiado] = useState(false);
    const [pegado, setPegado] = useState(false);

    function borrarSaltosLinea(texto) {
        const textoProcesado = texto
            .split('\n')
            .map(linea => linea.trim())
            .filter(linea => linea)
            .join(' ');

        setTextoOriginal(texto);
        setTextoTransformado(textoProcesado);
    }

    const handleChange = (e) => {
        borrarSaltosLinea(e.target.value);
    };

    const handlePegar = async () => {
        try {
            const textoPegar = await navigator.clipboard.readText();
            borrarSaltosLinea(textoPegar);
            setPegado(true);
            setTimeout(() => setPegado(false), 2000);
        } catch (error) {
            console.error("Error al pegar desde el portapapeles:", error);
        }
    };

    const handleCopiar = async () => {
        try {
            await navigator.clipboard.writeText(textoTransformado);
            setCopiado(true);
            setTimeout(() => setCopiado(false), 2000);
        } catch (error) {
            console.error("Error al copiar al portapapeles:", error);
        }
    };

    const handleReiniciar = () => {
        setTextoOriginal("");
        setTextoTransformado("");
        setCopiado(false);
        setPegado(false);
    };

    return (
        <Container fluid className="d-flex justify-content-center align-items-center" style={{ minHeight: 'calc(100vh - 56px)' }}>
            <div className="w-75">
                <h1 className="text-center">Borrar saltos de línea</h1>
                <p className="text-center">Esta herramienta permite borrar los saltos de línea y espacios innecesarios.</p>
                <div className="row mb-3">
                    <div className="col">
                        <textarea
                            className="form-control"
                            rows="10"
                            placeholder="Pega aquí tu texto"
                            value={textoOriginal}
                            onChange={handleChange}
                            style={{ resize: "none" }}
                        ></textarea>
                        <div className="d-flex justify-content-center mt-2">
                            <button className="btn btn-secondary" onClick={handlePegar}>
                                {pegado ? <FaCheck /> : <FaRegCopy />}{" "}
                                {pegado ? "Texto pegado" : "Pegar texto"}
                            </button>
                        </div>
                    </div>
                    <div className="col">
                        <textarea
                            className="form-control"
                            rows="10"
                            placeholder="Acá estará el resultado"
                            value={textoTransformado}
                            readOnly
                            style={{ resize: "none" }}
                        ></textarea>
                        <div className="d-flex flex-column align-items-center mt-2">
                            <button className="btn btn-secondary mb-1" onClick={handleCopiar}>
                                {copiado ? <FaCheck /> : <FaRegCopy />}{" "}
                                {copiado ? "Texto copiado" : "Copiar texto"}
                            </button>
                        </div>
                    </div>
                </div>
                <div className="d-flex justify-content-center mt-3">
                    <button className="btn btn-primary" onClick={handleReiniciar} style={{ borderRadius: '50%', width: '50px', height: '50px' }}>
                        <FaSyncAlt />
                    </button>
                </div>
            </div>
        </Container>
    );
}

export default SaltosLinea;
