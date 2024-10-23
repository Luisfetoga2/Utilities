// DetalleModelo.js
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

function DetalleModelo() {
  const { id } = useParams();
  const [modelo, setModelo] = useState(null);

  return (
    <div>
      <h1>detalle modelo {id}</h1>
    </div>
  );
}

export default DetalleModelo;
