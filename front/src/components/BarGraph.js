import React from 'react';
import { Bar } from 'react-chartjs-2';

const BarGraph = React.memo(({ coeficientes }) => {
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
});

export default BarGraph;