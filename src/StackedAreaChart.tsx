import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface HiddenLayerData {
  accuracy: number;
  test_size: number;
  random_state: number;
  activation: string;
  hidden_layer_sizes: number[];
}

const StackedAreaChart: React.FC = () => {
  const [hiddenLayerData, setHiddenLayerData] = useState<HiddenLayerData[]>([]);
  const [selectedLayerSize, setSelectedLayerSize] = useState<string | null>(null);

  // Cargar datos desde el archivo JSON al montar el componente
  useEffect(() => {
    fetch('/neuronal_model_results.json') // Asumiendo que el archivo JSON está en la carpeta `public/data`
      .then((response) => response.json())
      .then((data) => setHiddenLayerData(data))
      .catch((error) => console.error('Error al cargar los datos:', error));
  }, []);

  if (hiddenLayerData.length === 0) {
    return <div>Loading...</div>; // Mostrar un mensaje mientras se cargan los datos
  }

  const hiddenLayerSizes = [...new Set(hiddenLayerData.map(item => JSON.stringify(item.hidden_layer_sizes)))];
  const accuraciesByLayerSize = hiddenLayerSizes.map(size => {
    return hiddenLayerData.filter(item => JSON.stringify(item.hidden_layer_sizes) === size).map(item => item.accuracy);
  });

  const chartData = {
    labels: hiddenLayerSizes,
    datasets: accuraciesByLayerSize.map((accuracies, index) => {
      const isSelected = selectedLayerSize === hiddenLayerSizes[index];
      return {
        label: `Tamaño capa: ${hiddenLayerSizes[index]}`,
        data: accuracies,
        fill: true,
        backgroundColor: isSelected
          ? `rgba(75, 192, 192, 0.3)` // Sombra más visible para la capa seleccionada
          : `rgba(75, 192, 192, 0.1)`, // Menos visible para las demás capas
        borderColor: `rgba(75, 192, 192, 1)`,
        borderWidth: isSelected ? 3 : 2, // Más gruesa si está seleccionada
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.5,
        onClick: () => setSelectedLayerSize(isSelected ? null : hiddenLayerSizes[index]), // Permite alternar la capa seleccionada
      };
    }),
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: 'Precisión Según Tamaño de Capa',
        font: {
          size: 18,
          family: 'Roboto, sans-serif',
        },
        color: '#333',
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Tamaño de capa',
          font: {
            size: 14,
            family: 'Roboto, sans-serif',
          },
          color: '#555',
        },
        grid: {
          borderColor: '#ccc',
          borderWidth: 1,
          color: '#e5e5e5',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Precisión',
          font: {
            size: 14,
            family: 'Roboto, sans-serif',
          },
          color: '#555',
        },
        beginAtZero: false,
        min: 0.96,
        max: 0.99,
        grid: {
          borderColor: '#ccc',
          borderWidth: 1,
          color: '#e5e5e5',
        },
      },
    },
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div style={{ height: '400px' }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
};

export default StackedAreaChart;
