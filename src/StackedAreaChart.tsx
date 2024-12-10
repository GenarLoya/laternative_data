import React, { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface HiddenLayerData {
  accuracy: number;
  test_size: number;
  random_state: number;
  activation: string;
  hidden_layer_sizes: number[] | number;
}

const StackedAreaChart: React.FC = () => {
  const [hiddenLayerData, setHiddenLayerData] = useState<HiddenLayerData[]>([]);

  // Cargar datos desde el archivo JSON al montar el componente
  useEffect(() => {
    fetch("/neuronal_model_results.json") // Asumiendo que el archivo JSON está en la carpeta `public/data`
      .then((response) => response.json())
      .then((data) => setHiddenLayerData(data))
      .catch((error) => console.error("Error al cargar los datos:", error));
  }, []);

  if (hiddenLayerData.length === 0) {
    return <div>Loading...</div>; // Mostrar un mensaje mientras se cargan los datos
  }

  // Obtener activaciones únicas para el eje X
  const activations = [...new Set(hiddenLayerData.map((item) => item.activation))];

  // Agrupar datos por tamaño de capa
  const groupedByLayerSizes = hiddenLayerData.reduce((acc, item) => {
    const key = JSON.stringify(item.hidden_layer_sizes); // Convertir a cadena para usar como clave
    if (!acc[key]) acc[key] = [];
    acc[key].push(item);
    return acc;
  }, {} as Record<string, HiddenLayerData[]>);

  // Construir datos para el gráfico
  const chartData = {
    labels: activations,
    datasets: Object.entries(groupedByLayerSizes).map(([layerSize, data]) => {
      const accuracyByActivation = activations.map((activation) => {
        const entry = data.find((item) => item.activation === activation);
        return entry ? entry.accuracy : null; // Rellenar valores faltantes con null
      });

      return {
        label: `Capas: ${layerSize}`,
        data: accuracyByActivation,
        fill: false,
        borderColor: `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(
          Math.random() * 255
        )}, ${Math.floor(Math.random() * 255)}, 1)`, // Color aleatorio para cada línea
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
      };
    }),
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: "Precisión Según Activación",
        font: {
          size: 18,
          family: "Roboto, sans-serif",
        },
        color: "#333",
      },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        titleColor: "#fff",
        bodyColor: "#fff",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Función de Activación",
          font: {
            size: 14,
            family: "Roboto, sans-serif",
          },
          color: "#555",
        },
        grid: {
          borderColor: "#ccc",
          borderWidth: 1,
          color: "#e5e5e5",
        },
      },
      y: {
        title: {
          display: true,
          text: "Precisión",
          font: {
            size: 14,
            family: "Roboto, sans-serif",
          },
          color: "#555",
        },
        min: 0.95, // Mínimo del eje Y
        max: 0.99, // Máximo del eje Y
        beginAtZero: false,
        grid: {
          borderColor: "#ccc",
          borderWidth: 1,
          color: "#e5e5e5",
        },
      },
    },
  };


  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div style={{ height: "400px" }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
};

export default StackedAreaChart;
