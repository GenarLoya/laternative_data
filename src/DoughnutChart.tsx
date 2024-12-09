import React from "react";
import { Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  ChartOptions,
} from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

const DoughnutChart: React.FC = () => {
  // Datos estáticos
  const data = {
    spamCount: 1500,
    notSpamCount: 3672,
  };

  // Datos para la gráfica de dona
  const chartData = {
    labels: ["Spam", "No Spam"],
    datasets: [
      {
        data: [data.spamCount, data.notSpamCount],
        backgroundColor: ["#FF9800", "#5DC1B9"], // Colores personalizados
        borderWidth: 1,
      },
    ],
  };

  // Opciones para la gráfica de dona
  const options: ChartOptions<"doughnut"> = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            const value = context.raw as number;
            const percentage = value.toFixed();
            return `${context.label}: ${percentage}`;
          },
        },
      },
    },
    cutout: "70%", // Hace que la gráfica sea una dona
  };

  return (
    <div className="flex flex-col place-content-center bg-white">
      <h1 className="text-2xl font-bold mb-4">Análisis de Proporción de Correos Spam</h1>
      <Doughnut data={chartData} options={options} />
    </div>
  );
};

export default DoughnutChart;