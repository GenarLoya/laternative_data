import React, { useEffect, useState } from 'react';
import ReactWordcloud from 'react-d3-cloud';
import neuronal_nodel_results from '../public/neuronal_model_results.json'; // Asegúrate de que la ruta sea correcta

interface WordData {
  text: string;
  value: number;
}

const WordCloud: React.FC = () => {
  const [words, setWords] = useState<WordData[]>([]);

  useEffect(() => {
    // Cargar los datos de las palabras
    const fetchData = () => {
      const wordCounts: { [key: string]: number } = {};

      // Extraemos las activaciones de los datos
      neuronal_nodel_results.forEach((item: any) => {
        const activation = item.activation;
        wordCounts[activation] = (wordCounts[activation] || 0) + 1;
      });

      // Convertir a formato adecuado para react-d3-cloud
      const wordArray = Object.keys(wordCounts).map((key) => ({
        text: key,
        value: wordCounts[key],
      }));

      setWords(wordArray);
    };

    fetchData();
  }, []);

  const options = {
    width: 600,
    height: 400,
    fontSize: (word: any) => Math.log(word.value) * 10, // Tamaño de la fuente basado en el valor
    font: 'Arial, sans-serif',
    padding: 5,
    rotate: (word: any) => (Math.random() > 0.5 ? 0 : 90), // Rotación aleatoria
    scale: 'sqrt', // Escalado de palabras
    spiral: 'rectangular', // Disposición rectangular de las palabras
  };

  return (
    <div className="max-w-4xl mx-auto p-8 bg-white rounded-lg shadow-lg">
      <h2 className="text-center text-3xl font-semibold text-gray-800 mb-6">Word Cloud</h2>
      <div className="bg-gray-100 p-6 rounded-lg shadow-md">
        <ReactWordcloud words={words} options={options} />
      </div>
    </div>
  );
};

export default WordCloud;
