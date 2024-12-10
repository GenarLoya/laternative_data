import DoughnutChart from './DoughnutChart'
import { Predictor } from './Predictor';
import CloudGraph from '/CloudGraph.jpeg'
import StackedAreaChart from './StackedAreaChart'

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <header className="bg-white shadow p-4 mb-4 rounded-lg">
        <h1 className="text-2xl font-bold text-gray-700">Dashboard Spam Predictor</h1>
      </header>

      <section className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-3 gap-4 mb-4">
        <div className="bg-white rounded-lg shadow p-6">
          <DoughnutChart />
        </div>
        <section className="bg-white rounded-lg shadow p-6" style={{ gridColumn: "span 2" }}>
          <h1 className="text-2xl font-bold mb-4">Análisis de la palabras mas recurrentes para correos de spam y no spam</h1>
          <img className="items-center" src={CloudGraph} />
        </section>
      </section>
      <section className="bg-white w-full rounded-lg shadow p-6 mb-4">
        <h2 className="text-2xl font-bold mb-4">Detector de Spam</h2>
        <Predictor />
      </section>
      <section className="bg-white p-4 mb-4">
        <div className="bg-white p-6 place-content-center">
          <StackedAreaChart />
        </div>
      </section>
    </div>
  );
};

export default App;
