import DoughnutChart from './DoughnutChart'
import StackedAreaChart from './StackedAreaChart'

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <header className="bg-white shadow p-4 mb-4 rounded-lg">
        <h1 className="text-2xl font-bold text-gray-700">Dashboard Spam Predictor</h1>
      </header>
      
      <header className="bg-white p-4 mb-4">
          <div className="bg-white p-6 place-content-center">
            <StackedAreaChart />
          </div>
      </header>
      <main className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <DoughnutChart />
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-800">Gráfico de Nube</h2>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-800">Detor de Spam</h2>
        </div>
      </main>
    </div>
  );
};

export default App;
