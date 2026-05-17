import { useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from "recharts";

function App() {
  const [model, setModel] = useState("logistic_regression");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({});
  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        null,
        { params: { model_name: model } }
      );

      setResults(prev => ({
        ...prev,
        [model]: response.data
      }));

    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>
      <h1>ML Dashboard</h1>

      <div style={{ marginBottom: "20px" }}>
        <select onChange={(e) => setModel(e.target.value)}>
          <option value="logistic_regression">Logistic Regression</option>
          <option value="linear_regression">Linear Regression</option>
        </select>

        <button onClick={handlePredict} style={{ marginLeft: "10px" }}>
          {loading ? "Running..." : "Run Model"}
        </button>
      </div>

      {Object.keys(results).length > 0 && (
        <div>
          <h2>Accuracy: {results[model]?.accuracy?.toFixed(2)}</h2>

          <BarChart
            width={500}
            height={300}
            data={[
              { name: "TP", value: results[model]?.confusion_matrix?.TP },
              { name: "TN", value: results[model]?.confusion_matrix?.TN },
              { name: "FP", value: results[model]?.confusion_matrix?.FP },
              { name: "FN", value: results[model]?.confusion_matrix?.FN }
            ]}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" />
          </BarChart>
          {Object.keys(results).length > 0 && (
            <div>
              <h2>Model Comparison</h2>
              {Object.entries(results).map(([name, res]) => (
                <div key={name}>
                  <strong>{name}</strong>: {res.accuracy?.toFixed(2)}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;