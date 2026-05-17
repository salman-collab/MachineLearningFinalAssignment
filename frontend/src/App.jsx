import { useState } from "react";
import axios from "axios";
import "./App.css";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  LineChart,
  Line,
  Legend
} from "recharts";

function App() {
  const [model, setModel] = useState("logistic_regression");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        null,
        { params: { model_name: model } }
      );
      setData(response.data);
    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="container">

      <div className="header">
        ML Model Dashboard
      </div>

      {/* Controls */}
      <div className="card">
        <div className="controls">
          <select onChange={(e) => setModel(e.target.value)}>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="linear_regression">Linear Regression</option>
          </select>

          <button onClick={handlePredict}>
            {loading ? "Running..." : "Run Model"}
          </button>
        </div>
      </div>

      {/* ================= LOGISTIC REGRESSION ================= */}
      {data && data.accuracy && data.confusion_matrix && (
        <div className="card">
          <h3>Classification Performance</h3>

          <p><strong>Accuracy:</strong> {data.accuracy.toFixed(2)}</p>
          <p><strong>Precision:</strong> {data.precision.toFixed(2)}</p>
          <p><strong>Recall:</strong> {data.recall.toFixed(2)}</p>

          <BarChart
            width={500}
            height={300}
            data={[
              { name: "TP", value: data.confusion_matrix.TP },
              { name: "TN", value: data.confusion_matrix.TN },
              { name: "FP", value: data.confusion_matrix.FP },
              { name: "FN", value: data.confusion_matrix.FN }
            ]}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" />
          </BarChart>
        </div>
      )}

      {/* ================= LINEAR REGRESSION ================= */}
      {data && data.mse && data.actual && data.predicted && (
        <div className="card">
          <h3>Regression Performance</h3>

          <p><strong>MSE:</strong> {data.mse.toFixed(4)}</p>

          <LineChart
            width={600}
            height={300}
            data={data.actual.map((val, i) => ({
              index: i,
              actual: val,
              predicted: data.predicted[i]
            }))}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="index" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="actual" name="Actual" stroke="#4CAF50" strokeWidth={2} />
            <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#2196F3" strokeWidth={2} />
          </LineChart>
        </div>
      )}

    </div>
  );
}

export default App;