import { useState } from "react";
import InputBox from "./components/InputBox.jsx";
import Loader from "./components/Loader.jsx";
import ErrorMessage from "./components/ErrorMessage.jsx";
import Summary from "./components/Summary.jsx";
import Chart from "./components/Chart.jsx";
import CommentList from "./components/CommentList.jsx";
import './App.css'

function App() {
  const [prUrl, setPrUrl] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const API_URL = import.meta.env.VITE_API_URL;

  const handleAnalyze = async () => {
    const isValid = /github\.com\/.+\/.+\/pull\/\d+/.test(prUrl);

    if (!isValid) {
     setError("Please enter a valid GitHub PR URL");
     return;
    }

    setLoading(true);
    setError("");
    setData(null);

    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ pr_url: prUrl })
      });

      const result = await res.json();

      if (!res.ok || result.error) {
        setError(result.error || result.message || "Something went wrong");
      } else {
        setData(result);
      }
    } catch (err) {
      setError(err.message || "Backend not reachable");
      }

    setLoading(false);
  };

  return (
    <div className="container">
      <h1 className="title">Code Review Comment Analyzer</h1>
  
      <div className="card-box">
        <InputBox
          prUrl={prUrl}
          setPrUrl={setPrUrl}
          onAnalyze={handleAnalyze}
          loading={loading}
        />
        <ErrorMessage message={error} />
      </div>
  
      {loading && <Loader />}
  
      {data && (
        <>
          <div className="card-box">
            <Summary data={data} />
          </div>
  
          <div className="card-box">
            <CommentList results={data.results} />
          </div>
        </>
      )}
    </div>
  );
}

export default App;