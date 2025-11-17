import React, { useState } from "react";
import axios from "axios";

const API_BASE =
  (process.env.REACT_APP_API_BASE &&
    process.env.REACT_APP_API_BASE.replace(/\/+$/, "")) ||
  "http://127.0.0.1:5000";

function App() {
  const [news, setNews] = useState("");
  const [mode, setMode] = useState("LR"); // "LR" or "VOTE"
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setErr("");
    setRes(null);
    setLoading(true);
    try {
      const { data } = await axios.post(`${API_BASE}/predict`, {
        news,
        mode,
      });
      setRes(data);
    } catch (error) {
      setErr(error?.response?.data?.error || "Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  const pill = (text) => (
    <span style={{
      padding: "2px 8px", borderRadius: 999, fontSize: 12,
      background: text === "Fake News" ? "#fee2e2" : "#dcfce7",
      border: "1px solid rgba(0,0,0,0.1)"
    }}>
      {text}
    </span>
  );

  return (
    <div style={{maxWidth: 900, margin: "24px auto", padding: 16, fontFamily: "system-ui, Arial"}}>
      <h1 style={{fontSize: 28, marginBottom: 8}}>📰 Fake News Detector</h1>
      <p style={{marginTop: 0, color: "#555"}}>
        Paste a news article, choose how to decide the final result, and click <b>Predict</b>.
      </p>

      <form onSubmit={onSubmit} style={{marginTop: 16}}>
        <div style={{display: "flex", gap: 12, alignItems: "center", marginBottom: 8}}>
          <label>
            <input
              type="radio"
              name="mode"
              value="LR"
              checked={mode === "LR"}
              onChange={() => setMode("LR")}
            />{" "}
            Final = Logistic Regression (your preferred)
          </label>
          <label>
            <input
              type="radio"
              name="mode"
              value="VOTE"
              checked={mode === "VOTE"}
              onChange={() => setMode("VOTE")}
            />{" "}
            Final = Majority Vote (LR, DT, GB, RF)
          </label>
        </div>

        <textarea
          rows={10}
          placeholder="Paste news text here..."
          value={news}
          onChange={(e) => setNews(e.target.value)}
          required
          style={{
            width: "100%", padding: 12, borderRadius: 8,
            border: "1px solid #ccc", outline: "none", resize: "vertical"
          }}
        />

        <div style={{marginTop: 12, display: "flex", gap: 8}}>
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "10px 16px", borderRadius: 8,
              border: "none", background: "#2563eb", color: "white",
              cursor: "pointer", opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
          <button
            type="button"
            onClick={() => { setNews(""); setRes(null); setErr(""); }}
            style={{
              padding: "10px 16px", borderRadius: 8,
              border: "1px solid #ddd", background: "white", cursor: "pointer"
            }}
          >
            Clear
          </button>
        </div>
      </form>

      {err && (
        <div style={{marginTop: 16, color: "#b91c1c"}}>
          <b>Error:</b> {err}
        </div>
      )}

      {res && (
        <div style={{
          marginTop: 20, padding: 16, border: "1px solid #eee",
          borderRadius: 12, background: "#fafafa"
        }}>
          <div style={{display: "flex", justifyContent: "space-between", alignItems: "center"}}>
            <h2 style={{margin: 0}}>Results</h2>
            <div>Mode: <b>{res.mode}</b></div>
          </div>

          <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12}}>
            <div style={{padding: 12, background: "white", borderRadius: 8, border: "1px solid #eee"}}>
              <div>Logistic Regression</div>
              <div style={{marginTop: 6}}>{pill(res.models.LR)}</div>
              {typeof res.LR_probability_real === "number" && (
                <div style={{marginTop: 6, fontSize: 12, color: "#555"}}>
                  LR P(Not Fake): {(res.LR_probability_real * 100).toFixed(1)}%
                </div>
              )}
            </div>
            <div style={{padding: 12, background: "white", borderRadius: 8, border: "1px solid #eee"}}>
              <div>Decision Tree</div>
              <div style={{marginTop: 6}}>{pill(res.models.DT)}</div>
            </div>
            <div style={{padding: 12, background: "white", borderRadius: 8, border: "1px solid #eee"}}>
              <div>Gradient Boosting</div>
              <div style={{marginTop: 6}}>{pill(res.models.GB)}</div>
            </div>
            <div style={{padding: 12, background: "white", borderRadius: 8, border: "1px solid #eee"}}>
              <div>Random Forest</div>
              <div style={{marginTop: 6}}>{pill(res.models.RF)}</div>
            </div>
          </div>

          <div style={{marginTop: 16, padding: 12, borderRadius: 8, background: "white", border: "1px solid #eee"}}>
            <div style={{fontSize: 18}}>✅ Final Result: <b>{res.final}</b></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
