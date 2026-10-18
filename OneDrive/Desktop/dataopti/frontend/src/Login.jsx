import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./Auth.css";

function Login({ onLogin }) {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleLogin = (e) => {
    e.preventDefault();

    setError("");

    const normalizedEmail = email.trim().toLowerCase();

    if (!normalizedEmail || !password) {
      setError("Please enter email and password.");
      return;
    }

    const users = JSON.parse(
      localStorage.getItem("dataopti_users") || "[]"
    );

    const user = users.find((item) =>
      item.email?.trim().toLowerCase() === normalizedEmail && item.password === password
    );

    if (!user) {
      setError("Invalid email or password.");
      return;
    }

    localStorage.setItem(
      "dataopti_logged_user",
      JSON.stringify(user)
    );

    onLogin(user);

    navigate("/");
  };

  return (
    <div className="auth-container auth-login">
      <div className="auth-layout">
      <aside className="auth-showcase">
        <div className="showcase-brand">
          <span className="showcase-mark" aria-hidden="true">✦</span>
          <span className="showcase-wordmark">DataOpti <b>AI</b></span>
          <span className="showcase-product">DATA STUDIO</span>
        </div>

        <div className="showcase-copy">
          <span className="showcase-kicker">A SMARTER START FOR EVERY DATASET</span>
          <h2>Make room for<br /><span>better insights.</span></h2>
          <p>Explore your data, spot what needs attention, and get your machine-learning workflow ready.</p>
        </div>

        <div className="showcase-dashboard" aria-hidden="true">
          <div className="showcase-window-bar">
            <span>QUALITY OVERVIEW</span>
            <span className="window-dots">•••</span>
          </div>
          <div className="showcase-score-row">
            <div>
              <small>DATA HEALTH</small>
              <strong>92<span>/100</span></strong>
            </div>
            <span className="showcase-score-badge">↗ 18.4%</span>
          </div>
          <div className="showcase-chart">
            <i style={{ height: "34%" }} />
            <i style={{ height: "48%" }} />
            <i style={{ height: "42%" }} />
            <i style={{ height: "68%" }} />
            <i style={{ height: "57%" }} />
            <i style={{ height: "82%" }} />
            <i style={{ height: "72%" }} />
            <i style={{ height: "100%" }} />
          </div>
          <div className="showcase-insight">
            <span className="insight-check">✓</span>
            <span><b>Analysis ready</b><small>Quality checks complete</small></span>
            <span className="insight-arrow">↗</span>
          </div>
        </div>

        <div className="showcase-caption">
          <span>01 / 03</span>
          <span>From raw files to a clearer next step</span>
        </div>
      </aside>

      <main className="auth-panel">      <section className="auth-card">
        <div className="auth-context"><span>ACCOUNT PORTAL</span><span className="auth-context-dot" /> DATAOPTI WORKSPACE</div>

        <div className="auth-logo">
          <div className="logo-icon">D</div>
          <h1>DataOpti AI</h1>
        </div>

        <p className="auth-subtitle">
          AI-Based Automated Data Cleaning
        </p>

        <h2>Welcome Back</h2>

        <p className="auth-description">
          Login to continue to your DataOpti AI dashboard.
        </p>

        {error && (
          <div className="auth-error">
            {error}
          </div>
        )}

        <form onSubmit={handleLogin}>

          <label>Email Address</label>

          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
          />

          <label>Password</label>

          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />

          <button
            type="submit"
            className="auth-button"
          >
            Login
          </button>

        </form>

        <p className="auth-switch">
          Don't have an account?{" "}
          <Link to="/register">
            Create Account
          </Link>
        </p>

      </section>
      </main>
      </div>
    </div>
  );
}

export default Login;
