import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./Auth.css";

function Register() {
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] =
    useState("");

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleRegister = (e) => {
    e.preventDefault();

    setError("");
    setSuccess("");

    const normalizedName = name.trim();
    const normalizedEmail = email.trim().toLowerCase();

    if (!normalizedName || !normalizedEmail || !password || !confirmPassword) {
      setError("Please fill in all fields.");
      return;
    }

    if (password.length < 8) {
      setError(
        "Password must contain at least 8 characters."
      );
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    const users = JSON.parse(
      localStorage.getItem("dataopti_users") || "[]"
    );

    const existingUser = users.find(
      (user) => user.email?.trim().toLowerCase() === normalizedEmail
    );

    if (existingUser) {
      setError(
        "An account with this email already exists."
      );
      return;
    }

    const newUser = {
      id: Date.now(),
      name: normalizedName,
      email: normalizedEmail,
      password
    };

    users.push(newUser);

    localStorage.setItem(
      "dataopti_users",
      JSON.stringify(users)
    );

    setSuccess(
      "Account created successfully. Redirecting to login..."
    );

    setTimeout(() => {
      navigate("/login");
    }, 1200);
  };

  return (
    <div className="auth-container auth-register">
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

        <h2>Create Account</h2>

        <p className="auth-description">
          Enter your details to create a DataOpti AI account. Use your email and password to sign in.
        </p>

        {error && (
          <div className="auth-error">
            {error}
          </div>
        )}

        {success && (
          <div className="auth-success">
            {success}
          </div>
        )}

        <form onSubmit={handleRegister}>

          <label>Full Name</label>

          <input
            type="text"
            placeholder="Enter your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoComplete="name"
            required
          />

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
            placeholder="Create password (at least 8 characters)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
            minLength={8}
            required
          />

          <label>Confirm Password</label>

          <input
            type="password"
            placeholder="Confirm password"
            value={confirmPassword}
            onChange={(e) =>
              setConfirmPassword(e.target.value)
            }
            autoComplete="new-password"
            minLength={8}
            required
          />

          <button
            type="submit"
            className="auth-button"
          >
            Create Account
          </button>

        </form>

        <p className="auth-switch">
          Already have an account?{" "}
          <Link to="/login">
            Login
          </Link>
        </p>

      </section>
      </main>
      </div>
    </div>
  );
}

export default Register;
