/**
 * Main App Component
 *
 * React Router setup and main application layout
 */

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './pages/Home';
import Scanner from './pages/Scanner';
import Solver from './pages/Solver';
import GuidedSolver from './pages/GuidedSolver';
import About from './pages/About';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/scanner" element={<Scanner />} />
            <Route path="/solver" element={<Solver />} />
            <Route path="/guided" element={<GuidedSolver />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <footer className="footer">
          <div className="container">
            <p>
              Made with ❤️ using React, FastAPI, and ML |{' '}
              <a href="https://github.com/jitesh523/rubiks" target="_blank" rel="noopener noreferrer">
                View on GitHub
              </a>
            </p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
