/**
 * Navigation Component
 *
 * Top navigation bar with responsive design
 */

import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navigation.css';

function Navigation() {
    const [isOpen, setIsOpen] = useState(false);
    const location = useLocation();

    const toggleMenu = () => {
        setIsOpen(!isOpen);
    };

    const closeMenu = () => {
        setIsOpen(false);
    };

    const isActive = (path) => {
        return location.pathname === path;
    };

    return (
        <nav className="nav">
            <div className="container nav-container">
                <Link to="/" className="nav-logo" onClick={closeMenu}>
                    🎲 Cube Solver
                </Link>

                <button className="nav-toggle" onClick={toggleMenu} aria-label="Toggle menu">
                    {isOpen ? '✕' : '☰'}
                </button>

                <div className={`nav-links ${isOpen ? 'open' : ''}`}>
                    <Link
                        to="/"
                        className={`nav-link ${isActive('/') ? 'active' : ''}`}
                        onClick={closeMenu}
                    >
                        Home
                    </Link>
                    <Link
                        to="/scanner"
                        className={`nav-link ${isActive('/scanner') ? 'active' : ''}`}
                        onClick={closeMenu}
                    >
                        Scanner
                    </Link>
                    <Link
                        to="/guided"
                        className={`nav-link ${isActive('/guided') ? 'active' : ''}`}
                        onClick={closeMenu}
                    >
                        Guided Solver
                    </Link>
                    <Link
                        to="/solver"
                        className={`nav-link ${isActive('/solver') ? 'active' : ''}`}
                        onClick={closeMenu}
                    >
                        Solver
                    </Link>
                    <Link
                        to="/about"
                        className={`nav-link ${isActive('/about') ? 'active' : ''}`}
                        onClick={closeMenu}
                    >
                        About
                    </Link>
                </div>
            </div>
        </nav>
    );
}

export default Navigation;
