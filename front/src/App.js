// App.js
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Menu from './components/menu';
import Header from './components/header';

import SaltosLinea from './components/saltosLinea';
import ModelosML from './components/modelosML';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <Routes>
          <Route path="/" element={<Menu />} />
          <Route path="/saltosLinea" element={<SaltosLinea />} />
          <Route path="/modelos" element={<ModelosML />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
