import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './context/AuthContext'
import Layout from './components/Layout'
import AnimatedBackground from './components/AnimatedBackground'

// Pages
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import NewScreening from './pages/NewScreening'
import PatientList from './pages/PatientList'
import PatientDetail from './pages/PatientDetail'
import ReviewQueue from './pages/ReviewQueue'
import Research from './pages/Research'
import FederatedLearning from './pages/FederatedLearning'
import Explainability from './pages/Explainability'
import Reports from './pages/Reports'
import Admin from './pages/Admin'

const ProtectedRoute = ({ children }) => {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" />
  return children
}

function App() {
  const { user } = useAuth()

  return (
    <Router>
      <AnimatedBackground />
      <Routes>
        <Route path="/login" element={<Login />} />
        
        <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="screening/new" element={<NewScreening />} />
          <Route path="patients" element={<PatientList />} />
          <Route path="patients/:id" element={<PatientDetail />} />
          <Route path="reviews" element={<ReviewQueue />} />
          <Route path="explainability" element={<Explainability />} />
          <Route path="reports" element={<Reports />} />
          <Route path="research" element={<Research />} />
          <Route path="federated" element={<FederatedLearning />} />
          <Route path="admin" element={<Admin />} />
        </Route>
      </Routes>
    </Router>
  )
}

export default App
