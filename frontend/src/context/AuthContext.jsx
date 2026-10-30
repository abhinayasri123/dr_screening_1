import { createContext, useContext, useState, useEffect } from 'react'
import { post } from '../api/client'

const AuthContext = createContext(null)

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem('auth_user')
      return saved ? JSON.parse(saved) : null
    } catch {
      return null
    }
  })

  const login = async (email, password, role = 'clinician') => {
    try {
      const res = await post('/auth/login', { email, password })
      localStorage.setItem('auth_token', res.access_token)
      localStorage.setItem('token', res.access_token)
      const userObj = res.user || { email, role }
      localStorage.setItem('auth_user', JSON.stringify(userObj))
      setUser(userObj)
      return userObj
    } catch (err) {
      throw err
    }
  }

  const logout = () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('token')
    localStorage.removeItem('auth_user')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
