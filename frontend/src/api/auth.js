import { get, post } from './client'

export const login = (data) => post('/auth/login', data) // using Form data in Context instead usually
export const getMe = () => get('/auth/me')
