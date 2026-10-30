import axios from 'axios'

// Support custom external API backend or relative single-origin deployment
const apiBase = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL.replace(/\/$/, '')}/api/v1`
  : '/api/v1'

const client = axios.create({
  baseURL: apiBase,
  headers: {
    'Content-Type': 'application/json',
  },
})

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token') || localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('auth_token')
      localStorage.removeItem('token')
      localStorage.removeItem('auth_user')
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export const get = (url, params) => client.get(url, { params }).then(res => res.data)
export const post = (url, data) => client.post(url, data).then(res => res.data)
export const put = (url, data) => client.put(url, data).then(res => res.data)
export const del = (url) => client.delete(url).then(res => res.data)
export const postForm = (url, formData) => client.post(url, formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
}).then(res => res.data)

export default client
