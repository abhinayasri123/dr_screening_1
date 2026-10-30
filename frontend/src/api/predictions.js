import { get, post, postForm } from './client'

export const createPrediction = (formData) => postForm('/predictions', formData)
export const getPrediction = (id) => get(`/predictions/${id}`)
export const getPatientPredictions = (patientId) => get(`/patients/${patientId}/predictions`)
export const getPredictions = (skip = 0, limit = 100) => get(`/predictions?skip=${skip}&limit=${limit}`)
