import { get, post, del } from './client'

export const getPatients = (skip = 0, limit = 100) => get(`/patients?skip=${skip}&limit=${limit}`)
export const getPatient = (id) => get(`/patients/${id}`)
export const createPatient = (data) => post('/patients', data)
export const deletePatient = (id) => del(`/patients/${id}`)
export const getPatientEHR = (id) => get(`/patients/${id}/ehr`)
export const addEHRRecord = (patientId, data) => post(`/patients/${patientId}/ehr`, data)
