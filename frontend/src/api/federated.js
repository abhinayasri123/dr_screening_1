import { get, post } from './client'

export const getFLStatus = () => get('/federated/status')
export const startFLSimulation = (data) => post('/federated/start', data)
export const getFLResults = () => get('/federated/results')
export const getFLHistory = () => get('/federated/history')
export const getFLComparison = () => get('/federated/comparison')
