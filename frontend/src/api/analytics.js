import { get, post } from './client'

export const getDashboardStats = () => get('/analytics/dashboard-stats')
export const getResearchMetrics = () => get('/analytics/research-metrics')
export const triggerEvaluation = () => post('/analytics/evaluate')
