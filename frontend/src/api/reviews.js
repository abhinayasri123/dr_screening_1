import { get, post } from './client'

export const getPendingReviews = (skip = 0, limit = 100) => get(`/reviews/pending?skip=${skip}&limit=${limit}`)
export const getReviews = (skip = 0, limit = 100) => get(`/reviews?skip=${skip}&limit=${limit}`)
export const createReview = (data) => post('/reviews', data)
export const getReviewStats = () => get('/reviews/stats')
