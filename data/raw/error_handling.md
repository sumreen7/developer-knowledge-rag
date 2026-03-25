# Error Handling Guide

## HTTP Status Codes

### 400 Bad Request
The request was malformed or missing required fields.

### 401 Unauthorized
Your token is missing, expired, or invalid.
Solution: request a new token via POST /auth/token.

### 403 Forbidden
Your token is valid but you lack permission for this action.

### 404 Not Found
The resource does not exist. Double-check the ID in your URL.

### 429 Too Many Requests
You have exceeded the rate limit.
Free tier: 1000 requests/hour. Paid: 10000 requests/hour.
Wait for the rate limit window to reset or upgrade your plan.

### 500 Internal Server Error
Something went wrong on the server side.
Retry with exponential backoff.
