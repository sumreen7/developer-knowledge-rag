# Authentication Guide

## Overview
This API uses Bearer token authentication. All requests must include a valid token in the Authorization header.

## Getting a Token
Send a POST request to /auth/token with your credentials:
  username: your_username
  password: your_password

The response includes an access_token valid for 24 hours.

## Using the Token
Include the token in every request:
  Authorization: Bearer <your_access_token>

## Error Codes
- 401 Unauthorized: token missing or invalid
- 403 Forbidden: valid token but insufficient permissions
