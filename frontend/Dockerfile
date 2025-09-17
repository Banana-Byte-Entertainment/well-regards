# Multi-stage build
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install ALL dependencies (including devDependencies needed for build)
# RUN npm ci

# Copy source code
COPY . .

# # Build the app
# RUN npm run build

# Production stage
# FROM nginx:alpine

# # Copy built app from build stage
# COPY --from=build /app/build /usr/share/nginx/html

# # Copy custom nginx config
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# # Expose port
# EXPOSE 80

# CMD ["nginx", "-g", "daemon off;"]