# stage 1: builder
FROM python:3.12-slim-bullseye AS builder

# Set working directory
WORKDIR /app

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential gcc && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
ENV UV_COMPILE_BYTECODE=1

# Copy only the files needed for dependency installation to leverage layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the project files
COPY . .

# Sync the project itself
RUN uv sync --frozen --no-dev

# stage 2: runtime
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies (e.g., curl for healthchecks)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy necessary application files
COPY src/ /app/src/
COPY static/ /app/static/
COPY pyproject.toml /app/

# Create necessary data directories
RUN mkdir -p /app/data/uploads /app/data/files

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "src.api.main"]
