# Podman Deployment Guide

This project supports deployment using Podman. Podman is a daemonless alternative to Docker, providing better security and compatibility.

## Installing Podman

### macOS
```bash
brew install podman
podman machine init
podman machine start
```

### Linux (Fedora/RHEL)
```bash
sudo dnf install podman podman-compose
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install podman podman-compose
```

### Install podman-compose via pip
```bash
pip3 install podman-compose
```

## Usage

### Method 1: Using podman-compose (Recommended)

```bash
# Start all services
podman-compose -f podman-compose.yml up -d --build

# Check service status
podman-compose -f podman-compose.yml ps

# View logs
podman-compose -f podman-compose.yml logs -f

# Stop all services
podman-compose -f podman-compose.yml down

# Stop and remove volumes
podman-compose -f podman-compose.yml down -v
```

### Method 2: Directly using docker-compose.yml

`podman-compose` can also directly read `docker-compose.yml`:

```bash
podman-compose up -d --build
```

## Service Access URLs

- **Chatbot**: http://localhost:7860
- **RAG API**: http://localhost:8001
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (Username: admin, Password: admin)

## Common Podman Commands

```bash
# List running containers
podman ps

# List all containers (including stopped)
podman ps -a

# View container logs
podman logs <container_name>

# Enter container
podman exec -it <container_name> /bin/bash

# List images
podman images

# Clean up unused resources
podman system prune -a
```

## Key Differences from Docker

1. **Daemonless mode**: Podman doesn't require a background daemon, making it more secure
2. **Rootless execution**: Can run containers without root privileges
3. **Command compatibility**: Most commands are the same as Docker (`podman` vs `docker`)
4. **Networking**: Podman uses CNI networking, but bridge driver is also supported

## Troubleshooting

### If you encounter network issues
```bash
# Check Podman networks
podman network ls
podman network inspect inf4rag-network
```

### If you encounter permission issues (Linux)
```bash
# Ensure users can use podman without sudo
# Check subuid and subgid configuration
cat /etc/subuid
cat /etc/subgid
```

### If you encounter port conflicts
```bash
# Check port usage
podman ps --format "table {{.Names}}\t{{.Ports}}"
```

## Migration Notes

If you previously used Docker and want to switch to Podman:

1. Stop Docker service (if running)
2. Use `podman-compose` instead of `docker-compose`
3. All configuration files and commands are essentially the same

Note: Docker and Podman volumes and networks are independent and need to be recreated.
