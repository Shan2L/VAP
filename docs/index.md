# VAP Documentation

VAP is a local operations tool for vLLM deployment, benchmarking, profiler data collection, and trace visualization.

## Documentation

### [Deployment, Operations, and Technical Report](deployment.md)

For deployment engineers, platform operators, and developers who need to understand the system architecture. It covers:

- Environment requirements and installation methods;
- Web UI, CLI, SSH tunnel, and systemd deployment;
- Configuration files, logs, and runtime artifacts;
- Technical architecture, execution flow, and security design;
- Troubleshooting and production-readiness checklist.

### [User Guide](user-guide.md)

For day-to-day users. It explains configuration fields, Web UI operations, CLI commands, Agent usage, trace analysis, and common troubleshooting steps.

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/Shan2L/VAP/main/bootstrap.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vap start
```

Open the token URL printed in the terminal to enter the Web UI.

!!! warning "Security Notice"

    VAP currently uses host networking, ROCm devices, and elevated Docker capabilities.
    Run only trusted images, models, and configurations, and keep the Web service bound to `127.0.0.1`.

## Project Links

- [GitHub Repository](https://github.com/Shan2L/VAP)
- [Deployment Documentation](deployment.md)
- [User Guide](user-guide.md)
