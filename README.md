<p align="center">
  <img src="assets/vap-banner.svg" alt="VAP - Neon Control Plane for vLLM Profiling" width="100%" />
</p>

VAP is a lightweight tool for deploying a vLLM service, running benchmark workloads, collecting profiler output, and viewing run logs through a simple web UI.

For day-to-day operation, see the [User Guide](USER_GUIDE.md). For a production-oriented installation guide, deployment topology, operating procedures, and architecture report, see [VAP Deployment and Technical Report](DEPLOYMENT_AND_TECHNICAL_REPORT.md).

The project includes:

- `main.py`: runs the VAP workflow, including vLLM deployment, benchmark execution, profiling, TensorBoard, and Perfetto Trace Processor startup.
- `server.py`: starts a local configuration and control service.
- `public/index.html`: provides the browser UI for editing configs, validating resources, starting/stopping runs, viewing logs, and downloading trace archives.
- `example-config.json`: example configuration template.

## Setup

Install or update VAP directly from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/Shan2L/VAP/main/bootstrap.sh | bash
```

The bootstrap installer keeps a managed source checkout at `~/.local/share/vap/source` and then runs the verified project installer. Set `VAP_REF` to install a specific branch or tag.

### One-command remote deployment

Replace `{user}` and `{hostname}`, then run this command on your local machine:

```bash
ssh -t -L 8899:127.0.0.1:8899 {user}@{hostname} \
  'curl -fsSL https://raw.githubusercontent.com/Shan2L/VAP/main/bootstrap.sh | bash && exec ~/.local/bin/vap start --host 127.0.0.1'
```

The command installs VAP and starts it on the remote host while forwarding remote
`127.0.0.1:8899` to local `127.0.0.1:8899`. Open the tokenized URL printed in
the SSH session in your local browser. Keep the SSH session open; `Ctrl-C`
stops VAP and closes the tunnel.

If you already have a source checkout, run the project installer directly:

```bash
bash install.sh
```

The installer creates `~/.vap`, installs pinned and checksum-verified `uv` and Perfetto binaries into `~/.vap/bin`, creates `~/.vap/venv`, installs VAP and its dependencies in editable mode, verifies the `vap` command, installs a wrapper at `~/.local/bin/vap`, copies the default config to `~/.vap/config.json`, and warms the Perfetto cache under `~/.vap/perfetto-home/`.

The bundled installer currently supports Linux x86_64 only. It fails closed on unknown platforms instead of executing an unverified remote installer. Install `uv` and Perfetto manually when using another platform.

Make sure `~/.local/bin` is in `PATH` if `vap` is not found after installation.

Make sure Docker is available and the configured image, model path, devices, and mounts exist on the host.

## Commands

The installer prints this command summary when it finishes:

- `vap start [--host HOST] [--port PORT]` starts the Web UI and local control server. The default host is `0.0.0.0`.
- `vap run [--config FILE] [--visualization-host HOST]` runs deployment, benchmark, profiling, and visualization directly from the CLI.
- `vap clean [--logs-dir DIR]` removes generated logs from VAP's managed logs directory.
- `vap uninstall [--purge] [--remove-source] [--yes]` removes the managed installation. Config and logs are preserved unless `--purge` is used.
- `vap --help` lists commands. Run `vap <command> --help` for command-specific options.

## Uninstall

Stop VAP, then run:

```bash
vap uninstall
```

The default uninstall removes the managed `~/.local/bin/vap` wrapper, VAP executables, its virtual environment, downloaded tools, and caches while preserving `~/.vap/config.json` and `~/.vap/logs/`.

To remove all VAP runtime data and the managed bootstrap source checkout:

```bash
vap uninstall --purge --remove-source
```

Use `--yes` for non-interactive environments. Custom `VAP_HOME` and `VAP_SOURCE_DIR` values should be supplied to the uninstaller in the same way they were supplied during installation.
From a source checkout, `bash uninstall.sh` remains available as a fallback.

## Start the UI

Run the local control server:

```bash
vap start
```

By default, VAP listens on `0.0.0.0:8899`, so trusted machines on the same
network can connect through the server IP or hostname. Use
`vap start --host 127.0.0.1` to restrict access to the local machine.

The server prints local, hostname, and IP candidates with the same per-process
startup token:

```text
VAP session URL candidates:
  Local: http://127.0.0.1:8899/?token=...
  Network candidate: http://vap-host:8899/?token=...
  Network candidate: http://10.0.0.8:8899/?token=...
```

The first request stores the session in a `SameSite=Strict` cookie and removes the token from the visible browser URL. Direct API clients must send the token in `X-VAP-Token`; state-changing requests must also use JSON and a same-origin `Origin` header when one is present.

Hostname/IP entries are candidates, not a reachability guarantee. Remote access
also requires working DNS or routing and a firewall rule allowing `8899/tcp`.

The UI lets you:

- edit VAP configuration values;
- validate ports, model paths, Docker image, devices, mounts, and config structure;
- chat with the Agent panel after an LLM key is configured or validated;
- start or stop a VAP run after validation;
- view current run logs;
- open TensorBoard after it starts successfully;
- open Perfetto UI after the trace processor starts on port `9001`;
- download the current run's `vllm-profile` files as a zip archive.

## Run from CLI

You can also run VAP directly with a config file:

```bash
vap run --config ~/.vap/config.json --visualization-host 127.0.0.1
```

Run outputs are written under `~/.vap/logs/`.

To remove generated logs:

```bash
vap clean
```

## Configuration Notes

The web UI does not overwrite the original config when starting a run. It sends the current form data to the backend, which creates a temporary config file under `~/.vap/tmp/configs/` for that run.

The deploy and benchmark `--host` / `--port` values should stay consistent. The UI keeps these fields synchronized automatically.

`profiler_cfg.tensorboard_port` controls TensorBoard. Perfetto Trace Processor is fixed to local port `9001` so `https://ui.perfetto.dev/` can discover it through the standard local endpoint.

Configuration is strict: unknown fields, invalid ports, mismatched deploy/benchmark endpoints, and unsafe CLI values are rejected by both the UI and CLI. Distributed execution is not implemented yet; a present `distributed_cfg` produces a warning and VAP continues in local mode.

VAP safely quotes deploy, benchmark, and profiler arguments before invoking the fixed shell wrapper used for log redirection.

## Security Warnings

VAP currently keeps the existing profiling-compatible Docker behavior:

- host networking;
- `SYS_ADMIN` and `SYS_PTRACE` capabilities;
- `seccomp=unconfined`;
- ROCm device mounts, including `/dev/kfd` and `/dev/mem`.

The UI and CLI display a warning instead of blocking these settings. Run only trusted images, models, configs, and model repositories.

The Web UI also listens on all network interfaces by default. It requires the
session token but does not provide TLS; use a firewall or explicitly pass
`--host 127.0.0.1` on untrusted networks.

The example config still includes `--trust-remote-code` for compatibility. This allows model repository code to execute inside the elevated VAP container. Remove the option unless the model source is trusted.

Runtime config and temporary config files are stored with private permissions. Temporary run configs older than seven days are removed when a new temporary config is created.

## Trace Fusion Attribution

Multi-rank PyTorch JSON trace fusion is built into VAP and uses only the Python standard library. Its implementation is adapted from AMD-AGI TraceLens `TraceFuse` at commit `2eae9b056b3db46656bda030499ec6d4e1310ea4`, under the MIT License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Agent Chat

The Agent panel is a Hermes-style VAP tool agent specialized for vLLM profiling. It starts by asking which model you want to profile, then guides you through model paths, Docker/GPU settings, tensor parallel size, benchmark shape, profiler options, validation, and final run approval. It uses an AMD OpenAI-compatible endpoint for reasoning, but VAP operations are exposed through explicit backend tools instead of UI automation.

Configure the subscription key before starting the server:

```bash
export VAP_LLM_SUBSCRIPTION_KEY="..."
export VAP_LLM_BASE_URL="https://llm-api.amd.com/OpenAI"
export VAP_LLM_MODEL="gpt-5.6-sol"
export VAP_AGENT_MAX_TOOL_ROUNDS=8
vap start
```

If `VAP_LLM_SUBSCRIPTION_KEY` is not set, the Agent tab shows a centered unlock form. Enter the subscription key there; the backend validates it with the LLM API and stores it only in the current server process memory.

The agent can call read-only and safe tools for config, run status, logs, validation, port checks, and resource checks. Actions that affect running processes, such as `start_run` and `stop_run`, require explicit approval in the Agent panel before the backend executes them.

The VAP tool registry is kept separate from the LLM client so the same tools can later be exposed through MCP or a Hermes Agent bridge.

## Generated Files

Runtime state is generated under `~/.vap/`:

- `~/.vap/config.json`
- `~/.vap/logs/`
- `~/.vap/tmp/configs/`
- `~/.vap/bin/`
- `~/.vap/perfetto-home/`

These files are run artifacts and can be deleted when they are no longer needed.

`vap clean` only removes the configured VAP log directory and refuses arbitrary filesystem paths.
