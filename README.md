<p align="center">
  <img src="assets/vap-banner.svg" alt="VAP - Neon Control Plane for vLLM Profiling" width="100%" />
</p>

VAP is a lightweight tool for deploying a vLLM service, running benchmark workloads, collecting profiler output, and viewing run logs through a simple web UI.

The project includes:

- `main.py`: runs the VAP workflow, including vLLM deployment, benchmark execution, profiling, TensorBoard, and Perfetto Trace Processor startup.
- `server.py`: starts a local configuration and control service.
- `public/index.html`: provides the browser UI for editing configs, validating resources, starting/stopping runs, viewing logs, and downloading trace archives.
- `example-config.json`: example configuration template.

## Setup

Install dependencies with the project installer:

```bash
bash install.sh
```

The installer creates `~/.vap`, bootstraps local binaries into `~/.vap/bin`, creates `~/.vap/venv`, installs VAP in editable mode, installs a `vap` wrapper to `~/.local/bin/vap`, copies the default config to `~/.vap/config.json`, downloads `~/.vap/bin/trace_processor`, and warms the Perfetto cache under `~/.vap/perfetto-home/`.

Make sure `~/.local/bin` is in `PATH` if `vap` is not found after installation.

Make sure Docker is available and the configured image, model path, devices, and mounts exist on the host.

## Start the UI

Run the local control server:

```bash
vap start
```

Open the printed local URL in your browser. The UI lets you:

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
vap run
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

## Agent Chat

The Agent panel is a Hermes-style VAP tool agent specialized for vLLM profiling. It starts by asking which model you want to profile, then guides you through model paths, Docker/GPU settings, tensor parallel size, benchmark shape, profiler options, validation, and final run approval. It uses an AMD OpenAI-compatible endpoint for reasoning, but VAP operations are exposed through explicit backend tools instead of UI automation.

Configure the subscription key before starting the server:

```bash
export VAP_LLM_SUBSCRIPTION_KEY="..."
export VAP_LLM_BASE_URL="https://llm-api.amd.com/OpenAI"
export VAP_LLM_MODEL="gpt-5.5"
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
