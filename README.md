# better-wind

Site-specific probabilistic wind forecasting for small airports. One codebase,
per-airport YAML configs, per-airport trained models. Learns the residual bias
of HRRR/TAF wind guidance instead of predicting raw wind.

See [CLAUD.md](CLAUD.md) for the full spec and phased build plan.

## Quickstart

```bash
uv sync --extra dev

# Pull data for the primary airport
uv run wind-forecast ingest-metar --airport KMAN
uv run wind-forecast ingest-hrrr  --airport KMAN --start 2022-01-01 --end 2024-12-31
```

Adding a new airport is a YAML file under `config/airports/` — no code changes.
