# MDFactory Docs

This folder contains the Fumadocs + Next.js application that renders the MDFactory
documentation. The site mixes hand-written guides with API pages generated straight from the
Python package.

## Quick start with pixi

The root `pyproject.toml` defines a `docs` environment that installs Bun, the JS
dependencies, and `fumapy-generate` automatically:

```bash
pixi run -e docs docs-dev      # dev server with hot-reload
pixi run -e docs docs-build    # full static build to docs/out/
pixi run -e docs docs-generate # regenerate API docs only
```

Each command chains through `docs-install` → `docs-fumapy` automatically, so a
single command handles all setup. On subsequent runs the dependency steps are
fast (no-op when already satisfied).

## Manual setup (without pixi)

### Prerequisites

- [Bun](https://bun.sh/) 1.0+ (fast JavaScript runtime and package manager)
- Python 3.11 (to install `mdfactory` and `fumapy`)

Install the tooling once:

```bash
cd docs
bun install
python3.11 -m pip install -e ..                         # expose the local package
python3.11 -m pip install ./node_modules/fumadocs-python
```

### Commands

```bash
bun run docs:generate  # runs fumapy-generate + converts JSON into MDX
bun run dev            # regenerates the docs and starts Next.js locally
bun run build          # regenerates the docs and produces a production build
```

The generator writes JSON snapshots into `docs/mdfactory-api` and MDX files into
`docs/content/docs/(api)`. Both directories are ignored in Git because they are re-created on
every run.

## Static export & GitHub Pages

- `next.config.mjs` uses Next.js' `output: 'export'`, so `bun run build` emits the fully static site into `docs/out`.
- Set `GITHUB_PAGES=true` before running `bun run build` when you want to mirror the GitHub Pages base path locally (the workflow does this automatically).
- Set `NEXT_PUBLIC_BASE_PATH` when you need a prefix (e.g., `"/mdfactory"` for project pages). Use `__root__` for root deployments.
- The `.github/workflows/docs.yml` pipeline installs Python + Bun, regenerates the MDX files, runs `bun run build`, and publishes `docs/out` to GitHub Pages whenever `main` is updated or the workflow is triggered manually.
- You can preview the generated site locally after a build with any static server, e.g. `bunx serve docs/out` or `python -m http.server --directory docs/out`.
  - With `GITHUB_PAGES=true`, open the preview at `http://localhost:8000/<repo>/` so asset paths match the base path.

## Structure

- `content/docs` – manually written documentation entries.
- `content/docs/(api)` – generated MDX files (do not edit manually).
- `scripts/generate-docs.mjs` – orchestrates `fumapy-generate` + MDX conversion.
- `src/` – Next.js app, layouts, components, and styles.

To learn more about the UI layer itself, read the upstream [Fumadocs
documentation](https://fumadocs.dev/docs/ui).
