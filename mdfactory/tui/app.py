# ABOUTME: Main Textual application for interactive simulation browsing
# ABOUTME: Provides DataTable with live filtering by type, status, tags, hash, and SMILES
"""Main Textual application for interactive simulation browsing."""

from __future__ import annotations

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Label, Select, Static

from mdfactory.analysis.constants import STATUS_ORDER
from mdfactory.analysis.store import SimulationStore
from mdfactory.models.input import type_mapping

# Options for Select widgets (include "all" sentinel)
TYPE_OPTIONS: list[tuple[str, str | None]] = [("All", None)] + [(t, t) for t in type_mapping.keys()]
STATUS_OPTIONS: list[tuple[str, str | None]] = [("All", None)] + [(s, s) for s in STATUS_ORDER]


class SimulationDetail(Static):
    """Widget showing details of the selected simulation."""

    def update_detail(self, row_data: dict | None, store: SimulationStore | None = None) -> None:
        """Update the detail view with simulation data.

        Parameters
        ----------
        row_data : dict | None
            Dictionary with simulation details, or None to clear.
        store : SimulationStore | None
            Store to fetch full BuildInput metadata from.

        """
        if row_data is None:
            self.update("")
            return

        lines = []
        lines.append(f"[bold cyan]Hash:[/] {row_data.get('hash', 'N/A')}")
        lines.append(f"[bold green]Type:[/] {row_data.get('simulation_type', 'N/A')}")
        lines.append(f"[bold magenta]Status:[/] {row_data.get('status', 'N/A')}")
        lines.append(f"[bold]Path:[/] {row_data.get('path', 'N/A')}")

        tags = row_data.get("tags")
        if tags:
            tags_str = ", ".join(f"{k}={v}" for k, v in tags.items())
            lines.append(f"[bold yellow]Tags:[/] {tags_str}")
        else:
            lines.append("[bold yellow]Tags:[/] (none)")

        # Fetch full metadata from store
        sim_hash = row_data.get("hash")
        if store and sim_hash:
            try:
                sim = store.get_simulation(sim_hash)
                bi = sim.build_input
                meta = bi.metadata

                lines.append("")
                lines.append("[bold underline]Composition[/]")
                lines.append(f"  Total count: {meta.get('total_count', 'N/A')}")
                lines.append(f"  Parametrization: {meta.get('parametrization', 'N/A')}")
                lines.append(f"  Engine: {meta.get('engine', 'N/A')}")

                species = meta.get("species_composition", [])
                if species:
                    lines.append("")
                    lines.append("  [bold]Species:[/]")
                    for sp in species:
                        resname = sp.get("resname", "?")
                        count = sp.get("count", "?")
                        fraction = sp.get("fraction", 0)
                        smiles = getattr(
                            next(
                                (s for s in bi.system.species if s.resname == resname),
                                None,
                            ),
                            "smiles",
                            None,
                        )
                        smiles_str = f"  {smiles}" if smiles else ""
                        lines.append(f"    {resname}: {count} ({fraction:.1%}){smiles_str}")

                system_specific = meta.get("system_specific", {})
                if system_specific:
                    lines.append("")
                    lines.append("  [bold]Parameters:[/]")
                    for key, val in system_specific.items():
                        if isinstance(val, dict):
                            lines.append(f"    {key}:")
                            for k, v in val.items():
                                lines.append(f"      {k}: {v}")
                        else:
                            lines.append(f"    {key}: {val}")

            except (ValueError, KeyError):
                pass

        self.update("\n".join(lines))


class SimulationBrowser(App):
    """Interactive TUI for browsing and filtering simulations."""

    TITLE = "MDFactory Simulation Browser"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("escape", "clear_filters", "Clear filters"),
    ]

    CSS = """
    #filters {
        height: 6;
        dock: top;
        padding: 0 1;
    }

    #filter-row-1, #filter-row-2 {
        height: 3;
        width: 1fr;
    }

    #filter-row-1 Select {
        width: 1fr;
        margin: 0 1;
    }

    #filter-row-2 Input {
        width: 1fr;
        margin: 0 1;
    }

    Label {
        width: auto;
        padding: 0 1;
        content-align: center middle;
    }

    #main-content {
        height: 1fr;
    }

    #table-container {
        width: 3fr;
        height: 1fr;
    }

    #detail-panel {
        width: 1fr;
        min-width: 40;
        height: 1fr;
        border-left: solid $accent;
        padding: 1;
    }

    DataTable {
        height: 1fr;
    }
    """

    def __init__(
        self,
        store: SimulationStore,
        **kwargs,
    ) -> None:
        """Initialize the browser with a SimulationStore.

        Parameters
        ----------
        store : SimulationStore
            Pre-configured store instance for simulation discovery.

        """
        super().__init__(**kwargs)
        self.store = store
        self._results_cache: list[dict] = []

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        with Vertical(id="filters"):
            with Horizontal(id="filter-row-1"):
                yield Label("Type:")
                yield Select(TYPE_OPTIONS, value=None, id="filter-type", allow_blank=False)
                yield Label("Status:")
                yield Select(STATUS_OPTIONS, value=None, id="filter-status", allow_blank=False)
            with Horizontal(id="filter-row-2"):
                yield Label("Hash:")
                yield Input(placeholder="Hash prefix", id="filter-hash")
                yield Label("Tags:")
                yield Input(placeholder="k=v, k=v", id="filter-tags")
                yield Label("SMILES:")
                yield Input(placeholder="Substructure", id="filter-smiles")
        with Horizontal(id="main-content"):
            with Vertical(id="table-container"):
                yield DataTable(id="results-table")
            yield SimulationDetail(id="detail-panel")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the table and run initial search."""
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Hash", "Type", "Status", "Tags", "Path")
        table.cursor_type = "row"
        self._run_search()

    @on(Select.Changed)
    def on_select_changed(self, event: Select.Changed) -> None:
        """Re-run search when a select widget changes."""
        self._run_search()

    @on(Input.Changed)
    def on_filter_changed(self, event: Input.Changed) -> None:
        """Re-run search when any filter input changes."""
        self._run_search()

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update detail panel when a row is highlighted."""
        detail = self.query_one("#detail-panel", SimulationDetail)
        if event.cursor_row is not None and event.cursor_row < len(self._results_cache):
            detail.update_detail(self._results_cache[event.cursor_row], store=self.store)
        else:
            detail.update_detail(None)

    def action_refresh(self) -> None:
        """Refresh discovery and re-run search."""
        self.store.discover(refresh=True)
        self._run_search()

    def action_clear_filters(self) -> None:
        """Clear all filter inputs and reset selects."""
        for input_widget in self.query(Input):
            input_widget.value = ""
        self.query_one("#filter-type", Select).value = None
        self.query_one("#filter-status", Select).value = None

    @work(thread=True, exclusive=True)
    def _run_search(self) -> None:
        """Run search with current filter values and update table."""
        # Gather filter values from selects
        type_val = self.query_one("#filter-type", Select).value
        status_val = self.query_one("#filter-status", Select).value

        # Gather filter values from inputs
        hash_val = self.query_one("#filter-hash", Input).value.strip() or None
        tags_str = self.query_one("#filter-tags", Input).value.strip()
        smiles_val = self.query_one("#filter-smiles", Input).value.strip() or None

        # Parse tags
        tags_filter = None
        if tags_str:
            tags_filter = {}
            for raw_part in tags_str.split(","):
                token = raw_part.strip()
                if "=" in token:
                    k, v = token.split("=", 1)
                    tags_filter[k.strip()] = v.strip()

        # Run search (Select value is None for "All")
        try:
            results = self.store.search(
                simulation_type=type_val,
                status=status_val,
                hash_prefix=hash_val,
                tags=tags_filter,
                smiles=smiles_val,
            )
        except (ValueError, ImportError) as e:
            self.notify(str(e), severity="error")
            return

        # Convert to list of dicts for caching
        self._results_cache = results.to_dict("records")

        # Update table on main thread
        self.call_from_thread(self._update_table)

    def _update_table(self) -> None:
        """Update the DataTable with cached results."""
        table = self.query_one("#results-table", DataTable)
        table.clear()

        for row_data in self._results_cache:
            tags = row_data.get("tags")
            tags_str = ""
            if tags:
                tags_str = ", ".join(f"{k}={v}" for k, v in tags.items())

            table.add_row(
                str(row_data["hash"])[:12],
                str(row_data.get("simulation_type", "")),
                str(row_data.get("status", "")),
                tags_str,
                str(row_data.get("path", "")),
            )

        # Update title with count
        self.sub_title = f"{len(self._results_cache)} simulation(s)"
