# src/utils.py

import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

# This is a forward declaration. The actual class is in main.py
class ElitePredictionSystem:
    pass

def create_status_table(system: 'ElitePredictionSystem') -> Table:
    """Creates a table displaying the system's current status."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Value")

    # --- Data Engine ---
    table.add_row("[bold]Data Engine[/bold]", "")
    # In a real system, you'd get these from the data_engine instance
    table.add_row("  - Connection Status", "[green]Connected[/green]")
    table.add_row("  - Ticks per Second", "1,234,567")
    table.add_row("  - Data Quality", "0.98")

    # --- Prediction ---
    table.add_row("[bold]Last Prediction[/bold]", "")
    if hasattr(system, 'last_prediction') and system.last_prediction:
        pred = system.last_prediction
        table.add_row("  - Expected Return", f"{pred.expected_return:.5f}")
        table.add_row("  - Volatility", f"{pred.volatility:.5f}")
        table.add_row("  - Sharpe Prediction", f"{pred.sharpe_prediction:.2f}")
        table.add_row("  - Optimal Position", f"{pred.optimal_position:.2%}")
    else:
        table.add_row("  - Status", "Initializing...")

    # --- Meta-Controller ---
    table.add_row("[bold]Meta-Controller[/bold]", "")
    regime = getattr(system, 'current_regime', 'Detecting...')
    table.add_row("  - Current Regime", regime)

    # --- Online Learning ---
    table.add_row("[bold]Online Learning[/bold]", "")
    updates = getattr(system, 'online_updates', 0)
    table.add_row("  - Total Updates", str(updates))

    return table

def create_log_panel(system: 'ElitePredictionSystem') -> Panel:
    """Creates a panel to display recent log messages."""
    log_messages = getattr(system, 'log_messages', ["Initializing..."])
    log_text = "\n".join(log_messages)
    return Panel(Text(log_text, style="white"), title="Logs", border_style="blue")

def display_system_status(system: 'ElitePredictionSystem'):
    """
    Creates and displays a real-time terminal dashboard.
    """
    console = Console()
    layout = Layout()

    layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="main")
    )

    layout["main"].split_row(
        Layout(name="status"),
        Layout(name="logs")
    )

    layout["header"].update(
        Panel(Text("🚀 Elite ML Prediction System 🚀", justify="center", style="bold green"),
        border_style="green")
    )

    with Live(layout, screen=True, redirect_stderr=False, refresh_per_second=2) as live:
        while True:
            layout["status"].update(
                Panel(create_status_table(system), title="System Status", border_style="green")
            )
            layout["logs"].update(create_log_panel(system))
            time.sleep(0.5)
