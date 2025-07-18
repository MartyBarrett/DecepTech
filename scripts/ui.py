import typer
from rich.console import Console
from rich.table import Table
import os
import pandas as pd

app = typer.Typer()
console = Console()

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'csv', 'utterances.csv')

def load_utterances():
    if not os.path.exists(CSV_PATH):
        console.print("[bold red]CSV file not found. Run the parsing script first.[/bold red]")
        raise typer.Exit()
    return pd.read_csv(CSV_PATH)

@app.command()
def main():
    """Main menu UI for DecepTech."""
    while True:
        console.rule("[bold blue]DecepTech Terminal UI")
        console.print("\nSelect an option:")
        console.print("[1] View sample utterances")
        console.print("[2] Show model accuracy")
        console.print("[3] Exit\n")

        choice = console.input("[bold green]Enter your choice (1-3)[/bold green]: ").strip()

        if choice == "1":
            show_utterances()
        elif choice == "2":
            show_accuracy()
        elif choice == "3":
            console.print("Goodbye!")
            break
        else:
            console.print("[red]Invalid input. Please enter 1, 2, or 3.[/red]")

def show_utterances(n: int = 5):
    """Display N sample utterances."""
    try:
        df = load_utterances()
        table = Table(title="Sample Utterances")

        table.add_column("Video", style="cyan")
        table.add_column("Start", style="green")
        table.add_column("End", style="green")
        table.add_column("Label", style="magenta")
        table.add_column("Transcript", style="yellow")

        for _, row in df.head(n).iterrows():
            table.add_row(
                str(row['video_file']),
                str(row['start_time_ms']),
                str(row['end_time_ms']),
                str(row['veracity']),
                str(row['transcript']) if pd.notna(row['transcript']) else ""
            )
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error displaying utterances:[/red] {e}")

def show_accuracy():
    """Display a placeholder model accuracy."""
    # You can later replace this with a real metrics file
    accuracy = 0.834  # Example
    console.print(f"[bold green]Model Accuracy:[/bold green] {accuracy * 100:.2f}% (placeholder)")

if __name__ == "__main__":
    app()
