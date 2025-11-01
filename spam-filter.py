# spam_email_trainer_graphical.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ──────────────────────────────────────────────
# 1️⃣ Load the dataset
# ──────────────────────────────────────────────
console.rule("[bold cyan]📦 Loading Dataset[/bold cyan]")
data = pd.read_csv("spambase_csv.csv")

console.print(Panel.fit(f"[bold green]✅ Dataset Loaded Successfully![/bold green]\n"
                        f"[bold]Shape:[/bold] {data.shape}\n"
                        f"[bold]Columns:[/bold] {data.columns.tolist()[:10]} ...",
                        title="Dataset Info", border_style="green"))

# ──────────────────────────────────────────────
# 2️⃣ Split features and labels
# ──────────────────────────────────────────────
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

console.rule("[bold cyan]🧠 Training Naive Bayes Model[/bold cyan]")

# ──────────────────────────────────────────────
# 3️⃣ Train Gaussian Naive Bayes
# ──────────────────────────────────────────────
console.print("[yellow]Training Gaussian Naive Bayes model...[/yellow]")
model = GaussianNB()
model.fit(X_train, y_train)
console.print("[bold green]✅ Training Complete![/bold green]")

# ──────────────────────────────────────────────
# 4️⃣ Evaluate Model
# ──────────────────────────────────────────────
console.rule("[bold cyan]📊 Model Evaluation[/bold cyan]")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Accuracy panel
console.print(Panel.fit(f"[bold white on blue]Accuracy:[/bold white on blue] [bold yellow]{accuracy:.4f}[/bold yellow]",
                        border_style="bright_blue", title="Performance"))

# Classification report as table
table = Table(title="Classification Report", box=box.SIMPLE_HEAD)
table.add_column("Class", justify="center", style="cyan")
table.add_column("Precision", justify="center", style="magenta")
table.add_column("Recall", justify="center", style="magenta")
table.add_column("F1-Score", justify="center", style="magenta")

for cls, metrics in report.items():
    if cls not in ["accuracy", "macro avg", "weighted avg"]:
        table.add_row(cls,
                      f"{metrics['precision']:.3f}",
                      f"{metrics['recall']:.3f}",
                      f"{metrics['f1-score']:.3f}")

console.print(table)

# Confusion matrix
cm_table = Table(title="Confusion Matrix", box=box.SQUARE)
cm_table.add_column(" ", justify="center", style="bold white")
cm_table.add_column("Predicted: Not Spam (0)", justify="center", style="yellow")
cm_table.add_column("Predicted: Spam (1)", justify="center", style="yellow")

cm_table.add_row("Actual: Not Spam (0)", str(conf_matrix[0][0]), str(conf_matrix[0][1]))
cm_table.add_row("Actual: Spam (1)", str(conf_matrix[1][0]), str(conf_matrix[1][1]))
console.print(cm_table)

# ──────────────────────────────────────────────
# 5️⃣ Save Model
# ──────────────────────────────────────────────
joblib.dump(model, "spam_naivebayes_model.pkl")

console.rule("[bold cyan]💾 Saving Model[/bold cyan]")
console.print(Panel.fit("[bold green]Model saved successfully![/bold green]\n"
                        "🧠 [cyan]Model:[/cyan] spam_naivebayes_model.pkl",
                        border_style="green", title="Success"))




