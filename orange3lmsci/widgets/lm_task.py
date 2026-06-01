"""
Orange Data Mining Widget: LM Task
Interacts with a local Ollama server.

Features:
- Write prompts with {field} placeholders replaced by table columns
- Connect an input table to iterate over rows
- Output as text or augmented table (with 'llm' column)
- Standalone mode: [Query] button and [Query on load] checkbox
"""

import re
import json
import threading
import urllib.request
import urllib.error
from importlib.resources import files

from AnyQt.QtWidgets import (
    QPlainTextEdit, QComboBox, QCheckBox, QPushButton,
    QLabel, QSizePolicy, QHBoxLayout, QVBoxLayout,
    QGroupBox, QProgressBar, QSplitter, QTextEdit,
    QWidget, QFormLayout, QLineEdit
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal, QObject
from AnyQt.QtGui import QFont

import numpy as np

from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, StringVariable
from Orange.data.io import guess_data_type


OLLAMA_DEFAULT_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class OllamaWorker(QObject):
    """Run Ollama queries in a background thread."""
    progress = pyqtSignal(int, int)          # current, total
    result_text = pyqtSignal(str)            # single text result
    result_row = pyqtSignal(int, str)        # row index + result
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, url, model, prompts, mode="text"):
        super().__init__()
        self._url = url.rstrip("/")
        self._model = model
        self._prompts = prompts   # list of (index, prompt_str)
        self._mode = mode
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self._prompts)
        for i, (row_idx, prompt) in enumerate(self._prompts):
            if self._cancelled:
                break
            try:
                response = self._query(prompt)
            except Exception as e:
                self.error.emit(str(e))
                self.finished.emit()
                return

            self.progress.emit(i + 1, total)

            if self._mode == "text":
                self.result_text.emit(response)
            else:
                self.result_row.emit(row_idx, response)

        self.finished.emit()

    def _query(self, prompt):
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False
        }).encode()

        req = urllib.request.Request(
            f"{self._url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data.get("response", "")


def fetch_ollama_models(url):
    """Return list of model names from Ollama server, or empty list on error."""
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def build_prompt(template, row_dict):
    """Replace {field} placeholders with values from row_dict."""
    def replacer(match):
        key = match.group(1)
        return str(row_dict.get(key, match.group(0)))
    return re.sub(r"\{(\w+)\}", replacer, template)


def extract_fields(template):
    """Return list of {field} names found in the template."""
    return re.findall(r"\{(\w+)\}", template)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class OWLMTask(OWWidget):
    name = "LM Task"
    description = "Send prompts to a local Ollama server, optionally using input table columns as template fields."
    icon = str(files("orange3lmsci") / "icons" / "LMTask.svg")
    priority = 100
    category = "LMSci"
    keywords = ["llm", "ollama", "prompt", "language model"]

    want_main_area = True

    # Inputs / Outputs
    class Inputs:
        data = Input("Data", Table, auto_summary=False)

    class Outputs:
        data = Output("Data", Table, auto_summary=False)
        text = Output("Text", str, auto_summary=False)

    # Persistent settings
    ollama_url: str = Setting(OLLAMA_DEFAULT_URL)
    selected_model: str = Setting("")
    prompt_template: str = Setting("")
    output_mode: str = Setting("text")   # "text" or "table"
    query_on_load: bool = Setting(False)

    def __init__(self):
        super().__init__()
        self._input_data = None
        self._worker = None
        self._thread = None
        self._row_results = {}   # row_idx -> str

        self._build_control_area()
        self._build_main_area()

        # Refresh model list on startup (non-blocking)
        threading.Thread(target=self._refresh_models_bg, daemon=True).start()

        if self.query_on_load and not self._input_data:
            # Defer until event loop is running
            from AnyQt.QtCore import QTimer
            QTimer.singleShot(500, self._run_query)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_control_area(self):
        ca = self.controlArea

        # --- Ollama server ---
        server_box = gui.vBox(ca, "Ollama Server")

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("URL:"))
        self._url_edit = QLineEdit(self.ollama_url)
        self._url_edit.setPlaceholderText("http://localhost:11434")
        self._url_edit.editingFinished.connect(self._on_url_changed)
        url_row.addWidget(self._url_edit)
        server_box.layout().addLayout(url_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo)
        self._refresh_btn = QPushButton("↺")
        self._refresh_btn.setFixedWidth(30)
        self._refresh_btn.setToolTip("Refresh model list")
        self._refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self._refresh_btn)
        server_box.layout().addLayout(model_row)

        # --- Output mode ---
        out_box = gui.vBox(ca, "Output")
        gui.radioButtonsInBox(
            out_box, self, "output_mode",
            btnLabels=["Text", "Table (adds 'llm' column)"],
            callback=self._on_output_mode_changed
        )
        # Map radio index to string value
        self._output_mode_values = ["text", "table"]

        # --- Standalone controls ---
        standalone_box = gui.vBox(ca, "Standalone Mode")
        self._query_btn = gui.button(
            standalone_box, self, "Query", callback=self._run_query
        )
        self._qol_checkbox = gui.checkBox(
            standalone_box, self, "query_on_load", "Query on load"
        )

        # --- Progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        ca.layout().addWidget(self._progress_bar)

        self._cancel_btn = gui.button(ca, self, "Cancel", callback=self._cancel_query)
        self._cancel_btn.setVisible(False)

        gui.rubber(ca)

    def _build_main_area(self):
        ma = self.mainArea

        splitter = QSplitter(Qt.Vertical)
        ma.layout().addWidget(splitter)

        # --- Prompt editor ---
        prompt_group = QGroupBox("Prompt Template")
        prompt_layout = QVBoxLayout()
        prompt_group.setLayout(prompt_layout)

        hint = QLabel(
            "Use {column_name} as placeholders for input table columns."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        prompt_layout.addWidget(hint)

        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setPlaceholderText(
            "e.g. Summarize the following text:\n\n{text}"
        )
        self._prompt_edit.setPlainText(self.prompt_template)
        self._prompt_edit.setFont(QFont("Courier New", 10))
        self._prompt_edit.textChanged.connect(self._on_prompt_changed)
        prompt_layout.addWidget(self._prompt_edit)

        self._fields_label = QLabel()
        self._fields_label.setStyleSheet("color: #555; font-size: 11px;")
        prompt_layout.addWidget(self._fields_label)
        splitter.addWidget(prompt_group)

        # --- Output viewer ---
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)

        self._output_view = QTextEdit()
        self._output_view.setReadOnly(True)
        self._output_view.setFont(QFont("Courier New", 10))
        output_layout.addWidget(self._output_view)
        splitter.addWidget(output_group)

        splitter.setSizes([300, 200])
        self._update_fields_label()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @Inputs.data
    def set_data(self, data):
        self._input_data = data
        has_input = data is not None

        # Disable standalone controls when table is connected
        self._query_btn.setEnabled(not has_input)
        self._qol_checkbox.setEnabled(not has_input)

        self._update_fields_label()

        if has_input:
            self._output_view.clear()
            self._run_query()

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    # Orange radioButtonsInBox stores 0/1 in output_mode as int when using
    # the attribute name. We store "text"/"table" as the setting string but
    # use an index internally. Override with property tricks via callbacks.

    def _get_output_mode_idx(self):
        try:
            return self._output_mode_values.index(self.output_mode)
        except ValueError:
            return 0

    def _on_output_mode_changed(self):
        # The radioButtons widget sets self.output_mode to the index (int)
        # when callback fires; convert back to string.
        idx = self.output_mode if isinstance(self.output_mode, int) else self._get_output_mode_idx()
        if isinstance(idx, int) and 0 <= idx < len(self._output_mode_values):
            self.output_mode = self._output_mode_values[idx]

    def _on_url_changed(self):
        self.ollama_url = self._url_edit.text().strip() or OLLAMA_DEFAULT_URL
        self._refresh_models()

    def _on_model_changed(self, text):
        self.selected_model = text

    def _on_prompt_changed(self):
        self.prompt_template = self._prompt_edit.toPlainText()
        self._update_fields_label()

    # ------------------------------------------------------------------
    # Model list
    # ------------------------------------------------------------------

    def _refresh_models(self):
        self._refresh_btn.setEnabled(False)
        threading.Thread(target=self._refresh_models_bg, daemon=True).start()

    def _refresh_models_bg(self):
        models = fetch_ollama_models(self.ollama_url)
        # Update UI on main thread
        from AnyQt.QtCore import QMetaObject, Q_ARG
        QMetaObject.invokeMethod(self, "_update_model_list",
                                 Qt.QueuedConnection,
                                 Q_ARG("PyQt_PyObject", models))

    def _update_model_list(self, models):
        current = self._model_combo.currentText() or self.selected_model
        self._model_combo.clear()
        if models:
            self._model_combo.addItems(models)
            if current in models:
                self._model_combo.setCurrentText(current)
            elif self.selected_model in models:
                self._model_combo.setCurrentText(self.selected_model)
        else:
            # Server unreachable; allow manual entry
            if self.selected_model:
                self._model_combo.addItem(self.selected_model)
                self._model_combo.setCurrentText(self.selected_model)
        self._refresh_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Fields label
    # ------------------------------------------------------------------

    def _update_fields_label(self):
        fields = extract_fields(self.prompt_template)
        if fields:
            self._fields_label.setText(f"Fields detected: {', '.join(f'{{{f}}}' for f in fields)}")
        else:
            self._fields_label.setText("No {field} placeholders detected.")

        # Warn about missing columns if table is connected
        if self._input_data is not None and fields:
            col_names = [v.name for v in self._input_data.domain.variables
                         + self._input_data.domain.metas]
            missing = [f for f in fields if f not in col_names]
            if missing:
                self._fields_label.setText(
                    self._fields_label.text() +
                    f"  ⚠ Missing columns: {', '.join(missing)}"
                )

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def _run_query(self):
        template = self._prompt_edit.toPlainText().strip()
        if not template:
            self.warning("Please enter a prompt template.")
            return
        self.warning()

        model = self._model_combo.currentText().strip()
        if not model:
            self.error("Please select or enter a model name.")
            return
        self.error()

        self._output_view.clear()
        self._row_results = {}

        if self._input_data is not None:
            prompts = self._build_table_prompts(template)
        else:
            # Standalone: single query with template as-is
            prompts = [(0, template)]

        self._start_worker(prompts, model)

    def _build_table_prompts(self, template):
        data = self._input_data
        domain = data.domain
        all_vars = list(domain.variables) + list(domain.metas)

        prompts = []
        for i, row in enumerate(data):
            row_dict = {}
            for var in all_vars:
                try:
                    val = row[var]
                    row_dict[var.name] = var.repr_val(val) if hasattr(var, "repr_val") else str(val)
                except Exception:
                    row_dict[var.name] = ""
            prompts.append((i, build_prompt(template, row_dict)))
        return prompts

    def _start_worker(self, prompts, model):
        # Determine output mode string
        mode = self.output_mode
        if isinstance(mode, int):
            mode = self._output_mode_values[mode] if mode < len(self._output_mode_values) else "text"

        self._worker = OllamaWorker(self.ollama_url, model, prompts, mode=mode)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.result_text.connect(self._on_result_text)
        self._worker.result_row.connect(self._on_result_row)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._progress_bar.setRange(0, len(prompts))
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._query_btn.setEnabled(False)

        self._thread.start()

    def _cancel_query(self):
        if self._worker:
            self._worker.cancel()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, current, total):
        self._progress_bar.setValue(current)

    def _on_result_text(self, text):
        self._output_view.append(text)
        self._output_view.append("\n---\n")
        self.Outputs.text.send(text)

    def _on_result_row(self, row_idx, text):
        self._row_results[row_idx] = text
        # Show in output viewer
        self._output_view.append(f"[Row {row_idx}] {text}\n")

    def _on_error(self, msg):
        self.error(f"Ollama error: {msg}")
        self._cleanup_thread()

    def _on_finished(self):
        self._cleanup_thread()

        mode = self.output_mode
        if isinstance(mode, int):
            mode = self._output_mode_values[mode] if mode < len(self._output_mode_values) else "text"

        if mode == "table" and self._input_data is not None and self._row_results:
            self._emit_table()

    def _cleanup_thread(self):
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)
        has_input = self._input_data is not None
        self._query_btn.setEnabled(not has_input)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

    # ------------------------------------------------------------------
    # Table output
    # ------------------------------------------------------------------

    def _emit_table(self):
        data = self._input_data
        n = len(data)

        llm_values = np.array(
            [self._row_results.get(i, "") for i in range(n)],
            dtype=object
        ).reshape(-1, 1)

        llm_var = StringVariable("llm")
        new_domain = Domain(
            data.domain.attributes,
            data.domain.class_vars,
            list(data.domain.metas) + [llm_var]
        )
        new_table = data.transform(new_domain)
        # Append the llm column to metas
        if data.metas.shape[1] > 0:
            new_metas = np.hstack([data.metas, llm_values])
        else:
            new_metas = llm_values

        # Rebuild table with updated metas
        from Orange.data import Table as OTable
        out = OTable.from_numpy(
            new_domain,
            X=data.X,
            Y=data.Y if data.Y.size else None,
            metas=new_metas
        )
        self.Outputs.data.send(out)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWLMTask).run()
