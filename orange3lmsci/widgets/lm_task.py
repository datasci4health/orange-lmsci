"""
Orange Data Mining Widget: LM Task
Interacts with a local Ollama server.
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
    QLineEdit
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal, QObject

import numpy as np

from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, StringVariable


OLLAMA_DEFAULT_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class OllamaWorker(QObject):
    progress    = pyqtSignal(int, int)
    result_text = pyqtSignal(str)
    result_row  = pyqtSignal(int, str)
    finished    = pyqtSignal()
    error       = pyqtSignal(str)

    def __init__(self, url, model, prompts, mode="text"):
        super().__init__()
        self._url       = url.rstrip("/")
        self._model     = model
        self._prompts   = prompts   # list of (index, prompt_str)
        self._mode      = mode
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
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def build_prompt(template, row_dict):
    def replacer(match):
        key = match.group(1)
        return str(row_dict.get(key, match.group(0)))
    return re.sub(r"\{(\w+)\}", replacer, template)


def extract_fields(template):
    return re.findall(r"\{(\w+)\}", template)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class OWLMTask(OWWidget):
    name        = "LM Task"
    description = ("Send prompts to a local Ollama server, optionally using "
                   "input table columns as template fields.")
    icon        = str(files("orange3lmsci") / "icons" / "LMTask.svg")
    priority    = 100
    category    = "LMSci"
    keywords    = ["llm", "ollama", "prompt", "language model"]

    want_main_area = True

    class Inputs:
        data = Input("Data", Table, auto_summary=False)

    class Outputs:
        data = Output("Data", Table, auto_summary=False)
        text = Output("Text", str,   auto_summary=False)

    # Settings
    ollama_url      : str  = Setting(OLLAMA_DEFAULT_URL)
    selected_model  : str  = Setting("")
    prompt_template : str  = Setting("")
    # FIX #2 — store as int (radio-button index); 0=text, 1=table
    output_mode_idx : int  = Setting(0)
    query_on_load   : bool = Setting(False)

    OUTPUT_MODES = ["text", "table"]

    def __init__(self):
        super().__init__()
        self._input_data  = None
        self._worker      = None
        self._thread      = None
        self._row_results = {}

        self._build_control_area()
        self._build_main_area()

        # FIX #1 — refresh models using a plain daemon thread + Qt signal
        self._models_ready.connect(self._update_model_list)
        threading.Thread(target=self._refresh_models_bg, daemon=True).start()

        if self.query_on_load and self._input_data is None:
            from AnyQt.QtCore import QTimer
            QTimer.singleShot(500, self._run_query)

    # Signal used to safely deliver model list to the main thread
    from AnyQt.QtCore import pyqtSignal as _sig
    _models_ready = _sig(list)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_control_area(self):
        ca = self.controlArea

        # Server box
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

        # Output mode  — FIX #2: bind to output_mode_idx (int)
        out_box = gui.vBox(ca, "Output")
        gui.radioButtonsInBox(
            out_box, self, "output_mode_idx",
            btnLabels=["Text", "Table (adds 'llm' column)"],
        )

        # Standalone controls
        standalone_box = gui.vBox(ca, "Standalone Mode")
        self._query_btn = gui.button(
            standalone_box, self, "Query", callback=self._run_query
        )
        self._qol_checkbox = gui.checkBox(
            standalone_box, self, "query_on_load", "Query on load"
        )

        # Progress / cancel
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

        # Prompt editor
        prompt_group = QGroupBox("Prompt Template")
        prompt_layout = QVBoxLayout()
        prompt_group.setLayout(prompt_layout)

        hint = QLabel("Use {column_name} placeholders for input table columns.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        prompt_layout.addWidget(hint)

        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setPlaceholderText(
            "e.g. Summarize the following text:\n\n{text}"
        )
        self._prompt_edit.setPlainText(self.prompt_template)
        from AnyQt.QtGui import QFont
        self._prompt_edit.setFont(QFont("Courier New", 10))
        self._prompt_edit.textChanged.connect(self._on_prompt_changed)
        prompt_layout.addWidget(self._prompt_edit)

        self._fields_label = QLabel()
        self._fields_label.setStyleSheet("color: #555; font-size: 11px;")
        prompt_layout.addWidget(self._fields_label)
        splitter.addWidget(prompt_group)

        # Output viewer
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)

        self._output_view = QTextEdit()
        self._output_view.setReadOnly(True)
        from AnyQt.QtGui import QFont
        self._output_view.setFont(QFont("Courier New", 10))
        output_layout.addWidget(self._output_view)
        splitter.addWidget(output_group)

        splitter.setSizes([300, 200])
        self._update_fields_label()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @Inputs.data
    def set_data(self, data):
        self._input_data = data
        has_input = data is not None
        self._query_btn.setEnabled(not has_input)
        self._qol_checkbox.setEnabled(not has_input)
        self._update_fields_label()
        if has_input:
            self._output_view.clear()
            self._run_query()

    # ------------------------------------------------------------------
    # Settings callbacks
    # ------------------------------------------------------------------

    def _on_url_changed(self):
        self.ollama_url = self._url_edit.text().strip() or OLLAMA_DEFAULT_URL
        self._refresh_models()

    def _on_model_changed(self, text):
        self.selected_model = text

    def _on_prompt_changed(self):
        self.prompt_template = self._prompt_edit.toPlainText()
        self._update_fields_label()

    # ------------------------------------------------------------------
    # Model list  (FIX #1)
    # ------------------------------------------------------------------

    def _refresh_models(self):
        self._refresh_btn.setEnabled(False)
        threading.Thread(target=self._refresh_models_bg, daemon=True).start()

    def _refresh_models_bg(self):
        """Runs in daemon thread; emits signal to deliver results safely."""
        models = fetch_ollama_models(self.ollama_url)
        self._models_ready.emit(models)          # crosses thread boundary safely

    def _update_model_list(self, models):
        """Runs on main thread via signal."""
        current = self._model_combo.currentText() or self.selected_model
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        if models:
            self._model_combo.addItems(models)
            if current in models:
                self._model_combo.setCurrentText(current)
            elif self.selected_model in models:
                self._model_combo.setCurrentText(self.selected_model)
            else:
                self._model_combo.setCurrentIndex(0)
        else:
            if self.selected_model:
                self._model_combo.addItem(self.selected_model)
                self._model_combo.setCurrentText(self.selected_model)
        self._model_combo.blockSignals(False)
        self.selected_model = self._model_combo.currentText()
        self._refresh_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Fields label
    # ------------------------------------------------------------------

    def _update_fields_label(self):
        fields = extract_fields(self.prompt_template)
        if fields:
            self._fields_label.setText(
                f"Fields detected: {', '.join('{' + f + '}' for f in fields)}"
            )
        else:
            self._fields_label.setText("No {field} placeholders detected.")

        if self._input_data is not None and fields:
            col_names = [v.name for v in
                         list(self._input_data.domain.variables) +
                         list(self._input_data.domain.metas)]
            missing = [f for f in fields if f not in col_names]
            if missing:
                self._fields_label.setText(
                    self._fields_label.text() +
                    f"  ⚠ Missing columns: {', '.join(missing)}"
                )

    # ------------------------------------------------------------------
    # Query
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
            prompts = [(0, template)]

        self._start_worker(prompts, model)

    def _build_table_prompts(self, template):
        data     = self._input_data
        all_vars = list(data.domain.variables) + list(data.domain.metas)
        prompts  = []
        for i, row in enumerate(data):
            row_dict = {}
            for var in all_vars:
                try:
                    val = row[var]
                    row_dict[var.name] = (var.repr_val(val)
                                          if hasattr(var, "repr_val")
                                          else str(val))
                except Exception:
                    row_dict[var.name] = ""
            prompts.append((i, build_prompt(template, row_dict)))
        return prompts

    def _start_worker(self, prompts, model):
        mode = self.OUTPUT_MODES[self.output_mode_idx]

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
        self._output_view.append(f"[Row {row_idx}] {text}\n")

    def _on_error(self, msg):
        self.error(f"Ollama error: {msg}")
        self._cleanup_thread()

    def _on_finished(self):
        self._cleanup_thread()
        mode = self.OUTPUT_MODES[self.output_mode_idx]
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
        n    = len(data)

        llm_values = np.array(
            [self._row_results.get(i, "") for i in range(n)],
            dtype=object
        ).reshape(-1, 1)

        llm_var    = StringVariable("llm")
        new_domain = Domain(
            data.domain.attributes,
            data.domain.class_vars,
            list(data.domain.metas) + [llm_var]
        )

        if data.metas.size:
            new_metas = np.hstack([data.metas, llm_values])
        else:
            new_metas = llm_values

        out = Table.from_numpy(
            new_domain,
            X=data.X,
            Y=data.Y if data.Y.size else None,
            metas=new_metas
        )
        self.Outputs.data.send(out)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWLMTask).run()
