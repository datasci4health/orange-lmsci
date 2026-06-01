# Orange LMSci

# LM Task — Orange Data Mining Widget

An Orange 3 widget that connects to a locally running **Ollama** server and lets you run LLM prompts against your data.

---

## Usage

### With an input table

1. Connect a **Data** table to the widget's input.
2. Write a prompt template using `{column_name}` placeholders matching your table's column names:

   ```
   Classify the sentiment of this review:

   {review_text}

   Reply with only: positive, negative, or neutral.
   ```

3. Choose **Output mode**:
   - **Text** — each LLM response is emitted on the *Text* output one by one.
   - **Table** — when all rows are processed, a copy of the input table is sent on the *Data* output with an extra `llm` column containing each row's LLM response.

4. The widget processes every row automatically when data is connected.

### Standalone (no input table)

- The **Query** button and **Query on load** checkbox are enabled.
- Write any prompt (without placeholders, or with literal `{braces}` — they won't be substituted).
- Click **Query** to send it once.
- Check **Query on load** to run the prompt automatically every time the workflow is opened.

---

## Controls

| Control | Description |
|---|---|
| **URL** | Ollama server base URL (default: `http://localhost:11434`) |
| **Model** | Dropdown populated from `/api/tags`; editable for manual entry |
| **↺ (Refresh)** | Re-fetches the model list from the server |
| **Output: Text / Table** | Selects output mode |
| **Query** | Sends the standalone prompt (disabled when table input is connected) |
| **Query on load** | Auto-query on workflow load when no input is connected |
| **Cancel** | Cancels the current in-progress query batch |

---

## Requirements

- Orange 3 (`orange3`)
- A running [Ollama](https://ollama.com) server (e.g. `ollama serve`)
- No extra Python dependencies — uses only stdlib `urllib` and `json`
