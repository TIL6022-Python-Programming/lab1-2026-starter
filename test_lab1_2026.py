"""
pytest test suite for Lab 1 (2026) - Python Environment assignment.

Cells are identified by JUPYTER CELL TAGS. Each answer
cell in the template notebook must carry a metadata tag:

    Q1 -> "q1-answer"      Q4 -> "q4-answer"
    Q2 -> "q2-answer"      Q5 -> "q5-answer"
    Q3 -> "q3-answer"      Q6 -> "q6-answer"

Tags live in `cell.metadata.tags`, which is separate from the cell's visible
source code. 

ADDING TAGS TO A TEMPLATE NOTEBOOK
-----------------------------------
- JupyterLab: select the cell, open the Property Inspector (the wrench/tag
  icon in the right sidebar), and add the tag under "Cell Tags".
- Classic Notebook: View menu -> Cell Toolbar -> Tags, then type the tag
  into the box that appears on the cell and press Enter.
- VS Code (Jupyter extension): click "..." on the cell, then "Add Cell Tag"
  (or use the Command Palette: "Notebook: Add Cell Tag").
- Directly in the .ipynb JSON: set `cell["metadata"]["tags"] = ["q1-answer"]`
  for the relevant cell (this is exactly what the helper script below does).

WHAT THE TESTS DO
------------------
For each notebook found, every code cell is executed in order in one shared
namespace (mirroring "Restart & Run All"), and each cell's stdout is
captured individually. For each question, the test finds the cell tagged
"qN-answer" (searching the whole notebook, regardless of position) and
compares what THAT cell printed to the expected output from the answer key.
If no cell carries the expected tag (or more than one does), the test fails
immediately with instructions, instead of silently comparing the wrong cell.

USAGE
-----
Point the suite at a single submitted notebook:

    NOTEBOOK_PATH=/path/to/student_submission.ipynb pytest test_lab1_2026.py -v --tb=short

Or point it at a folder full of submissions to grade the whole class in one
run (every question is tested against every *.ipynb file found):

    NOTEBOOK_PATH=/path/to/submissions_folder pytest test_lab1_2026.py -v --tb=short

If NOTEBOOK_PATH isn't set, the suite looks for a "submissions/" folder next
to this test file, and falls back to any *.ipynb file in the current
directory.

"""

import glob
import io
import json
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Configuration: the cell tag that identifies each question's answer cell,
# and the exact stdout that cell must produce.
# ---------------------------------------------------------------------------

CELL_TAGS = {
    "Q1": "q1-answer",
    "Q2": "q2-answer",
    "Q3": "q3-answer",
    "Q4": "q4-answer",
    "Q5": "q5-answer",
    "Q6": "q6-answer",
}

EXPECTED_OUTPUT = {
    "Q1": "5\n",
    "Q2": "the sum is positive\n",
    "Q3": "Anna\n",
    "Q4": "Anna\n-------\nBob\n-------\nCharlie\n-------\nDavid\n-------\n",
    "Q5": "0 record\n1 record\n2 record\n3 record\n",
    "Q6": "0 Anna\n1 Bob\n2 Charlie\n3 David\n",
}


# ---------------------------------------------------------------------------
# Notebook discovery
# ---------------------------------------------------------------------------

def _discover_notebooks():
    """Return a sorted list of notebook paths to test, based on the
    NOTEBOOK_PATH environment variable (a single .ipynb file OR a folder of
    them), falling back to a local 'submissions/' folder or any *.ipynb in
    the current directory."""
    env_path = os.environ.get("NOTEBOOK_PATH")

    if env_path:
        if os.path.isdir(env_path):
            return sorted(glob.glob(os.path.join(env_path, "*.ipynb")))
        return [env_path]

    if os.path.isdir("submissions"):
        return sorted(glob.glob(os.path.join("submissions", "*.ipynb")))

    return sorted(glob.glob("*.ipynb"))


def pytest_generate_tests(metafunc):
    if "notebook_path" in metafunc.fixturenames:
        paths = _discover_notebooks()
        if not paths:
            paths = [None]  # yields one clearly-failing test instead of "no tests collected"
        ids = [os.path.basename(p) if p else "NO_NOTEBOOK_FOUND" for p in paths]
        metafunc.parametrize("notebook_path", paths, ids=ids)


# ---------------------------------------------------------------------------
# Notebook execution helpers
# ---------------------------------------------------------------------------

def _get_source(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src


def _get_tags(cell):
    return cell.get("metadata", {}).get("tags", []) or []


def _load_cells(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return nb["cells"]


def _run_notebook_capture_per_cell_stdout(cells):
    """Execute every code cell in order inside one shared namespace
    (mirroring 'Restart & Run All') and return a dict mapping cell index ->
    whatever that specific cell printed to stdout. A cell that raises an
    exception records the exception text instead of stopping the run, so a
    crash in an unrelated cell (e.g. a missing optional import) doesn't
    prevent later answer cells from being checked."""
    namespace = {}
    outputs = {}
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = _get_source(cell)
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            exec(compile(source, f"<cell {idx}>", "exec"), namespace)
        except Exception as e:  # noqa: BLE001 - deliberately broad, we record and move on
            outputs[idx] = f"<<EXCEPTION during cell {idx}: {type(e).__name__}: {e}>>"
        else:
            outputs[idx] = captured.getvalue()
        finally:
            sys.stdout = old_stdout
    return outputs


def _find_cell_by_tag(cells, tag):
    """Return the index of the single cell carrying `tag`. Fails loudly
    (via pytest.fail) if zero or more than one cell has it, rather than
    guessing."""
    matches = [idx for idx, cell in enumerate(cells) if tag in _get_tags(cell)]

    if len(matches) == 0:
        pytest.fail(
            f"No cell tagged '{tag}' was found in this notebook. Add that "
            f"tag to the intended answer cell -- in VS Code, click '...' "
            f"on the cell, then 'Add Cell Tag', type the '{tag}'. See the "
            f"module docstring in this test file for other Jupyter versions. ",
            pytrace=False,
        )
    if len(matches) > 1:
        pytest.fail(
            f"Cell tag '{tag}' appears on {len(matches)} cells (indices "
            f"{matches}) -- it must be unique. Remove the tag from whichever "
            f"cell isn't the intended answer.",
            pytrace=False,
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_notebooks():
    """Cache: notebook path -> (cells, per-cell stdout outputs). Computed
    once per notebook even though multiple question tests need it."""
    cache = {}

    def _load(path):
        if path is None:
            pytest.fail(
                "No notebook found to test. Set the NOTEBOOK_PATH environment "
                "variable to a student's .ipynb file (or a folder of them), "
                "e.g.: NOTEBOOK_PATH=submission.ipynb pytest test_lab1_2026.py -v",
                pytrace=False,
            )
        if not os.path.exists(path):
            pytest.fail(f"Notebook not found: {path}", pytrace=False)
        if path not in cache:
            cells = _load_cells(path)
            outputs = _run_notebook_capture_per_cell_stdout(cells)
            cache[path] = (cells, outputs)
        return cache[path]

    return _load


def _check_question(loaded_notebooks, notebook_path, question):
    cells, outputs = loaded_notebooks(notebook_path)
    tag = CELL_TAGS[question]
    idx = _find_cell_by_tag(cells, tag)

    if cells[idx].get("cell_type") != "code":
        pytest.fail(
            f"{question}: the cell tagged '{tag}' in "
            f"{os.path.basename(notebook_path)} is a "
            f"'{cells[idx].get('cell_type')}' cell, not code. Move the tag "
            f"onto the actual answer code cell.",
            pytrace=False,
        )

    actual = outputs.get(idx, "")
    expected = EXPECTED_OUTPUT[question]
    if actual != expected:
        # A short, message-only failure (no Python traceback)
        pytest.fail(
            f"{question} MISMATCH -- got: {actual!r} | expected: {expected!r}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Tests -- one per question, parametrized over every discovered notebook
# ---------------------------------------------------------------------------

def test_q1_sum_of_a_and_b(loaded_notebooks, notebook_path):
    _check_question(loaded_notebooks, notebook_path, "Q1")


def test_q2_sum_is_positive(loaded_notebooks, notebook_path):
    _check_question(loaded_notebooks, notebook_path, "Q2")


def test_q3_first_name_in_student_list(loaded_notebooks, notebook_path):
    _check_question(loaded_notebooks, notebook_path, "Q3")


def test_q4_names_separated_by_dashes(loaded_notebooks, notebook_path):
    _check_question(loaded_notebooks, notebook_path, "Q4")


def test_q5_items_with_record_label(loaded_notebooks, notebook_path):
    _check_question(loaded_notebooks, notebook_path, "Q5")


def test_q6_combined_id_and_name(loaded_notebooks, notebook_path):
    _check_question(loaded_notebooks, notebook_path, "Q6")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
