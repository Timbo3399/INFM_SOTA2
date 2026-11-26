from pathlib import Path
from unittest import TestCase

import nbformat

NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks"
TIMO_NOTEBOOKS = [
    "workbook_timo_lotr.ipynb",
    "workbook_timo_attention_v2.ipynb",
    "workbook_timo_cluster.ipynb",
]


class NotebookTest(TestCase):
    def test_timo_notebooks_install_from_main(self):
        # Feature-Branches werden nach dem Merge gelöscht, dann scheitert in Colab
        # das `%pip install` aus dem Branch
        for name in TIMO_NOTEBOOKS:
            with self.subTest(notebook=name):
                notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
                branch_lines = [
                    line.split("#")[0].strip()
                    for cell in notebook.cells
                    if cell.cell_type == "code"
                    for line in cell.source.splitlines()
                    if line.startswith("BRANCH =")
                ]
                self.assertEqual(branch_lines, ['BRANCH = "main"'])
                self.assertIn(f"blob/main/notebooks/{name}", notebook.cells[0].source)
