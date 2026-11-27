import ast
import importlib
from pathlib import Path
from unittest import TestCase

import nbformat

NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS = ["workbook_timo_cluster.ipynb"]


def package_imports(notebook) -> list[tuple[str, str]]:
    """Jedes ``from infm_sota2... import name`` in den Code-Zellen."""
    found = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        # Zeilen für IPython wie %pip oder !git sind kein Python. pass hält die
        # Einrückung, falls so eine Zeile allein in einem Block steht.
        lines = [
            line[: len(line) - len(line.lstrip())] + "pass"
            if line.lstrip().startswith(("%", "!"))
            else line
            for line in cell.source.splitlines()
        ]
        source = "\n".join(lines)
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.split(".")[0] == "infm_sota2"
            ):
                found += [(node.module, alias.name) for alias in node.names]
    return found


def importable(module_name: str, name: str) -> bool:
    """Wie ``from module_name import name``: ein Name im Modul oder ein Untermodul."""
    module = importlib.import_module(module_name)
    if hasattr(module, name):
        return True
    try:
        importlib.import_module(f"{module_name}.{name}")
    except ModuleNotFoundError:
        return False
    return True


class NotebookTest(TestCase):
    def test_notebooks_install_from_main(self):
        # Feature-Branches werden nach dem Merge gelöscht, dann scheitert in Colab
        # das `%pip install` aus dem Branch
        for name in NOTEBOOKS:
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

    def test_notebook_imports_exist(self):
        # Colab installiert das Paket von main, die Importe müssen dazu passen
        for name in NOTEBOOKS:
            notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
            imports = package_imports(notebook)
            self.assertTrue(imports, name)
            for module_name, imported in imports:
                with self.subTest(notebook=name, module=module_name, name=imported):
                    self.assertTrue(importable(module_name, imported))
