import nbformat
import check50
import time
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

import contextlib

def cells_up_to(notebook_path, cell_id):
    """
    Grab all cells from notebook_path upto cell_id
    """
    # Open and parse the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Grab all code cells up to cell_id
    cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            cells.append(cell)

        if cell.metadata.get("nbgrader", {}).get("grade_id") == cell_id:
            break
    else:
        raise Exception(f"Cell with ID:{cell_id} not found.")

    return cells


def create_cell(code):
    """
    Create a new cell
    """
    return nbformat.notebooknode.from_dict(
        {"source": code,
         "cell_type": "code",
         "metadata": {}
        })


def execute(cells):
    """
    Execute all cells
    """
    with executor() as execute:
        return execute(cells)


@contextlib.contextmanager
def executor():
    """
    Creates an executor context in which any call to the executor
    is executed in sequence in the same kernel. For example:
    with executor() as execute:
        execute(create_cell("a = 3"))
        execute(create_cell("print(a)"))
    """
    def execute(cells):
        # If just one cell is passed, put it in a list
        if not isinstance(cells, list) and not isinstance(cells, tuple):
            cells = [cells]

        # Execute all cells
        results = []
        for index, cell in enumerate(cells):
            try:
                results.append(ep.preprocess_cell(cell, {}, index))
            except CellExecutionError as e:
                time.sleep(.5)
                raise check50.Failure(str(e))

        return results

    # Start an ExecutePreprocessor: https://nbconvert.readthedocs.io/en/latest/execute_api.html
    # https://github.com/jupyter/nbconvert/blob/master/nbconvert/preprocessors/execute.py
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    # Start a Kernel Manager
    with ep.setup_preprocessor(None, {}):
        yield execute


def output_from_cell(cell):
    """
    Grabs the output from a cell.
    """
    output_data = cell[0]["outputs"][0]

    if output_data["output_type"] == "stream":
        return output_data["text"]

    if output_data["output_type"] == "display_data":
        return output_data["data"]["text/plain"]

    raise Exception("Unknown output format")


def get_cell_id(cell):
    """
    Get the nbgrader grade_id from a cell.
    Return the empty string "" if no id found.
    """
    return cell.metadata.get("nbgrader", {}).get("grade_id", "")


def is_test_cell(cell):
    """
    Returns True if nbgrader grade_id starts with "test_".
    """
    return get_cell_id(cell).startswith("test_")



"""
nbgrader generate_assignment --CourseDirectory.source_directory=. --CourseDirectory.release_directory=release/. --assignment=.
"""
