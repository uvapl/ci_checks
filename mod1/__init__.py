import check50
import check50.internal
import re
import nbformat
import shutil

import os
import subprocess
from importlib.metadata import version
import nbconvert

NOTEBOOK_PATH = "module1.ipynb"

# TODO: move this into check_jupyter.py
def get_test_ids(notebook_path):
    """
    Get all test ids from a notebook
    A test is marked by an nbgrader id with 'test_' as prefix
    """
    
    #raise Exception(os.getcwd() + ", ".join(os.listdir()))
    
    if not os.path.exists(NOTEBOOK_PATH) and check50.internal.run_root_dir != check50.internal.student_dir:
        shutil.copyfile(check50.internal.student_dir / NOTEBOOK_PATH, check50.internal.run_root_dir / NOTEBOOK_PATH)
    
#     process = subprocess.run(['pip', 'show', 'nbconvert'], 
#                          stdout=subprocess.PIPE, 
#                          universal_newlines=True)
    
#     raise Exception(process.stdout)

    raise Exception(version(nbconvert))
    
    # Open and parse the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Grab all test cells
    test_cells = []
    for cell in nb.cells:
        # TODO: replace with "check_jupyter.get_cell_id"
        cell_id = cell.metadata.get("nbgrader", {}).get("grade_id", "")

        # TODO: replace with "check_jupyter.is_test_cell"
        if cell_id.startswith("test_"):
            test_cells.append(cell_id)

    return test_cells


@check50.check()
def exists():
    """Notebook exists"""
    check50.include("check_jupyter.py", "data", "answers.py")

    # Grab the last test
    test_ids = get_test_ids(NOTEBOOK_PATH)
    last_test_id = test_ids[-1]

    # Grab all cells up to the last test cell
    check_jupyter = check50.internal.import_file("check_jupyter", "check_jupyter.py")
    cells = check_jupyter.cells_up_to(NOTEBOOK_PATH, last_test_id)

    # Gather all results
    results = []

    with check_jupyter.executor() as execute:
        for cell in cells:
            # Execute the cell
            try:
                execute(cell)
                exception = ""
                passed = True
            except check50.Failure as f:
                exception = str(f)
                passed = False

            # If it's a test cell, record result
            if check_jupyter.is_test_cell(cell):
                results.append((check_jupyter.get_cell_id(cell), passed, exception))


    # Pass down the results to the actual checks
    return tuple(results)


def create_check(test_id):
    """
    Create a check for a test cell with test_id.
    The check will perform a lookup in passed down results.
    Results is a tuple with the following structure:
        ((test_id, passed, exception), (test_id, passed, exception), ...)
    """
    def check(results):
        for id, passed, exception in results:
            if id == test_id and not passed:
                raise check50.Failure(exception)
        return results

    check.__name__ = test_id
    check.__doc__ = f"{test_id} passes"
    return check


def init():
    for test_id in get_test_ids(NOTEBOOK_PATH ):
        check = create_check(test_id)

        # Register the check with check50
        check = check50.check(dependency=exists)(check)

        # Add the check to global module scope
        globals()[test_id] = check

init()
