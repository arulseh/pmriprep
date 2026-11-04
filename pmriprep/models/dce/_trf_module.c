/* Private entry points; Python owns and validates all numerical storage. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "_trf_solver.c"
#include "_trf_recurrence.c"

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_native_trf",
    "Private four-coordinate DCE numerical kernel.", -1, NULL
};

PyMODINIT_FUNC PyInit__native_trf(void) {
    PyObject *result = PyModule_Create(&module);
    if (!result) return NULL;
    const char *names[] = {"evaluate", "solve", "svd", "step", "recurrence"};
    void *functions[] = {(void *)evaluate, (void *)trf_solve, (void *)trf_svd,
                         (void *)trf_step, (void *)batch_recurrence};
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
        PyObject *address = PyLong_FromVoidPtr(functions[i]);
        if (!address) { Py_DECREF(result); return NULL; }
        if (PyModule_AddObject(result, names[i], address) < 0) {
            Py_DECREF(address);
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}
