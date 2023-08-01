//
// Created by spappalardo on 20/07/23.
//

#include "systolic_injector.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *spam_system(PyObject *, PyObject*);

static PyObject *SpamError;

static PyMethodDef SpamMethods[] = {
        {"system", spam_system, METH_VARARGS, "Execute a shell command."},
        {0, 0, 0, 0}
};

static struct PyModuleDef systolic_injector = {
        PyModuleDef_HEAD_INIT,
        "systolic_injector",
        NULL,
        -1,
        SpamMethods
};


volatile PyMODINIT_FUNC
PyInit_systolic_injector(void){
    PyObject *m;

    m = PyModule_Create(&systolic_injector);
    if(m==NULL) return NULL;

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(SpamError);
    if(PyModule_AddObject(m, "error", SpamError) < 0){
        Py_XDECREF(SpamError);
        Py_CLEAR(SpamError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

static PyObject * spam_system(PyObject *self, PyObject *args){
    const char* command;
    int sts;

    if(!PyArg_ParseTuple(args, "s", &command)) return NULL;

    sts = system(command);
    return PyLong_FromLong(sts);
}

