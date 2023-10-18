//
// Created by spappalardo on 26/09/23.
//

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/object.hpp>
#include <boost/python/str.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/iterator.hpp>
#include <numpy/arrayobject.h>

using namespace boost::python;
namespace np = boost::python::numpy;

np::ndarray matmul(object self_systolic, np::ndarray A, np::ndarray B){
    /*! \brief This method performs the systolic matrix multiplication between a and b. The object
     *          self_systolic is needed to provide class attributes and parameters specific to the
     *          implementation.
     *
     *  TODO: Detailed description goes here....
     */
    std::cout << "[C++] Processing matrix multiplication..." << std::endl;

    std::cout << "[C++] Self object is" << extract<char const*>(str(self_systolic)) << std::endl;
    std::cout << "[C++] N1 is " << extract<int>( self_systolic.attr("N1") ) << std::endl;

    // Array conversion (check that all the values are in-range
    np::dtype in_dtype = np::dtype( self_systolic.attr("in_dtype") );
    std::cout << "[C++] in_dtype is " << extract<char const*>(str(in_dtype)) << std::endl;

    // TODO: I guess the best course of action is gonna be get the actual numpy object and work on it directly... So that is should be simpler to manipulate it using native methods.
    if( A.is_none() ){
        std::cout << "Pointer is null here..." << std::endl;
    }
    else std::cout << "Pointer is fine!" << std::endl;
    PyObject *A_python_obj = A.ptr();

    std::cout << "before pyarray check" << std::endl;
    PyArray_Check(A_python_obj);
    std::cout << "after pyarray check" << std::endl;

    if(PyArray_Check(A_python_obj)){
        std::cout << "This array is legit" << std::endl;
    }else std::cout << "Oh no. There is a big problem here." << std::endl;

    auto * _A = reinterpret_cast<PyArrayObject*>( A.ptr() );
    auto * _B = reinterpret_cast<PyArrayObject*>( B.ptr() );
    int np_in_dtype = PyArray_TYPE(_A);
    int itemsize = PyArray_ITEMSIZE(_A);

    PyArray_Descr *arr_desc = PyArray_DESCR(_A);
    int type_num = arr_desc->type_num;

    std::cout << "The array is signed: " << static_cast<bool>(PyDataType_ISSIGNED(arr_desc)) << "\n"
        << "The array is integer: " << static_cast<bool>(PyDataType_ISINTEGER(arr_desc)) << "\n"
        << std::endl;


    double max, min;
    switch (type_num) {
        case NPY_INT8:
            min = -128;
            max = 127;
            break;
        case NPY_INT16:
            min = - (0x1 << 16);
            max = + (0x1 << 16) - 1;
            break;
        case NPY_LONG:
            std::cout << "Input type is NPY_LONG" << std::endl;
            break;
        default:
            std::cout << "There was an error here..." << std::endl;
            break;
    }

    auto _in_dtype = reinterpret_cast<PyArray_Descr*>(in_dtype.ptr());
    switch (_in_dtype->type_num) {
        case NPY_BYTE:
            std::cout << "It is the expected one!!!! :D" << std::endl;
            break;
        default:
            std::cout << "something unexpected! :(" << std::endl;
            break;
    }

    std::cout << "So, the dtype is " << np_in_dtype << " with itemsize " << itemsize
        << " while min = " << min << " and max is " << max
        << " typenum is " << type_num
        << std::endl;

    // Ok, let's convert the array with the correct data-type
    auto A_np = reinterpret_cast<PyArrayObject*>(PyArray_CastToType(_A, _in_dtype, 0));
    auto B_np = reinterpret_cast<PyArrayObject*>(PyArray_CastToType(_B, _in_dtype, 0));

    npy_intp N1, N2, N3;

    // First, we check whether the two arrays are bi-dimensional (i.e. matrices)
    {
        int a_dims = PyArray_NDIM(A_np);
        int b_dims = PyArray_NDIM(B_np);

        if(a_dims != 2 || b_dims != 2) {
            std::cout << "[C++] matmul only accepts 2D matrices!!!" << std::endl;
            return np::empty(make_tuple(0), np::dtype::get_builtin<int>());
        }

        npy_intp *a_shape = PyArray_SHAPE(A_np),
                *b_shape = PyArray_SHAPE(B_np);
        if(a_shape[1] != b_shape[0]) {
            std::cout << "[C++] Matrices dimensions are not compatible" << std::endl;
            return np::empty(make_tuple(0), np::dtype::get_builtin<int>());
        }

        N1 = a_shape[0];
        N2 = b_shape[1];
        N3 = a_shape[1]; // same as b_shape[0]
    }

    // Now we gotta check all the values are the same, iterating over the arrays
    {
        // TODO: We are assuming only int8 values!!! Nevertheless, we should take a look at how to generalize with _in_dtype!!!
        npy_byte *aptr, *a_npptr;  // The types of these pointers should be dynamic based on the dtype (as the to_do says).
        for(npy_intp i=0; i<N1; i++){
            for(npy_intp j=0; j<N3; j++){
                aptr    = reinterpret_cast<npy_byte*>( PyArray_GETPTR2(_A, i, j) );
                a_npptr = reinterpret_cast<npy_byte*>(PyArray_GETPTR2(A_np, i, j));

                std::cout << "aptr is " << reinterpret_cast<npy_byte>(*aptr) << " a_npptr is " << (int) *a_npptr << std::endl;
                if(*aptr != *a_npptr){
                    std::cout << "[C++] Couldn't convert matrix A!!!!" << std::endl;
                    return np::empty(make_tuple(0), np::dtype::get_builtin<int>());
                }
            }
        }
    }





    std::cout << "A_np_converted is of type " << PyArray_TYPE(A_np) << " while original type was " <<
        type_num << " eheh" << std::endl;

    object result = object( handle<>(borrowed( reinterpret_cast<PyObject *>(A_np) )));
    // object result = object(handle<>(A_np_converted));
    np::ndarray actual_res = np::from_object(result, np::ndarray::NONE);
    return actual_res;
}

int init_numpy(){
    np::initialize();
    int a = _import_array();
    return a;
}

BOOST_PYTHON_MODULE(saffira){
    Py_Initialize();
    init_numpy();
    def("matmul", matmul, args("self", "a", "b"));
}

/* Old version...
#include <vector>
#include "numpy/arrayobject.h"

#define PY_SSIZE_T_CLEAN
#include "Python.h"

static PyObject *saffira_matmul(PyObject* self, PyObject* args);

static PyObject *error;

static PyMethodDef SaffiraMethods[] = {
        {"matmul", saffira_matmul, METH_VARARGS, "doc_string"},
        {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef saffira = {
        PyModuleDef_HEAD_INIT,
        "saffira",
        nullptr,
        -1,
        SaffiraMethods
};

PyMODINIT_FUNC
PyInit_saffira(void){
    return PyModule_Create(&saffira);
}

static PyObject *saffira_matmul(PyObject* self, PyObject * args){
    printf("Ciao!\n");

    PyObject *saffira, *a, *b;
    PyArrayObject *a_arr, *b_arr;

    printf("Reading the argument\n");
    if(! PyArg_ParseTuple(args, "OOO", &saffira, &a, &b) )
        Py_RETURN_NONE;
    printf("Successfully read!\n");

    printf("self is %x ", self);
    printf("saffira is %x\n", saffira);

    printf("So, b is %x\n", b);
    printf("is b == to PyTypeInt? %d\n", (void*) b == (void*) (&PyLong_Type) );
    printf("So PyLong_Type is in position %x\n", &PyLong_Type);

    PyObject *arrayType = (void *) &PyArray_Type;
    if(PyObject_IsInstance(a, (PyObject*) (void *) &PyArray_Type) ){
        printf("Correct!");
        a_arr = (PyArrayObject*) a;
    }
    else printf("Wrong");

    printf("You passed a numpy array with %d dimensions\n", a_arr->nd);


    Py_RETURN_NONE;
}
*/