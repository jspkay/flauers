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

np::ndarray matmul(object, np::ndarray, np::ndarray);
object _inject_value(npy_intp old_value, object srb, object bit, object polarity);
bool checkConversion(PyArrayObject *, PyArrayObject *);

int init_numpy();

PyObject * LineType = nullptr;

BOOST_PYTHON_MODULE(saffira_core){
    Py_Initialize();
    init_numpy();
    def("matmul", matmul, args("self", "a", "b"));
}

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

    npy_intp N1, N2, N3;

    // First, we check whether the two arrays are bi-dimensional (i.e. matrices)
    {
        int a_dims = PyArray_NDIM(_A);
        int b_dims = PyArray_NDIM(_B);

        if(a_dims != 2 || b_dims != 2) {
            std::cout << "[C++] matmul only accepts 2D matrices!!!" << std::endl;
            return np::empty(make_tuple(0), np::dtype::get_builtin<int>());
        }

        npy_intp *a_shape = PyArray_SHAPE(_A),
                *b_shape = PyArray_SHAPE(_B);
        if(a_shape[1] != b_shape[0]) {
            std::cout << "[C++] Matrices dimensions are not compatible" << std::endl;
            return np::empty(make_tuple(0), np::dtype::get_builtin<int>());
        }

        N1 = a_shape[0] + 1;
        N2 = b_shape[1] + 1;
        N3 = a_shape[1] + 1; // same as b_shape[0]
    }

    // Ok, let's convert the array with the correct data-type
    auto A_np = reinterpret_cast<PyArrayObject*>(PyArray_CastToType(_A, _in_dtype, 0));
    auto B_np = reinterpret_cast<PyArrayObject*>(PyArray_CastToType(_B, _in_dtype, 0));

    // Now we have to check all the values are the same, iterating over the arrays
    if( !checkConversion(_A, A_np) ){
        std::cout << "[C++] Couldn't convert matrix A!!!" << std::endl;
        return np::empty(make_tuple(0), np::dtype::get_builtin<int>());
    }
    if( !checkConversion(_B, B_np) ){
        std::cout << "[C++] Couldn't convert matrix B!!!" << std::endl;
        return np::empty(make_tuple(0), np::dtype::get_builtin<int>());
    }
    std::cout << "A_np_converted is of type " << PyArray_TYPE(A_np) << " while original type was " <<
              type_num << " eheh" << std::endl;

    // check whether the systolic object is big enough
    auto selfN1 = extract<npy_intp>(self_systolic.attr("N1"));
    auto selfN2 = extract<npy_intp>(self_systolic.attr("N2"));
    auto selfN3 = extract<npy_intp>(self_systolic.attr("N3"));

    if(selfN1 < N1){
        std::cout << "The systolic array object was constructed with N1 = " << selfN1 << " which is too small for "
                     "this matrix multiplication (that has N1=" << N1 << ")." << std::endl;
    }
    if(selfN2 < N2){
        std::cout << "The systolic array object was constructed with N1 = " << selfN1 << " which is too small for "
                                                                                         "this matrix multiplication (that has N1=" << N1 << ")." << std::endl;
    }
    if(selfN3 < N3){
        std::cout << "The systolic array object was constructed with N1 = " << selfN1 << " which is too small for "
                                                                                         "this matrix multiplication (that has N1=" << N1 << ")." << std::endl;
    }

    std::cout << "[C++] Requiremnts are ok!\nPreparing data structures..." << std::endl;

    // now we make the three arrays a, b, c
    np::ndarray a = np::zeros(make_tuple(N1, N2, N3), np::dtype(self_systolic.attr("in_dtype")));
    np::ndarray b = np::zeros(make_tuple(N1, N2, N3), np::dtype(self_systolic.attr("in_dtype")));
    np::ndarray c = np::zeros(make_tuple(N1, N2, N3), np::dtype(self_systolic.attr("mac_dtype")));

    // Input operations
    npy_intp i, j, k, tmp_i, tmp_j, tmp_k;
    std::cout << "[C++] Performing input operations...." << std::endl;
    j = 0;
    for(i = 0; i < N1; i++){
        for(k=0; k < N3; k++){
            tmp_i = i == 0 ? 1 : i;
            tmp_k = k == 0 ? 1 : k;
            std::cout << "We iterate over eheh = " << (int) *(npy_int8*) PyArray_GETPTR2(A_np, tmp_i - 1, tmp_k - 1)<< std::endl;
            a[i][j][k] = *reinterpret_cast<npy_int8 *>(PyArray_GETPTR2(A_np, tmp_i - 1, tmp_k - 1));
            auto *dunno = reinterpret_cast<PyArrayObject*>(a.ptr());
            std::cout << "We iterate over a[i,j,k] = " << *(npy_int32*)PyArray_GETPTR3(dunno, i, j, k) << std::endl;
        }
    }
    i = 0;
    for(j = 0; j < N1; j++){
        for(k=0; k < N3; k++){
            tmp_j = j == 0 ? 1 : j;
            tmp_k = k == 0 ? 1 : k;
            std::cout << "We iterate over eheh = " << (int) *(npy_int8*) PyArray_GETPTR2(B_np, tmp_j - 1, tmp_k - 1)<< std::endl;
            b[i][j][k] = *reinterpret_cast<npy_int8 *>(PyArray_GETPTR2(B_np, tmp_k - 1, tmp_j - 1));
            auto *dunno = reinterpret_cast<PyArrayObject*>(b.ptr());
            std::cout << "We iterate over b[i,j,k] = " << *(npy_int32*)PyArray_GETPTR3(dunno, i, j, k) << std::endl;
        }
    }

    auto dunno = reinterpret_cast<PyArrayObject *>(b.ptr());
    for(j = 0; j<N2; j++){
        for(i = 0; i < N1; i++){
            for(k = 0; k<N3; k++){
                std::cout << (int) *(npy_int8*)PyArray_GETPTR3(dunno, i, j, k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\n" << std::endl;
    }

    // Differently than in python, here we directly go for the three nested loops, that we would anyways do...
    auto d1 = reinterpret_cast<PyArrayObject*>(a.ptr()),
        d2 = reinterpret_cast<PyArrayObject*>(b.ptr()),
        d3 = reinterpret_cast<PyArrayObject*>(c.ptr());

    bool should_inject = extract<bool>(self_systolic.attr("should_inject"));
    int LineType_a = extract<int>(self_systolic.attr("LineType").attr("a").attr("value")) - 1,
            LineType_b = extract<int>(self_systolic.attr("LineType").attr("b").attr("value")) - 1,
            LineType_c = extract<int>(self_systolic.attr("LineType").attr("c").attr("value")) - 1;

    std::cout << "[C++] Starting the actual computations..." << std::endl;
    for(i = 1; i < N1; i++){
        for(j = 1; j < N2; j++){
            for(k = 1; k < N3; k++){
                a[i][j][k] = a[i][j-1][k];
                b[i][j][k] = b[i-1][j][k];

                // std::cout << "a[i,j-1,k] is " << (int) *(npy_int8 *) PyArray_GETPTR3(d1, i, j-1, k) << std::endl;
                // std::cout << "b[i-1,j,k] is " << (int) *(npy_int8 *) PyArray_GETPTR3(d2, i-1, j, k) << std::endl;

                c[i][j][k] = c[i][j][k-1] + a[i][j-1][k] * b[i-1][j][k];
                // std::cout << "c[i,j,k] is " << (int) *(npy_int32 *) PyArray_GETPTR3(d3, i, j, k) << std::endl;

                if(should_inject){
                    list all_faults = extract<list>(self_systolic.attr("_line_faults"));
                    dict line_c_faults = extract<dict>(all_faults[LineType_c]);
                    object this_itearation_faults = line_c_faults.get(make_tuple(i, j, k));

                    if(this_itearation_faults.is_none()){
                        std::cout << "[C++] For iteration " << i << ", " << j << ", " << k << " there was no fault" << std::endl;
                        continue;
                    }

                    for(int fff = 0; fff < len(this_itearation_faults); fff++){
                        object fault = this_itearation_faults[fff];
                        c[i][j][k] = _inject_value(
                                np::ndarray(c[i][j][k]),
                                fault.attr("should_reverse_bits"),
                                fault.attr("bit"),
                                fault.attr("polarity")
                                );
                        std::cout << extract<char *>(str(fault)) << std::endl;
                    }

                }

            }
        }
    }

    np::ndarray result = np::zeros(make_tuple(N1-1, N2-1), np::dtype(self_systolic.attr("mac_dtype")));

    std::cout << "result has shape " << result.shape(0) << " - " << result.shape(1) << std::endl;

    for(i = 0; i<N1-1; i++){
        for(j = 0; j<N2-1; j++){
            result[i][j] = c[i+1][j+1][N3-1];
        }
    }

    return result;

    // object result = object( handle<>(borrowed( reinterpret_cast<PyObject *>(A_np) )));
    // object result = object(handle<>(A_np_converted));
    // np::ndarray actual_res = np::from_object(result, np::ndarray::NONE);
    // return actual_res;
}

object _inject_value(npy_intp old_value, object srb, object bit, object polarity){
    int b = extract<int>(bit);
    int p = extract<int>(polarity);

    npy_intp newValue = old_value;

    if(p) newValue &= 0x1 << b;
    else newValue |= ~(0x1 << b);

    std::cout << "The value to inject is " << old_value <<
    " and its new value would be " << newValue;

    return object(3);
}

bool checkConversion(PyArrayObject *in_array, PyArrayObject *out_array){

    // TODO: We are assuming only int8 values!!! Nevertheless, we should take a look at how to generalize with _in_dtype!!!
    // Read other comments that starts with [PROBLEM] to learn more

    npy_intp *shape = PyArray_SHAPE(in_array);

    npy_intp nrows = shape[0], ncols = shape[1];

    npy_longlong *ptr; // [PROBLEM] So, here we are using npy_longlong, but the type can be different than this. How
                        // whould we make a general version? I don't know
    npy_byte *npptr;  // [PROBLEM] The same goes here, what if dtype is not int8?!
                        // [PROBLEM] In general, the types of these pointers should be dynamic based on the dtype (as the to_do says).
    for(npy_intp i=0; i<nrows; i++){
        for(npy_intp j=0; j<ncols; j++){
            ptr    = reinterpret_cast<npy_longlong*>( PyArray_GETPTR2(in_array, i, j) );
            npptr = reinterpret_cast<npy_byte*>( PyArray_GETPTR2(out_array, i, j) );

            std::cout << "aptr is " << (int) *ptr << " a_npptr is " << (int) *npptr << std::endl;
            if(*ptr != *npptr){
                std::cout << "[C++] Couldn't convert matrix!!" << std::endl;
                return false;
            }
        }
    }
    return true;
}

int init_numpy(){
    np::initialize();
    int a = _import_array();
    return a;
}

/* Exmple checking for module import
if(LineType == nullptr){
    LineType = PyImport_GetModuleDict();
    if(PyObject_HasAttrString(LineType, "get") ){
        // PyObject *getFunc = PyObject_CallMethod(LineType, "get", "s", "logging");
        PyObject *getFunc = PyImport_GetModule(PyUnicode_FromString("logging"));
        PyObject * str_funcs = PyObject_Str(getFunc);
        std::cout << "str_funcs is surely: " << str_funcs << std::endl;
        if(PyUnicode_1BYTE_KIND == PyUnicode_KIND(str_funcs)){
            char * stringa = (char*) PyUnicode_DATA(str_funcs);
            std::cout << "stringa is: " << stringa << std::endl ;
        }
        std::cout << "eheh. IT's there";
        std::exit(-1);
    }
    std::cout << "[C++] LineType was not imported correctly!" << std::endl;
    std::exit(-1);
}
*/

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