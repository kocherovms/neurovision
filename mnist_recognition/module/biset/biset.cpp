#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <unordered_set>
#include <string>
#include <sstream>

struct VectorHash {
    size_t operator()(const std::vector<int> & theV) const {
        std::hash<int> hasher;
        size_t seed = 0;
        
        for (int i : theV)
            seed ^= hasher(i) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        
        return seed;
    }
};

typedef std::vector<int> KeyType;
typedef std::unordered_set<KeyType, VectorHash> biset;

static std::string
get_key_repr(const int theKeyInd, const KeyType & theKey) {
    int maxElemsToDump = 1000;
    std::stringstream ss;
    bool isFirst = true;
    ss << "#" << theKeyInd << ": [";

    for(int e : theKey) {
        if(maxElemsToDump <= 0) {
            ss << "...";
            break;
        }

        ss << (isFirst > 0 ? "" : ", ") << e;
        maxElemsToDump--;
        isFirst = false;
    }

    ss << "]";
    return ss.str();
}

static PyObject *
biset_create(PyObject *, PyObject * theArgs) {
    biset * inst = new biset();
    return PyLong_FromVoidPtr(inst);
}

static PyObject *
biset_destroy(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    
    if(!PyArg_ParseTuple(theArgs, "L", &inst))
        return NULL;

    if(!inst)
        return PyErr_Format(PyExc_Exception, "bad arguments");

    delete (biset *)inst;
    Py_RETURN_NONE;
}

static PyObject *
biset_get_len(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    
    if(!PyArg_ParseTuple(theArgs, "L", &inst))
        return NULL;

    if(!inst)
        return PyErr_Format(PyExc_Exception, "bad arguments");

    const biset * const s = (biset *)inst;
    return PyLong_FromLong(s->size());
}

static PyObject *
biset_contains(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    int keySize = 0;
    Py_buffer keyBuf;
    
    if(!PyArg_ParseTuple(theArgs, "Liy*", &inst, &keySize, &keyBuf))
        return NULL;

    PyObject * rv = NULL;

    if(inst && keySize >= 0 && keyBuf.buf) {
        const biset * const s = (biset *)inst;
        const int * const key = (int *)keyBuf.buf;
        const std::vector<int> v(key, key + keySize);
        rv = s->find(v) != s->end() ? Py_True : Py_False;
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&keyBuf);
    return rv;
}

static PyObject *
biset_get(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    int keySize = 0;
    Py_buffer keyBuf;
    
    if(!PyArg_ParseTuple(theArgs, "Liy*", &inst, &keySize, &keyBuf))
        return NULL;

    PyObject * rv = NULL;

    if(inst && keySize >= 0 && keyBuf.buf) {
        const biset * const s = (biset *)inst;
        const int * const key = (int *)keyBuf.buf;
        const std::vector<int> v(key, key + keySize);
        const auto it = s->find(v);

        if(it != s->end()) {
            rv = PyList_New(0);
            assert(rv);

            for(int i: v) {
                PyObject * const pyI = PyLong_FromLong(i);
                assert(pyI);
                PyList_Append(rv, pyI);
            }
        }
        else {
            PyErr_Format(PyExc_KeyError, "%s", get_key_repr(0, v).c_str());
        }
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&keyBuf);
    return rv;
}

static PyObject *
biset_add_many(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    int keysCount = 0;
    int keyStride = 0;
    Py_buffer keySizesBuf;
    Py_buffer keysBuf;
    Py_buffer isAddedBoolmapBuf;
    
    if(!PyArg_ParseTuple(theArgs, "Liiy*y*y*", &inst, &keysCount, &keyStride, &keySizesBuf, &keysBuf, &isAddedBoolmapBuf))
        return NULL;

    PyObject * rv = NULL;

    if(inst && keysCount >= 0 && keyStride > 0 && keySizesBuf.buf && keysBuf.buf && isAddedBoolmapBuf.buf) {
        biset * const s = (biset *)inst;
        const int * const keySizes = (int *)keySizesBuf.buf;
        const int * const keys = (int *)keysBuf.buf;
        bool * const isAddedBoolmap = (bool *)isAddedBoolmapBuf.buf;
        rv = Py_None;
    
        for(int i = 0; i < keysCount; i++) {
            const int keySize = keySizes[i];
    
            if(keySize > keyStride || keySize < 0) {
                PyErr_Format(PyExc_KeyError, "#%d, stride=%d, size=%d", i, keyStride, keySize);
                rv = NULL;
                break;
            }
            
            const int * const key = keys + i * keyStride;
            const std::vector<int> v(key, key + keySize);
            const auto it = s->insert(v);
            isAddedBoolmap[i] = it.second;
        }
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&keySizesBuf);
    PyBuffer_Release(&keysBuf);
    PyBuffer_Release(&isAddedBoolmapBuf);
    return rv;
}

static PyObject *
biset_add(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    int keySize = 0;
    Py_buffer keyBuf;
    
    if(!PyArg_ParseTuple(theArgs, "Liy*", &inst, &keySize, &keyBuf))
        return NULL;

    PyObject * rv = NULL;

    if(inst && keySize >= 0 && keyBuf.buf) {
        biset * const s = (biset *)inst;
        const int * const key = (int *)keyBuf.buf;
        const std::vector<int> v(key, key + keySize);
        rv = s->insert(v).second ? Py_True : Py_False;
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&keyBuf);
    return rv;
}

static PyObject *
biset_remove_many(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    int keysCount = 0;
    int keyStride = 0;
    Py_buffer keySizesBuf;
    Py_buffer keysBuf;
    Py_buffer isRemovedBoolmapBuf;
    
    if(!PyArg_ParseTuple(theArgs, "Liiy*y*y*", &inst, &keysCount, &keyStride, &keySizesBuf, &keysBuf, &isRemovedBoolmapBuf))
        return NULL;

    PyObject * rv = NULL;

    if(inst && keysCount >= 0 && keyStride > 0 && keySizesBuf.buf && keysBuf.buf && isRemovedBoolmapBuf.buf) {
        biset * const s = (biset *)inst;
        const int * const keySizes = (int *)keySizesBuf.buf;
        const int * const keys = (int *)keysBuf.buf;
        bool * const isRemovedBoolmap = (bool *)isRemovedBoolmapBuf.buf;
        rv = Py_None;
        
        for(int i = 0; i < keysCount; i++) {
            const int keySize = keySizes[i];
    
            if(keySize > keyStride || keySize < 0) {
                PyErr_Format(PyExc_KeyError, "#%d, stride=%d, size=%d", i, keyStride, keySize);
                rv = NULL;
                break;
            }
            
            const int * const key = keys + i * keyStride;
            const std::vector<int> v(key, key + keySize);
            const auto erased_count = s->erase(v);
            isRemovedBoolmap[i] = erased_count > 0;
        }
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&keySizesBuf);
    PyBuffer_Release(&keysBuf);
    PyBuffer_Release(&isRemovedBoolmapBuf);
    return rv;
}

static PyObject *
biset_remove(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    int keySize = 0;
    Py_buffer keyBuf;
    
    if(!PyArg_ParseTuple(theArgs, "Liy*", &inst, &keySize, &keyBuf))
        return NULL;

    PyObject * rv = NULL;

    if(inst && keySize >= 0 && keyBuf.buf) {
        biset * const s = (biset *)inst;
        const int * const key = (int *)keyBuf.buf;
        const std::vector<int> v(key, key + keySize);
        const auto erased_count = s->erase(v);

        if(erased_count <= 0) {
            PyErr_Format(PyExc_KeyError, "%s", get_key_repr(0, v).c_str());
        }
        else {
            rv = Py_None;
        }
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&keyBuf);
    return rv;
}

static PyObject *
biset_replace_many(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    int keysCount = 0;
    int keyStrideFrom = 0;
    int keyStrideTo = 0;
    Py_buffer keySizesFromBuf;
    Py_buffer keysFromBuf;
    Py_buffer keySizesToBuf;
    Py_buffer keysToBuf;
    Py_buffer isReplacedBoolmapBuf;
    
    if(!PyArg_ParseTuple(theArgs, "Liiy*y*iy*y*y*", &inst, &keysCount, &keyStrideFrom, &keySizesFromBuf, &keysFromBuf, &keyStrideTo, &keySizesToBuf, &keysToBuf, &isReplacedBoolmapBuf))
        return NULL;

    PyObject * rv = NULL;

    if(inst && keysCount >= 0 && 
        keyStrideFrom > 0 && keySizesFromBuf.buf && keysFromBuf.buf && 
        keyStrideTo > 0 && keySizesToBuf.buf && keysToBuf.buf && 
        isReplacedBoolmapBuf.buf) {
        
        biset * const s = (biset *)inst;
        const int * const keySizesFrom = (int *)keySizesFromBuf.buf;
        const int * const keysFrom = (int *)keysFromBuf.buf;
        const int * const keySizesTo = (int *)keySizesToBuf.buf;
        const int * const keysTo = (int *)keysToBuf.buf;
        bool * const isReplacedBoolmap = (bool *)isReplacedBoolmapBuf.buf;
        rv = Py_None;
        
        for(int i = 0; i < keysCount; i++) {
            const int keySizeFrom = keySizesFrom[i];
    
            if(keySizeFrom > keyStrideFrom || keySizeFrom < 0) {
                PyErr_Format(PyExc_KeyError, "keyFrom #%d, stride=%d, size=%d", i, keyStrideFrom, keySizeFrom);
                rv = NULL;
                break;
            }
            
            const int * const keyFrom = keysFrom + i * keyStrideFrom;
            const std::vector<int> vFrom(keyFrom, keyFrom + keySizeFrom);

            const int keySizeTo = keySizesTo[i];
    
            if(keySizeTo > keyStrideTo || keySizeTo < 0) {
                PyErr_Format(PyExc_KeyError, "keyTo #%d, stride=%d, size=%d", i, keyStrideTo, keySizeTo);
                rv = NULL;
                break;
            }
            
            const int * const keyTo = keysTo + i * keyStrideTo;
            const std::vector<int> vTo(keyTo, keyTo + keySizeTo);
            
            const auto itFrom = s->find(vFrom);
    
            if(itFrom != s->end()) {
                const auto itTo = s->find(vTo);
        
                if(itTo == s->end()) {
                    // keyFrom exists, but keyTo doesn't - normal case
                    s->erase(itFrom);
                    s->insert(vTo);
                    isReplacedBoolmap[i] = true;
                }
                else if(itFrom == itTo) {
                    // keyFrom exists, but keyTo exists and they are the same - normal case
                    isReplacedBoolmap[i] = true;
                }
                else {
                    // keyFrom exists, but keyTo exists as well as it's different - conflict or duplicate
                    isReplacedBoolmap[i] = false;
                }
            }
            else {
                PyErr_Format(PyExc_KeyError, "keyFrom %s", get_key_repr(i, vFrom).c_str());
                rv = NULL;
                break;
            }
        }
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }
    
    PyBuffer_Release(&keySizesFromBuf);
    PyBuffer_Release(&keysFromBuf);
    PyBuffer_Release(&keySizesToBuf);
    PyBuffer_Release(&keysToBuf);
    PyBuffer_Release(&isReplacedBoolmapBuf);
    return rv;
}

static PyObject *
biset_replace(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    int keySizeFrom = 0;
    int keySizeTo = 0;
    Py_buffer keyFromBuf;
    Py_buffer keyToBuf;
    
    if(!PyArg_ParseTuple(theArgs, "Liy*iy*", &inst, &keySizeFrom, &keyFromBuf, &keySizeTo, &keyToBuf))
        return NULL;

    PyObject * rv = NULL;
    
    if(inst && keySizeFrom >= 0 && keySizeTo >= 0 && keyFromBuf.buf && keyToBuf.buf) {
        biset * const s = (biset *)inst;
        const int * const keyFrom = (int *)keyFromBuf.buf;
        const int * const keyTo = (int *)keyToBuf.buf;
        const std::vector<int> vFrom(keyFrom, keyFrom + keySizeFrom);
        const std::vector<int> vTo(keyTo, keyTo + keySizeTo);
        const auto itFrom = s->find(vFrom);
    
        if(itFrom != s->end()) {
            const auto itTo = s->find(vTo);
    
            if(itTo == s->end()) {
                // keyFrom exists, but keyTo doesn't - normal case
                s->erase(itFrom);
                s->insert(vTo);
                rv = Py_True;
            }
            else if(itFrom == itTo) {
                // keyFrom exists, but keyTo exists and they are the same - normal case
                rv = Py_True;
            }
            else {
                // keyFrom exists, but keyTo exists as well as it's different - conflict or duplicate
                rv = Py_False;
            }
        }
        else {
            PyErr_Format(PyExc_KeyError, "keyFrom %s", get_key_repr(0, vFrom).c_str());
        }
    }
    else {
        PyErr_SetString(PyExc_Exception, "bad arguments");
    }

    PyBuffer_Release(&keyFromBuf);
    PyBuffer_Release(&keyToBuf);
    return rv;
}

static PyObject *
biset_clear(PyObject *, PyObject * theArgs) {
    long long inst = 0;
    
    if(!PyArg_ParseTuple(theArgs, "L", &inst))
        return NULL;

    if(!inst)
        return PyErr_Format(PyExc_Exception, "bad arguments");

    biset * const s = (biset *)inst;
    s->clear();
    Py_RETURN_NONE;
}

static PyMethodDef biset_methods[] = {
    {"biset_create", biset_create, METH_VARARGS, ""},
    {"biset_destroy", biset_destroy, METH_VARARGS, ""},
    {"biset_get_len", biset_get_len, METH_VARARGS, ""},
    {"biset_contains", biset_contains, METH_VARARGS, ""},
    {"biset_get", biset_get, METH_VARARGS, ""},
    {"biset_add_many", biset_add_many, METH_VARARGS, ""},
    {"biset_add", biset_add, METH_VARARGS, ""},
    {"biset_remove_many", biset_remove_many, METH_VARARGS, ""},
    {"biset_remove", biset_remove, METH_VARARGS, ""},
    {"biset_replace_many", biset_replace_many, METH_VARARGS, ""},
    {"biset_replace", biset_replace, METH_VARARGS, ""},
    {"biset_clear", biset_clear, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef biset_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "biset",
    .m_doc = "Bulk Int Set - implementation of set() for int-vectors with bulk methods",
    .m_size = 0,  
    .m_methods = biset_methods,
};

PyMODINIT_FUNC 
PyInit_biset(void) {
    return PyModuleDef_Init(&biset_module);
}

