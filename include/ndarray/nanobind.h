/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef NDARRAY_nanobind_h_INCLUDED
#define NDARRAY_nanobind_h_INCLUDED

/**
 *  @file ndarray/nanobind.h
 *  @brief Public header file for pybind11-based Python support.
 *
 *  \warning Both the Numpy C-API headers "arrayobject.h" and
 *  "ufuncobject.h" must be included before ndarray/python.hpp
 *  or any of the files in ndarray/python.
 *
 *  \note This file is not included by the main "ndarray.h" header file.
 */

/** \defgroup ndarrayPythonGroup Python Support
 *
 *  The ndarray Python support module provides conversion
 *  functions between ndarray objects, notably Array and
 *  Vector, and Python Numpy objects.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/operators.h>

#include "ndarray.h"
#include "nanobind/stl/detail/nb_array.h"
#include "nanobind/stl/detail/nb_list.h"
#include "ndarray/eigen.h"
#include "ndarray/Array.h"

#include <pybind11/numpy.h>
#include <iostream>

namespace nb = nanobind;
namespace ndarray {

    namespace detail {

        inline void destroyCapsule(PyObject *p) {
            void *m = PyCapsule_GetPointer(p, "ndarray.Manager");
            Manager::Ptr *b = reinterpret_cast<Manager::Ptr *>(m);
            delete b;
        }

    } // namespace ndarray::detail

    inline PyObject *makePyManager(Manager::Ptr const &m) {
        return PyCapsule_New(
                new Manager::Ptr(m),
                "ndarray.Manager",
                detail::destroyCapsule
        );
    }


    using nanobind_np_size_t = ssize_t;

    template<typename T, int N, int C>
    struct
#ifdef __GNUG__
// pybind11 hides all symbols in its namespace only when this is set,
// and in that case we should hide these classes too.
            __attribute__((visibility("hidden")))
#endif
    NanobindHelper {
    };

} // namespace ndarray


NAMESPACE_BEGIN(NB_NAMESPACE)
    NAMESPACE_BEGIN(detail)
        struct managed_dltensor {
            dlpack::dltensor dltensor;
            void *manager_ctx;
            void (*deleter)(managed_dltensor *);
        };

        struct ndarray_handle {
            managed_dltensor *ndarray;
            std::atomic<size_t> refcount;
            PyObject *owner, *self;
            bool free_shape;
            bool free_strides;
            bool call_deleter;
            bool ro;
        };
        template<typename T, int N, int C>
        struct type_caster<::ndarray::Array<T,N,C>> {
            using Value = ::ndarray::Array<T, N, C>;
            static constexpr auto Name =
                    const_name("ndarray")  + const_name("[") + concat_maybe(detail::ndarray_arg<T>::name) +
                    const_name("]");
            template<typename T_> using Cast = movable_cast_t<T_>;
#if 0
            static handle from_cpp(Value *p, rv_policy policy, cleanup_list *list) {
                if (!p)return none().release();
                return from_cpp(*p, policy, list);
            }
#endif
            void set_value() {
               //value = _helper.convert();
            }

            explicit operator Value *() {
              //  if (_helper.isNone) {
             //       return nullptr;
              //  } else {
             //       set_value();
              //      return &value;
               // }
               return 0;
            }

            explicit operator Value &() { return (Value &) value; }

            explicit operator Value &&() { return (Value &&) value; }

            using Helper = ::ndarray::NanobindHelper<T, N, C>;

            bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
                std::cout << "from_python" << std::endl;
                using Wrapper = nb::ndarray<> ;
                nb::object object;
                bool isNone = src.is_none();
                if (isNone) {
                    return true;
                }
                try {
                   object  = borrow(src);
                } catch (...) {
                    return false;
                }
                print(object);
                return true;//_helper.init(src) && _helper.check();
            }

            static nb::handle from_cpp(const ::ndarray::Array<T, N, C> &src, rv_policy policy,
                                   cleanup_list *cleanup) noexcept {
                using ArrayType = nb::ndarray<nb::numpy, typename std::remove_const_t<T>> ;
                using Array = std::conditional_t<std::is_const_v<T>, const ArrayType, ArrayType>;
                using Element =  typename std::remove_const_t<T>;
                ::ndarray::Vector<::ndarray::Size,N> nShape = src.getShape();
                ::ndarray::Vector<::ndarray::Offset,N> nStrides = src.getStrides();
                std::vector<size_t> pShape(N);
                std::vector<int64_t> pStrides(N);
                for (int i = 0; i < N; ++i) {
                    pShape[i] = nShape[i];
                    pStrides[i] = nStrides[i] ;
                }
                nb::object base = nb::object();
                if (src.getManager()) {
                    base = nb::steal<nb::object>(::ndarray::makePyManager(src.getManager()));
                }
               Array array((Element*)src.getData(), N, pShape.data(), base, pStrides.data());
                if (std::is_const_v<T>) {
                   //base.attr("flags")["WRITEABLE"] = false;
                }
              nb::handle result = ndarray_wrap(array.handle(),  int(nanobind::detail::ndarray_framework::numpy),policy, cleanup);
              return result;
            }
        private:
            ::ndarray::Array<T, N, C> value;
        };
    NAMESPACE_END(detail);
NAMESPACE_END(NB_NAMESPACE);

#endif