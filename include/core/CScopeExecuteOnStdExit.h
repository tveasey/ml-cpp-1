/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CScopeExecuteOnStdExit_h
#define INCLUDED_ml_core_CScopeExecuteOnStdExit_h

#include <core/ImportExport.h>

#include <thread>
#include <vector>

namespace ml {
namespace core {

//! \brief This registers callbacks for the life time of the object in such a way
//! that they will get called on std::exit.
//!
//! IMPLEMENTATION:\n
//! If std::exit is called then the stack is not unwound and types which have
//! automatic storage do not have their destructors called. When a callback is
//! passed to the constructor of this type it is placed in a static registry
//! for the life time of the object. The destructor of this registry calls all
//! registered callbacks. Since objects with static storage are destructed when
//! std::exit is called this means any callback which is in the registry at the
//! time std::exit is called will be called.
//!
//! The number of callbacks registered is expected to be small since generally
//! very little needs to happen after exit is called. This therefore has one
//! global array which must be searched while holding a global lock when an
//! object is created.
class CORE_EXPORT CScopeExecuteOnStdExit {
public:
    using TCallback = std::function<void()>;

public:
    CScopeExecuteOnStdExit(TCallback callback);
    ~CScopeExecuteOnStdExit();
    CScopeExecuteOnStdExit(const CScopeExecuteOnStdExit&) = delete;
    CScopeExecuteOnStdExit& operator=(const CScopeExecuteOnStdExit&) = delete;

private:
    class CExecuteOnDestruct {
    public:
        CExecuteOnDestruct() = default;
        ~CExecuteOnDestruct() {
            if (m_Callback != nullptr) {
                m_Callback();
            }
        }
        CExecuteOnDestruct(const CExecuteOnDestruct&) = delete;
        CExecuteOnDestruct& operator=(const CExecuteOnDestruct&) = delete;
        CExecuteOnDestruct(CExecuteOnDestruct&&) = default;
        CExecuteOnDestruct& operator=(CExecuteOnDestruct&&) = default;

        void reset(TCallback callback = nullptr) {
            m_Callback = std::move(callback);
        }
        bool isEmpty() const { return m_Callback == nullptr; }

    private:
        TCallback m_Callback = nullptr;
    };
    using TExplicitlyDestroyVec = std::vector<CExecuteOnDestruct>;

private:
    static std::mutex ms_RegistryLock;
    static TExplicitlyDestroyVec ms_Registry;
    std::size_t m_Slot;
};

std::mutex CScopeExecuteOnStdExit::ms_RegistryLock;
CScopeExecuteOnStdExit::TExplicitlyDestroyVec CScopeExecuteOnStdExit::ms_Registry;
}
}

#endif //INCLUDED_ml_core_CScopeExecuteOnStdExit_h
