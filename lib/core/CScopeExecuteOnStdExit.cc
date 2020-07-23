/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CScopeExecuteOnStdExit.h>

namespace ml {
namespace core {

CScopeExecuteOnStdExit::CScopeExecuteOnStdExit(TCallback callback) {
    std::unique_lock<std::mutex> lock{ms_RegistryLock};
    auto slot = ms_Registry.begin();
    for (/**/; slot != ms_Registry.end(); ++slot) {
        if (slot->isEmpty()) {
            break;
        }
    }
    if (slot == ms_Registry.end()) {
        ms_Registry.resize(ms_Registry.size() + 1);
        slot = ms_Registry.end() - 1;
    }
    slot->reset(std::move(callback));
}

CScopeExecuteOnStdExit::~CScopeExecuteOnStdExit() {
    std::unique_lock<std::mutex> lock{ms_RegistryLock};
    ms_Registry[m_Slot].reset();
}
}
}
