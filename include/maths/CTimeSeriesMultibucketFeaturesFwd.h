/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesMultibucketFeaturesFwd_h
#define INCLUDED_ml_maths_CTimeSeriesMultibucketFeaturesFwd_h

#include <core/CoreTypes.h>

#include <maths/CLinearAlgebraFwd.h>
#include <maths/MathsTypes.h>

#include <functional>

namespace ml {
namespace core {
template<typename, std::size_t>
class CSmallVector;
}
namespace maths {
template<typename, typename>
class CTimeSeriesMultibucketFeature;
using CTimeSeriesMultibucketScalarFeature =
    CTimeSeriesMultibucketFeature<double, std::function<double(core_t::TTime)>>;
using CTimeSeriesMultibucketVectorFeature =
    CTimeSeriesMultibucketFeature<core::CSmallVector<double, 10>,
                                  std::function<CVector<CFloatStorage> (core_t::TTime)>>;
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesMultibucketFeaturesFwd_h
