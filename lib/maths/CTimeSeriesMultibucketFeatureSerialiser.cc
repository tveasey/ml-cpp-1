/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesMultibucketFeatureSerialiser.h>

#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CTimeSeriesMultibucketFeatures.h>

namespace ml {
namespace maths {
namespace {
const std::string UNIVARIATE_MEAN_TAG{"a"};
const std::string MULTIVARIATE_MEAN_TAG{"b"};
}

bool CTimeSeriesMultibucketFeatureSerialiser::
operator()(TScalarFeaturePtr& result, core::CStateRestoreTraverser& traverser) const {
    do {
        const std::string& name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(
            UNIVARIATE_MEAN_TAG,
            result = std::make_unique<CTimeSeriesMultibucketScalarMean>(),
            traverser.traverseSubLevel(std::bind<bool>(
                &TScalarFeature::acceptRestoreTraverser, result.get(), std::placeholders::_1)),
            /**/)
    } while (traverser.next());
    return true;
}

bool CTimeSeriesMultibucketFeatureSerialiser::
operator()(TVectorFeaturePtr& result, core::CStateRestoreTraverser& traverser) const {
    do {
        const std::string& name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(
            MULTIVARIATE_MEAN_TAG,
            result = std::make_unique<CTimeSeriesMultibucketVectorMean>(),
            traverser.traverseSubLevel(std::bind<bool>(
                &TVectorFeature::acceptRestoreTraverser, result.get(), std::placeholders::_1)),
            /**/)
    } while (traverser.next());
    return true;
}

void CTimeSeriesMultibucketFeatureSerialiser::
operator()(const TScalarFeaturePtr& feature, core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesMultibucketScalarMean*>(feature.get()) != nullptr) {
        inserter.insertLevel(UNIVARIATE_MEAN_TAG,
                             std::bind(&TScalarFeature::acceptPersistInserter,
                                       feature.get(), std::placeholders::_1));
    } else {
        LOG_ERROR(<< "Feature with type '" << typeid(feature).name() << "' has no defined name");
    }
}

void CTimeSeriesMultibucketFeatureSerialiser::
operator()(const TVectorFeaturePtr& feature, core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesMultibucketVectorMean*>(feature.get()) != nullptr) {
        inserter.insertLevel(MULTIVARIATE_MEAN_TAG,
                             std::bind(&TVectorFeature::acceptPersistInserter,
                                       feature.get(), std::placeholders::_1));
    } else {
        LOG_ERROR(<< "Feature with type '" << typeid(feature).name() << "' has no defined name");
    }
}
}
}
