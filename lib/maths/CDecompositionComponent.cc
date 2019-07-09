/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDecompositionComponent.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CIntegerTools.h>
#include <maths/CSampling.h>
#include <maths/CSeasonalTime.h>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

#include <ios>
#include <vector>

namespace ml {
namespace maths {
namespace {

using TDoubleDoublePr = maths_t::TDoubleDoublePr;

const std::string MAX_SIZE_TAG{"a"};
const std::string RNG_TAG{"b"};
const std::string BOUNDARY_CONDITION_TAG{"c"};
const std::string BUCKETING_TAG{"d"};
const std::string SPLINES_TAG{"e"};

// Nested tags
const std::string ESTIMATED_TAG{"a"};
const std::string KNOTS_TAG{"b"};
const std::string VALUES_TAG{"c"};
const std::string VARIANCES_TAG{"d"};

const std::string EMPTY_STRING;
}

CDecompositionComponent::CDecompositionComponent(std::size_t maxSize,
                                                 CSplineTypes::EBoundaryCondition boundaryCondition,
                                                 CSplineTypes::EType valueInterpolationType,
                                                 CSplineTypes::EType varianceInterpolationType)
    : m_MaxSize{maxSize}, m_BoundaryCondition{boundaryCondition}, m_Splines{valueInterpolationType,
                                                                            varianceInterpolationType} {
}

bool CDecompositionComponent::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(MAX_SIZE_TAG, m_MaxSize)
        RESTORE_SETUP_TEARDOWN(
            BOUNDARY_CONDITION_TAG, int boundaryCondition,
            core::CStringUtils::stringToType(traverser.value(), boundaryCondition),
            m_BoundaryCondition = static_cast<CSplineTypes::EBoundaryCondition>(boundaryCondition))
        RESTORE(SPLINES_TAG, traverser.traverseSubLevel(std::bind(
                                 &CPackedSplines::acceptRestoreTraverser, &m_Splines,
                                 m_BoundaryCondition, std::placeholders::_1)))
    } while (traverser.next());

    if (this->initialized()) {
        m_MeanValue = this->valueSpline().mean();
    }

    return true;
}

void CDecompositionComponent::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(MAX_SIZE_TAG, m_MaxSize);
    inserter.insertValue(BOUNDARY_CONDITION_TAG, static_cast<int>(m_BoundaryCondition));
    inserter.insertLevel(SPLINES_TAG, std::bind(&CPackedSplines::acceptPersistInserter,
                                                &m_Splines, std::placeholders::_1));
}

void CDecompositionComponent::swap(CDecompositionComponent& other) {
    std::swap(m_MaxSize, other.m_MaxSize);
    std::swap(m_BoundaryCondition, other.m_BoundaryCondition);
    std::swap(m_MeanValue, other.m_MeanValue);
    m_Splines.swap(other.m_Splines);
}

bool CDecompositionComponent::initialized() const {
    return m_Splines.initialized();
}

void CDecompositionComponent::clear() {
    if (m_Splines.initialized()) {
        m_Splines.clear();
    }
    m_MeanValue = 0.0;
}

void CDecompositionComponent::interpolate(const TDoubleVec& knots,
                                          const TDoubleVec& values,
                                          const TDoubleVec& variances,
                                          const TDoubleVec& logValues,
                                          const TDoubleVec& logVariances) {
    m_Splines.interpolate(knots, values, variances, logValues, logVariances, m_BoundaryCondition);
    m_MeanValue = this->valueSpline().mean();
}

void CDecompositionComponent::shiftLevel(double shift) {
    m_Splines.shift(shift);
    m_MeanValue += shift;
}

TDoubleDoublePr CDecompositionComponent::value(double offset, double n, double confidence) const {
    // In order to compute a confidence interval we need to know
    // the distribution of the samples. In practice, as long as
    // they are independent, then the sample mean will be
    // asymptotically normal with mean equal to the sample mean
    // and variance equal to the sample variance divided by root
    // of the number of samples.
    if (this->initialized()) {
        double m{this->valueSpline().value(offset)};

        if (confidence == 0.0) {
            return {m, m};
        }

        n = std::max(n, 1.0);
        double sd{::sqrt(std::max(this->varianceSpline().value(offset), 0.0) / n)};
        if (sd == 0.0) {
            return {m, m};
        }

        try {
            boost::math::normal normal{m, sd};
            double ql{boost::math::quantile(normal, (100.0 - confidence) / 200.0)};
            double qu{boost::math::quantile(normal, (100.0 + confidence) / 200.0)};
            return {ql, qu};
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed calculating confidence interval: " << e.what()
                      << ", n = " << n << ", m = " << m << ", sd = " << sd
                      << ", confidence = " << confidence);
        }
        return {m, m};
    }

    return {m_MeanValue, m_MeanValue};
}

double CDecompositionComponent::meanValue() const {
    return m_MeanValue;
}

std::size_t CDecompositionComponent::maxSize() const {
    return std::max(m_MaxSize, MIN_MAX_SIZE);
}

CSplineTypes::EBoundaryCondition CDecompositionComponent::boundaryCondition() const {
    return m_BoundaryCondition;
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::valueSpline() const {
    return m_Splines.spline(CPackedSplines::E_Value);
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::varianceSpline() const {
    return m_Splines.spline(CPackedSplines::E_Variance);
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::logValueSpline() const {
    return m_Splines.spline(CPackedSplines::E_LogValue);
}

CDecompositionComponent::TSplineCRef CDecompositionComponent::logVarianceSpline() const {
    return m_Splines.spline(CPackedSplines::E_LogVariance);
}

uint64_t CDecompositionComponent::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_MaxSize);
    seed = CChecksum::calculate(seed, m_BoundaryCondition);
    seed = CChecksum::calculate(seed, m_Splines);
    return CChecksum::calculate(seed, m_MeanValue);
}

const CDecompositionComponent::CPackedSplines& CDecompositionComponent::splines() const {
    return m_Splines;
}

const std::size_t CDecompositionComponent::MIN_MAX_SIZE{1};

////// CDecompositionComponent::CPackedSplines //////

CDecompositionComponent::CPackedSplines::CPackedSplines(CSplineTypes::EType valueInterpolationType,
                                                        CSplineTypes::EType varianceInterpolationType) {
    m_Types[E_Value] = valueInterpolationType;
    m_Types[E_LogValue] = valueInterpolationType;
    m_Types[E_Variance] = varianceInterpolationType;
    m_Types[E_LogVariance] = varianceInterpolationType;
}

bool CDecompositionComponent::CPackedSplines::acceptRestoreTraverser(
    CSplineTypes::EBoundaryCondition boundary,
    core::CStateRestoreTraverser& traverser) {
    int estimated{0};
    TDoubleVec knots;
    TDoubleVec values;
    TDoubleVec variances;
    TDoubleVec logValues;
    TDoubleVec logVariances;

    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(ESTIMATED_TAG, estimated)
        RESTORE(KNOTS_TAG, core::CPersistUtils::fromString(traverser.value(), knots))
        RESTORE(VALUES_TAG, core::CPersistUtils::fromString(traverser.value(), values))
        RESTORE(VARIANCES_TAG, core::CPersistUtils::fromString(traverser.value(), variances))
        RESTORE(LOG_VALUES_TAG, core::CPersistUtils::fromString(traverser.value(), logValues))
        RESTORE(LOG_VARIANCES_TAG, core::CPersistUtils::fromString(traverser.value(), logVariances))
    } while (traverser.next());

    if (estimated == 1) {
        this->interpolate(knots, values, variances, logValues, logVariances, boundary);
    }

    return true;
}

void CDecompositionComponent::CPackedSplines::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(ESTIMATED_TAG, static_cast<int>(this->initialized()));
    if (this->initialized()) {
        inserter.insertValue(KNOTS_TAG, core::CPersistUtils::toString(m_Knots));
        inserter.insertValue(VALUES_TAG, core::CPersistUtils::toString(m_Values[0]));
        inserter.insertValue(VARIANCES_TAG, core::CPersistUtils::toString(m_Values[1]));
        inserter.insertValue(LOG_VALUES_TAG, core::CPersistUtils::toString(m_Values[2]));
        inserter.insertValue(LOG_VARIANCES_TAG, core::CPersistUtils::toString(m_Values[3]));
    }
}

void CDecompositionComponent::CPackedSplines::swap(CPackedSplines& other) {
    std::swap(m_Types, other.m_Types);
    m_Knots.swap(other.m_Knots);
    for (int i : {E_Value, E_Variance, E_LogValue, E_LogVariance}) {
        m_Values[i].swap(other.m_Values[i]);
        m_Curvatures[i].swap(other.m_Curvatures[i]);
    }
}

bool CDecompositionComponent::CPackedSplines::initialized() const {
    return m_Knots.size() > 0;
}

void CDecompositionComponent::CPackedSplines::clear() {
    for (int i : {E_Value, E_Variance, E_LogValue, E_LogVariance}) {
        this->spline(i).clear();
    }
}

void CDecompositionComponent::CPackedSplines::shift(double shift) {
    for (auto& value : m_Values[E_Value]) {
        value += shift;
    }
    for (auto& value : m_Values[E_LogValue]) {
        value += shift;
    }
}

CDecompositionComponent::TSplineCRef
CDecompositionComponent::CPackedSplines::spline(ESpline spline) const {
    return TSplineCRef(m_Types[spline], std::cref(m_Knots),
                       std::cref(m_Values[spline]),
                       std::cref(m_Curvatures[spline]));
}

CDecompositionComponent::TSplineRef
CDecompositionComponent::CPackedSplines::spline(ESpline spline) {
    return TSplineRef(m_Types[spline], std::ref(m_Knots),
                      std::ref(m_Values[spline]),
                      std::ref(m_Curvatures[spline]));
}

const CDecompositionComponent::TFloatVec& CDecompositionComponent::CPackedSplines::knots() const {
    return m_Knots;
}

void CDecompositionComponent::CPackedSplines::interpolate(const TDoubleVec& knots,
                                                          const TDoubleVec& values,
                                                          const TDoubleVec& variances,
                                                          const TDoubleVec& logValues,
                                                          const TDoubleVec& logVariances,
                                                          CSplineTypes::EBoundaryCondition boundary) {
    CPackedSplines oldSplines{m_Types[0], m_Types[1]};
    this->swap(oldSplines);

    TSplineRef valueSpline{this->spline(E_Value)};
    TSplineRef varianceSpline{this->spline(E_Variance)};
    TSplineRef logValueSpline{this->spline(E_LogValue)};
    TSplineRef logVarianceSpline{this->spline(E_LogVariance)};

    if (valueSpline.interpolate(knots, values, boundary) == false ||
        varianceSpline.interpolate(knots, variances, boundary) == false ||
        logValueSpline.interpolate(knots, logValues, boundary) == false ||
        logVarianceSpline.interpolate(knots, logVariances, boundary) == false) {

        this->swap(oldSplines);
    }
    LOG_TRACE(<< "types = " << core::CContainerPrinter::print(m_Types));
    LOG_TRACE(<< "knots = " << core::CContainerPrinter::print(m_Knots));
    LOG_TRACE(<< "values = " << core::CContainerPrinter::print(m_Values));
    LOG_TRACE(<< "curvatures = " << core::CContainerPrinter::print(m_Curvatures));
}

uint64_t CDecompositionComponent::CPackedSplines::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Types);
    seed = CChecksum::calculate(seed, m_Knots);
    seed = CChecksum::calculate(seed, m_Values);
    return CChecksum::calculate(seed, m_Curvatures);
}

void CDecompositionComponent::CPackedSplines::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CPackedSplines");
    core::CMemoryDebug::dynamicSize("m_Knots", m_Knots, mem);
    for (int i : {E_Value, E_Variance, E_LogValue, E_LogVariance}) {
        core::CMemoryDebug::dynamicSize("m_Values[" + std::to_string(i) + "]", m_Values[i], mem);
        core::CMemoryDebug::dynamicSize("m_Curvatures[" + std::to_string(i) + "]", m_Curvatures[i], mem);
    }
}

std::size_t CDecompositionComponent::CPackedSplines::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_Knots)};
    for (int i : {E_Value, E_Variance, E_LogValue, E_LogVariance}) {
        mem += core::CMemory::dynamicSize(m_Values[i]);
        mem += core::CMemory::dynamicSize(m_Curvatures[i]);
    }
    return mem;
}
}
}
