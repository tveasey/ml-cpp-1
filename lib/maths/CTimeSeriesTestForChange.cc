
/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesTestForChange.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCalendarComponent.h>
#include <maths/CSeasonalComponent.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesSegmentation.h>
#include <maths/CTools.h>
#include <maths/CTrendComponent.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TSegmentation = CTimeSeriesSegmentation;

double rightTailFTest(double v0, double v1, double df0, double df1) {
    if (df1 <= 0.0) {
        return 1.0;
    }
    double F{v0 == v1 ? 1.0 : (v0 / df0) / (v1 / df1)};
    return CStatisticalTests::rightTailFTest(F, df0, df1);
}

const std::size_t H0{0};
const std::size_t H1{1};
const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime HOUR{core::constants::HOUR};
const std::string NO_CHANGE{"no change"};
}

void CLevelShift::apply(CTrendComponent& component) const {
    component.shiftLevel(this->time(), m_ValueAtShift, m_Shift);
}

const std::string& CLevelShift::type() const {
    return TYPE;
}

const std::string CLevelShift::TYPE{"level shift"};

void CScale::apply(CTrendComponent& component) const {
    component.linearScale(m_Scale);
}

void CScale::apply(CSeasonalComponent& component) const {
    component.linearScale(this->time(), m_Scale);
}

void CScale::apply(CCalendarComponent& component) const {
    component.linearScale(this->time(), m_Scale);
}

const std::string& CScale::type() const {
    return TYPE;
}

const std::string CScale::TYPE{"scale"};

void CTimeShift::apply(CTimeSeriesDecomposition& decomposition) const {
    decomposition.shiftTime(m_Shift);
}

const std::string& CTimeShift::type() const {
    return TYPE;
}

const std::string CTimeShift::TYPE{"time shift"};

CTimeSeriesTestForChange::CTimeSeriesTestForChange(core_t::TTime valuesStartTime,
                                                   core_t::TTime bucketStartTime,
                                                   core_t::TTime bucketLength,
                                                   core_t::TTime predictionInterval,
                                                   TPredictor predictor,
                                                   TFloatMeanAccumulatorVec values,
                                                   double outlierFraction)
    : m_ValuesStartTime{valuesStartTime}, m_BucketStartTime{bucketStartTime}, m_BucketLength{bucketLength},
      m_PredictionInterval{predictionInterval}, m_OutlierFraction{outlierFraction},
      m_Predictor{std::move(predictor)}, m_Values{std::move(values)},
      m_Outliers{static_cast<std::size_t>(std::max(
          outlierFraction * static_cast<double>(CSignal::countNotMissing(m_Values)) + 0.5,
          1.0))} {

    TMeanVarAccumulator moments{this->truncatedMoments(m_OutlierFraction, m_Values)};
    TMeanVarAccumulator meanAbs{this->truncatedMoments(
        m_OutlierFraction, m_Values, [](const TFloatMeanAccumulator& value) {
            return std::fabs(CBasicStatistics::mean(value));
        })};

    // Note we don't bother modelling changes whose size is too small compared
    // to the absolute values. We won't raise anomalies for differences from our
    // predictions which are smaller than this anyway.
    m_EpsVariance = std::max(
        CTools::pow2(1000.0 * std::numeric_limits<double>::epsilon()) *
            CBasicStatistics::maximumLikelihoodVariance(moments),
        CTools::pow2(MINIMUM_COEFFICIENT_OF_VARIATION * CBasicStatistics::mean(meanAbs)));
    LOG_TRACE(<< "eps variance = " << m_EpsVariance);
}

CTimeSeriesTestForChange::TShockUPtr CTimeSeriesTestForChange::test() const {

    using TShockVec = std::vector<SShock>;

    TShockVec shocks;
    shocks.reserve(3);
    shocks.push_back(this->levelShift());
    shocks.push_back(this->scale());
    shocks.push_back(this->timeShift());

    shocks.erase(std::remove_if(
                     shocks.begin(), shocks.end(),
                     [](const auto& shock) { return shock.s_Type == E_NoShock; }),
                 shocks.end());
    LOG_TRACE(<< "# shocks = " << shocks.size());

    if (shocks.size() > 0) {
        std::stable_sort(shocks.begin(), shocks.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.s_NumberParameters < rhs.s_NumberParameters;
        });

        // If there is strong evidence for a more complex explanation select that
        // otherwise fallback to AIC.

        double selectedEvidence{aic(shocks[0])};
        std::size_t selected{0};
        LOG_TRACE(<< print(shocks[0].s_Type) << " evidence = " << selectedEvidence);

        double n{static_cast<double>(CSignal::countNotMissing(m_Values))};
        for (std::size_t candidate = 1; candidate < shocks.size(); ++candidate) {
            double pValue{std::min(
                rightTailFTest(shocks[selected].s_TruncatedResidualVariance,
                               shocks[candidate].s_TruncatedResidualVariance,
                               (1.0 - m_OutlierFraction) * n - shocks[selected].s_NumberParameters,
                               (1.0 - m_OutlierFraction) * n - shocks[candidate].s_NumberParameters),
                rightTailFTest(shocks[selected].s_ResidualVariance,
                               shocks[candidate].s_ResidualVariance,
                               n - shocks[selected].s_NumberParameters,
                               n - shocks[candidate].s_NumberParameters))};
            double evidence{aic(shocks[H1])};
            LOG_TRACE(<< print(shocks[H1].s_Type) << " p-value = " << pValue
                      << ", evidence = " << evidence);
            if (pValue < m_SignificantPValue || evidence < selectedEvidence) {
                std::tie(selectedEvidence, selected) = std::make_pair(evidence, candidate);
            }
        }

        switch (shocks[selected].s_Type) {
        case E_LevelShift:
            return std::make_unique<CLevelShift>(shocks[selected].s_Time,
                                                 shocks[selected].s_ValueAtChange,
                                                 shocks[selected].s_LevelShift);
        case E_Scale:
            return std::make_unique<CScale>(shocks[selected].s_Time,
                                            shocks[selected].s_Scale);
        case E_TimeShift:
            return std::make_unique<CTimeShift>(shocks[selected].s_Time,
                                                shocks[selected].s_TimeShift);
        case E_NoShock:
            LOG_ERROR(<< "Unexpected type");
            break;
        }
    }

    return {};
}

CTimeSeriesTestForChange::SShock CTimeSeriesTestForChange::levelShift() const {

    // Test for piecewise linear shift. We use a hypothesis test against a null
    // hypothesis that there is a quadratic trend.

    using TRegression = CLeastSquaresOnlineRegression<2, double>;

    m_ValuesMinusPredictions = this->removePredictions(this->bucketPredictor(), m_Values);

    TSizeVec trendSegments{TSegmentation::piecewiseLinear(
        m_ValuesMinusPredictions, m_SignificantPValue, m_OutlierFraction)};
    LOG_TRACE(<< "trend segments = " << core::CContainerPrinter::print(trendSegments));

    if (trendSegments.size() > 2) {
        double n{(1.0 - m_OutlierFraction) *
                 static_cast<double>(CSignal::countNotMissing(m_ValuesMinusPredictions))};
        double truncatedVariances[2];
        double degreesFreedom[2];

        TDoubleVec shifts;
        auto residuals = TSegmentation::removePiecewiseLinear(
            m_ValuesMinusPredictions, trendSegments, m_OutlierFraction, &shifts);
        double variance{CBasicStatistics::maximumLikelihoodVariance(
            this->truncatedMoments(0.0, residuals))};
        truncatedVariances[H1] = CBasicStatistics::maximumLikelihoodVariance(
            this->truncatedMoments(m_OutlierFraction, residuals));
        degreesFreedom[H1] = n - 2.0 * static_cast<double>(trendSegments.size() - 1);
        LOG_TRACE(<< "shifts = " << core::CContainerPrinter::print(shifts));
        LOG_TRACE(<< "variance = " << variance);
        LOG_TRACE(<< "truncated variance = " << truncatedVariances[H1]);

        TRegression trend;
        auto predictor = [&](std::size_t i) {
            return trend.predict(static_cast<double>(i));
        };
        for (std::size_t i = 0; i < 2; ++i) {
            CSignal::reweightOutliers(predictor, m_OutlierFraction, m_ValuesMinusPredictions);
            for (std::size_t j = 0; j < m_ValuesMinusPredictions.size(); ++j) {
                trend.add(static_cast<double>(j),
                          CBasicStatistics::mean(m_ValuesMinusPredictions[j]),
                          CBasicStatistics::count(m_ValuesMinusPredictions[j]));
            }
        }
        truncatedVariances[H0] = CBasicStatistics::maximumLikelihoodVariance(this->truncatedMoments(
            m_OutlierFraction,
            this->removePredictions(predictor, std::move(m_ValuesMinusPredictions))));
        degreesFreedom[H0] = n - 3.0;
        double pValue{rightTailFTest(truncatedVariances[H0], truncatedVariances[H1],
                                     degreesFreedom[H0], degreesFreedom[H1])};
        LOG_TRACE(<< "p-value vs quadratic trend = " << pValue);

        if (pValue < m_AcceptedFalsePostiveRate) {
            std::size_t changeIndex{trendSegments[*std::max_element(
                boost::make_counting_iterator(std::size_t{1}),
                boost::make_counting_iterator(shifts.size()), [&](auto lhs, auto rhs) {
                    return std::fabs(shifts[lhs] - shifts[lhs - 1]) <
                           std::fabs(shifts[rhs] - shifts[rhs - 1]);
                })]};
            core_t::TTime changeTime{this->changeTime(changeIndex)};
            double changeValue{this->changeValue(changeIndex)};
            LOG_TRACE(<< "change index = " << changeIndex
                      << ", time = " << changeTime << ", value = " << changeValue);

            SShock shock{E_LevelShift,
                         changeTime,
                         changeValue,
                         variance,
                         truncatedVariances[H1],
                         2.0 * static_cast<double>(trendSegments.size() - 1)};
            shock.s_LevelShift = shifts.back();

            return shock;
        }
    }

    return {};
}

CTimeSeriesTestForChange::SShock CTimeSeriesTestForChange::scale() const {

    // Test for linear scales of the base predictor.

    auto predictor = this->bucketPredictor();

    TSizeVec scaleSegments{TSegmentation::piecewiseLinearScaledSeasonal(
        m_Values, predictor, 1, m_SignificantPValue)};
    LOG_TRACE(<< "scale segments = " << core::CContainerPrinter::print(scaleSegments));

    if (scaleSegments.size() > 2) {
        TDoubleVec scales;
        auto residuals = TSegmentation::removePiecewiseLinearScaledSeasonal(
            m_Values, predictor, scaleSegments, m_OutlierFraction, &scales);
        double variance{CBasicStatistics::maximumLikelihoodVariance(
            this->truncatedMoments(0.0, residuals))};
        double truncatedVariance{CBasicStatistics::maximumLikelihoodVariance(
            this->truncatedMoments(m_OutlierFraction, residuals))};
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
        LOG_TRACE(<< "variance = " << variance);
        LOG_TRACE(<< "truncated variance = " << truncatedVariance);

        std::size_t changeIndex{
            scaleSegments[std::max_element(scales.begin(), scales.end(),
                                           [](auto lhs, auto rhs) {
                                               return std::max(lhs, 1.0 / lhs) <
                                                      std::max(rhs, 1.0 / rhs);
                                           }) -
                          scales.begin()]};
        core_t::TTime changeTime{this->changeTime(changeIndex)};
        double changeValue{this->changeValue(changeIndex)};
        LOG_TRACE(<< "change index = " << changeIndex
                  << ", time = " << changeTime << ", value = " << changeValue);

        SShock shock{
            E_Scale,           changeTime,
            changeValue,       variance,
            truncatedVariance, static_cast<double>(scaleSegments.size() - 1)};
        shock.s_Scale = scales.back();

        return shock;
    }

    return {};
}

CTimeSeriesTestForChange::SShock CTimeSeriesTestForChange::timeShift() const {

    // Test for time shifts of the base predictor.

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    auto predictor = [this](core_t::TTime time) {
        TMeanAccumulator result;
        for (core_t::TTime offset = 0; offset < m_BucketLength; offset += m_PredictionInterval) {
            result.add(m_Predictor(m_BucketStartTime + time + offset));
        }
        return CBasicStatistics::mean(result);
    };

    TSegmentation::TTimeVec candidateShifts;
    for (core_t::TTime shift = -6 * HOUR; shift < 0; shift += HALF_HOUR) {
        candidateShifts.push_back(shift);
    }
    for (core_t::TTime shift = HALF_HOUR; shift <= 6 * HOUR; shift += HALF_HOUR) {
        candidateShifts.push_back(shift);
    }

    TSegmentation::TTimeVec shifts;
    TSizeVec shiftSegments{TSegmentation::piecewiseTimeShifted(
        m_Values, m_BucketLength, candidateShifts, predictor, 1, m_SignificantPValue, &shifts)};
    LOG_TRACE(<< "shift segments = " << core::CContainerPrinter::print(shiftSegments));
    LOG_TRACE(<< "shifts = " << core::CContainerPrinter::print(shifts));

    if (shiftSegments.size() > 2) {
        auto shiftedPredictor = [&](std::size_t i) {
            return m_Predictor(m_ValuesStartTime +
                               m_BucketLength * static_cast<core_t::TTime>(i) +
                               TSegmentation::shiftAt(i, shiftSegments, shifts));
        };
        auto residuals = removePredictions(shiftedPredictor, m_Values);
        double variance{CBasicStatistics::maximumLikelihoodVariance(
            this->truncatedMoments(0.0, residuals))};
        double truncatedVariance{CBasicStatistics::maximumLikelihoodVariance(
            this->truncatedMoments(m_OutlierFraction, residuals))};
        LOG_TRACE(<< "variance = " << variance);
        LOG_TRACE(<< "truncated variance = " << truncatedVariance);

        std::size_t changeIndex{
            shiftSegments[std::max_element(shifts.begin(), shifts.end(),
                                           [](auto lhs, auto rhs) {
                                               return std::abs(lhs) < std::abs(rhs);
                                           }) -
                          shifts.begin()]};
        core_t::TTime changeTime{this->changeTime(changeIndex)};
        double changeValue{this->changeValue(changeIndex)};
        LOG_TRACE(<< "change index = " << changeIndex
                  << ", time = " << changeTime << ", value = " << changeValue);

        SShock shock{
            E_TimeShift,       changeTime,
            changeValue,       variance,
            truncatedVariance, static_cast<double>(shiftSegments.size() - 1)};
        shock.s_TimeShift = std::accumulate(shifts.begin(), shifts.end(), 0);

        return shock;
    }

    return {};
}

CTimeSeriesTestForChange::TBucketPredictor CTimeSeriesTestForChange::bucketPredictor() const {
    return [this](std::size_t i) {
        return m_Predictor(m_ValuesStartTime +
                           m_BucketLength * static_cast<core_t::TTime>(i));
    };
}

CTimeSeriesTestForChange::TMeanVarAccumulator
CTimeSeriesTestForChange::truncatedMoments(double outlierFraction,
                                           const TFloatMeanAccumulatorVec& residuals,
                                           const TTransform& transform) const {
    double cutoff{std::numeric_limits<double>::max()};
    std::size_t count{CSignal::countNotMissing(residuals)};
    std::size_t numberOutliers{
        static_cast<std::size_t>(outlierFraction * static_cast<double>(count) + 0.5)};
    if (numberOutliers > 0) {
        m_Outliers.clear();
        m_Outliers.resize(numberOutliers);
        for (const auto& value : residuals) {
            if (CBasicStatistics::count(value) > 0.0) {
                m_Outliers.add(std::fabs(transform(value)));
            }
        }
        cutoff = m_Outliers.biggest();
        count -= m_Outliers.count();
    }
    LOG_TRACE(<< "cutoff = " << cutoff << ", count = " << count);

    TMeanVarAccumulator moments;
    for (const auto& value : residuals) {
        if (CBasicStatistics::count(value) > 0.0 && std::fabs(transform(value)) < cutoff) {
            moments.add(transform(value));
        }
    }
    if (numberOutliers > 0) {
        moments.add(cutoff, static_cast<double>(count) - CBasicStatistics::count(moments));
    }
    CBasicStatistics::moment<1>(moments) += m_EpsVariance;

    return moments;
}

core_t::TTime CTimeSeriesTestForChange::changeTime(std::size_t changeIndex) const {
    return m_ValuesStartTime + m_BucketLength * static_cast<core_t::TTime>(changeIndex);
}

double CTimeSeriesTestForChange::changeValue(std::size_t changeIndex) const {
    return CBasicStatistics::mean(m_Values[changeIndex - 1]);
}

CTimeSeriesTestForChange::TFloatMeanAccumulatorVec
CTimeSeriesTestForChange::removePredictions(const TBucketPredictor& predictor,
                                            TFloatMeanAccumulatorVec values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (CBasicStatistics::count(values[i]) > 0.0) {
            CBasicStatistics::moment<0>(values[i]) -= predictor(i);
        }
    }
    return values;
}

std::size_t CTimeSeriesTestForChange::buckets(core_t::TTime bucketLength,
                                              core_t::TTime interval) {
    return static_cast<std::size_t>((interval + bucketLength / 2) / bucketLength);
}

double CTimeSeriesTestForChange::aic(const SShock& shock) {
    // This is max_{\theta}{ -2 log(P(y | \theta)) + 2 * # parameters }
    //
    // We assume that the data are normally distributed.
    return -std::log(std::exp(-2.0) / boost::math::double_constants::two_pi /
                     shock.s_ResidualVariance) +
           -std::log(std::exp(-2.0) / boost::math::double_constants::two_pi /
                     shock.s_TruncatedResidualVariance) +
           4.0 * shock.s_NumberParameters;
}

const std::string& CTimeSeriesTestForChange::print(EType type) {
    switch (type) {
    case E_LevelShift:
        return CLevelShift::TYPE;
    case E_Scale:
        return CScale::TYPE;
    case E_TimeShift:
        return CTimeShift::TYPE;
    case E_NoShock:
        return NO_CHANGE;
    }
}
}
}
