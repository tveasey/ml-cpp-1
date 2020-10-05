/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesSegmentation.h>

#include <core/CContainerPrinter.h>
#include <core/CTriple.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CSignal.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTools.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>

namespace ml {
namespace maths {

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                         double pValueToSegment,
                                         double outlierFraction) {
    TFloatMeanAccumulatorVec reweighted;
    TSizeVec segmentation{0, values.size()};
    fitTopDownPiecewiseLinear(values.cbegin(), values.cend(), 0, 0.0, pValueToSegment,
                              outlierFraction, segmentation, reweighted);
    std::sort(segmentation.begin(), segmentation.end());
    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                                               const TSizeVec& segmentation,
                                               double outlierFraction,
                                               TDoubleVec* shifts) {
    TFloatMeanAccumulatorVec reweighted{values};
    auto predict = fitPiecewiseLinear(segmentation, outlierFraction, reweighted);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (CBasicStatistics::count(values[i]) > 0.0) {
            CBasicStatistics::moment<0>(values[i]) -= predict(static_cast<double>(i));
        }
    }
    if (shifts != nullptr) {
        shifts->clear();
        shifts->reserve(segmentation.size() - 2);
        for (std::size_t i = 1; i < segmentation.size(); ++i) {
            TMeanAccumulator shift;
            for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
                shift.add(predict(static_cast<double>(j)));
            }
            shifts->push_back(CBasicStatistics::mean(shift));
        }
    }
    return values;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinearDiscontinuities(TFloatMeanAccumulatorVec values,
                                                              const TSizeVec& segmentation,
                                                              double outlierFraction) {
    TFloatMeanAccumulatorVec reweighted{values};
    auto predict = fitPiecewiseLinear(segmentation, outlierFraction, reweighted);
    TDoubleVec steps;
    steps.reserve(segmentation.size() - 1);
    for (std::size_t i = 1; i + 1 < segmentation.size(); ++i) {
        double j{static_cast<double>(segmentation[i])};
        steps.push_back(predict(j - 0.05) - predict(j + 0.05));
    }
    steps.push_back(0.0);
    LOG_TRACE(<< "steps = " << core::CContainerPrinter::print(steps));
    std::partial_sum(steps.rbegin(), steps.rend(), steps.rbegin());
    for (std::size_t i = segmentation.size() - 1; i > 0; --i) {
        LOG_TRACE(<< "Shifting [" << segmentation[i - 1] << ","
                  << segmentation[i] << ") by " << steps[i - 1]);
        for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
            CBasicStatistics::moment<0>(values[j]) -= steps[i - 1];
        }
    }
    return values;
}

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                       const TSeasonality& model,
                                                       std::size_t minimumSegmentLength,
                                                       double pValueToSegment) {
    minimumSegmentLength = std::max(minimumSegmentLength, std::size_t{1});
    TSizeVec segmentation{0, values.size()};
    fitTopDownPiecewiseLinearScaledSeasonal(values.cbegin(), values.cend(), 0,
                                            minimumSegmentLength, model,
                                            pValueToSegment, segmentation);
    std::sort(segmentation.begin(), segmentation.end());
    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinearScaledSeasonal(TFloatMeanAccumulatorVec values,
                                                             const TSeasonality& model,
                                                             const TSizeVec& segmentation,
                                                             double outlierFraction,
                                                             TDoubleVec* scales_) {
    TFloatMeanAccumulatorVec reweighted{values};
    TDoubleVec scales;
    fitPiecewiseLinearScaledSeasonal(values, model, segmentation,
                                     outlierFraction, reweighted, scales);
    for (std::size_t i = 0; i < values.size(); ++i) {
        CBasicStatistics::moment<0>(values[i]) -= scaleAt(i, segmentation, scales) *
                                                  model(i);
    }
    if (scales_ != nullptr) {
        *scales_ = std::move(scales);
    }
    return values;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::meanScalePiecewiseLinearScaledSeasonal(
    const TFloatMeanAccumulatorVec& values,
    const TSeasonalComponentVec& periods,
    const TSizeVec& segmentation,
    const TIndexWeight& indexWeight,
    double outlierFraction) {

    using TMinMaxAccumulator = CBasicStatistics::CMinMax<double>;

    TDoubleVecVec models;
    TDoubleVec scales;
    TFloatMeanAccumulatorVec scaledValues;
    fitPiecewiseLinearScaledSeasonal(values, periods, segmentation, outlierFraction,
                                     scaledValues, models, scales);
    LOG_TRACE(<< "models = " << core::CContainerPrinter::print(models));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    double scale{meanScale(segmentation, scales, indexWeight)};
    LOG_TRACE(<< "mean scale = " << scale);

    scaledValues = values;
    for (std::size_t i = 0; i != values.size(); ++i) {
        for (std::size_t j = 0; j < periods.size(); ++j) {
            if (periods[j].contains(i) && CBasicStatistics::count(values[i]) > 0.0) {
                CBasicStatistics::moment<0>(scaledValues[i]) -=
                    scaleAt(i, segmentation, scales) * models[j][periods[j].offset(i)];
            }
        }
    }

    double noiseStandardDeviation{std::sqrt([&] {
        TMeanVarAccumulator moments;
        for (std::size_t i = 0; i < scaledValues.size(); ++i) {
            if (std::any_of(periods.begin(), periods.end(), [&](const auto& period) {
                    return period.contains(i);
                })) {
                moments.add(CBasicStatistics::mean(scaledValues[i]),
                            CBasicStatistics::count(scaledValues[i]));
            }
        }
        return CBasicStatistics::maximumLikelihoodVariance(moments);
    }())};

    double amplitude{[&] {
        TMinMaxAccumulator minmax;
        for (std::size_t i = 0; i < scaledValues.size(); ++i) {
            double prediction{0.0};
            for (std::size_t j = 0; j < periods.size(); ++j) {
                if (periods[j].contains(i)) {
                    prediction += models[j][periods[j].offset(i)];
                }
            }
            minmax.add(prediction);
        }
        return minmax.range();
    }()};

    LOG_TRACE(<< "amplitude = " << amplitude << ", sd noise = " << noiseStandardDeviation);

    if (*std::min_element(scales.begin(), scales.end()) < 0.0 ||
        2.0 * *std::max_element(scales.begin(), scales.end()) * amplitude <= noiseStandardDeviation) {
        return values;
    }

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::any_of(periods.begin(), periods.end(),
                        [&](const auto& period) { return period.contains(i); })) {
            // If the component is "scaled away" we treat it as missing.
            double weight{std::min(2.0 * scaleAt(i, segmentation, scales) * amplitude - noiseStandardDeviation,
                                   1.0)};
            double prediction{0.0};
            for (std::size_t j = 0; j < periods.size(); ++j) {
                if (periods[j].contains(i)) {
                    prediction += models[j][periods[j].offset(i)];
                }
            }

            if (weight <= 0.0) {
                scaledValues[i] = TFloatMeanAccumulator{};
            } else {
                CBasicStatistics::count(scaledValues[i]) *= weight;
                CBasicStatistics::moment<0>(scaledValues[i]) += scale * prediction;
            }
        }
    }

    return scaledValues;
}

double CTimeSeriesSegmentation::meanScale(const TSizeVec& segmentation,
                                          const TDoubleVec& scales,
                                          const TIndexWeight& indexWeight) {
    TMeanAccumulator result;
    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
            result.add(scales[i - 1], indexWeight(j));
        }
    }
    return CBasicStatistics::mean(result);
}

double CTimeSeriesSegmentation::scaleAt(std::size_t index,
                                        const TSizeVec& segmentation,
                                        const TDoubleVec& scales) {
    return scales[std::upper_bound(segmentation.begin(), segmentation.end(), index) -
                  segmentation.begin() - 1];
}

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseTimeShifted(const TFloatMeanAccumulatorVec& values,
                                              core_t::TTime bucketLength,
                                              const TTimeVec& candidateShifts,
                                              const TModel& model,
                                              std::size_t minimumSegmentLength,
                                              double pValueToSegment,
                                              TTimeVec* shifts) {

    minimumSegmentLength = std::max(minimumSegmentLength, std::size_t{1});

    TDoubleVecVec predictions(candidateShifts.size() + 1);
    for (std::size_t j = 0; j < values.size(); ++j) {
        predictions[0].push_back(model(bucketLength * static_cast<core_t::TTime>(j)));
    }
    for (std::size_t i = 0; i < candidateShifts.size(); ++i) {
        for (std::size_t j = 0; j < values.size(); ++j) {
            predictions[i + 1].push_back(model(
                bucketLength * static_cast<core_t::TTime>(j) + candidateShifts[i]));
        }
    }

    TFloatMeanAccumulatorVec reweighted;
    TSizeVec segmentation{0, values.size()};
    fitTopDownPiecewiseTimeShifted(values.cbegin(), values.cend(), 0, minimumSegmentLength,
                                   predictions, pValueToSegment, segmentation);

    std::sort(segmentation.begin(), segmentation.end());
    if (shifts != nullptr) {
        shifts->clear();
        shifts->reserve(segmentation.size() - 1);
        for (std::size_t i = 1; i < segmentation.size(); ++i) {
            std::size_t shift;
            std::tie(std::ignore, shift) = residualMoments(
                values.begin() + segmentation[i - 1],
                values.begin() + segmentation[i], segmentation[i - 1], predictions);
            shifts->push_back(shift == 0 ? 0 : candidateShifts[shift - 1]);
        }
    }

    return segmentation;
}

core_t::TTime CTimeSeriesSegmentation::shiftAt(std::size_t index,
                                               const TSizeVec& segmentation,
                                               const TTimeVec& shifts) {
    return shifts[std::upper_bound(segmentation.begin(), segmentation.end(), index) -
                  segmentation.begin() - 1];
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseLinear(ITR begin,
                                                        ITR end,
                                                        std::size_t offset,
                                                        double startTime,
                                                        double pValueToSegment,
                                                        double outlierFraction,
                                                        TSizeVec& segmentation,
                                                        TFloatMeanAccumulatorVec& reweighted) {

    // We want to find the partition of [begin, end) into linear models which
    // efficiently predicts the values. To this end we maximize R^2 whilst
    // maintaining a parsimonious model. We achieve the later objective by
    // doing model selection for each candidate segment using the significance
    // of the variance ratio.

    auto range = std::distance(begin, end);
    reweighted.assign(begin, end);
    TMeanVarAccumulator moments{[&] {
        TRegression model{fitLinearModel(reweighted.cbegin(), reweighted.cend(), startTime)};
        auto predict = [&](std::size_t i) {
            double time{startTime + static_cast<double>(i)};
            return model.predict(time);
        };
        CSignal::reweightOutliers(predict, outlierFraction, reweighted);
        return centredResidualMoments(reweighted.cbegin(), reweighted.cend(), startTime);
    }()};

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    int step{3};
    if (range >= 2 * step && CBasicStatistics::maximumLikelihoodVariance(moments) > 0.0) {
        using TDoubleDoubleItrTr = core::CTriple<double, double, ITR>;
        using TMinAccumulator =
            CBasicStatistics::COrderStatisticsStack<TDoubleDoubleItrTr, 1>;

        auto variance = [](const TMeanVarAccumulator& l, const TMeanVarAccumulator& r) {
            double nl{CBasicStatistics::count(l)};
            double nr{CBasicStatistics::count(r)};
            return (nl * CBasicStatistics::maximumLikelihoodVariance(l) +
                    nr * CBasicStatistics::maximumLikelihoodVariance(r)) /
                   (nl + nr);
        };

        // We iterate through every possible binary split of the data into
        // contiguous ranges looking for the split which minimizes the total
        // variance of the values minus linear model predictions. Outliers
        // are removed based on an initial best split.

        TMinAccumulator minResidualVariance;
        reweighted.assign(begin, end);

        for (auto reweights : {0, 1}) {
            LOG_TRACE(<< "  reweights = " << reweights);

            ITR i = reweighted.cbegin();
            minResidualVariance = TMinAccumulator{};
            double splitTime{startTime};
            TRegression leftModel;
            TRegression rightModel{fitLinearModel(i, reweighted.cend(), splitTime)};

            auto varianceAfterStep = [&](int j) {
                TRegression deltaModel{fitLinearModel(i, i + j, splitTime)};
                return variance(residualMoments(reweighted.cbegin(), i + j,
                                                startTime, leftModel + deltaModel),
                                residualMoments(i + j, reweighted.cend(), splitTime + j,
                                                rightModel - deltaModel));
            };

            for (/**/; i + step < reweighted.cend(); i += step, splitTime += step) {
                if (minResidualVariance.add(
                        {varianceAfterStep(step), splitTime + step, i + step})) {
                    for (int j = 1; j < step; ++j) {
                        minResidualVariance.add(
                            {varianceAfterStep(j), splitTime + j, i + j});
                    }
                }
                TRegression deltaModel{fitLinearModel(i, i + step, splitTime)};
                leftModel += deltaModel;
                rightModel -= deltaModel;
            }

            if (outlierFraction < 0.0 || outlierFraction >= 1.0) {
                LOG_ERROR(<< "Ignoring bad value for outlier fraction " << outlierFraction
                          << ". This must be in the range [0,1).");
                break;
            } else if (reweights == 1 || outlierFraction == 0.0) {
                break;
            } else {
                splitTime = minResidualVariance[0].second;
                ITR split{minResidualVariance[0].third};
                leftModel = fitLinearModel(reweighted.cbegin(), split, startTime);
                rightModel = fitLinearModel(split, reweighted.cend(), splitTime);
                auto predict = [&](std::size_t j) {
                    double time{startTime + static_cast<double>(j)};
                    return (time < splitTime ? leftModel : rightModel).predict(time);
                };
                reweighted.assign(begin, end);
                CSignal::reweightOutliers(predict, outlierFraction, reweighted);
            }
        }

        double splitTime{minResidualVariance[0].second};
        ITR split{minResidualVariance[0].third};
        std::size_t splitOffset(std::distance(reweighted.cbegin(), split));
        LOG_TRACE(<< "  candidate = " << offset + splitOffset
                  << ", variance = " << minResidualVariance[0].first);

        TMeanVarAccumulator leftMoments{
            centredResidualMoments(reweighted.cbegin(), split, startTime)};
        TMeanVarAccumulator rightMoments{
            centredResidualMoments(split, reweighted.cend(), splitTime)};
        TMeanAccumulator meanAbs{std::accumulate(
            begin, end, TMeanAccumulator{}, [](auto partialMean, const auto& value) {
                partialMean.add(std::fabs(CBasicStatistics::mean(value)),
                                CBasicStatistics::count(value));
                return partialMean;
            })};
        LOG_TRACE(<< "  total moments = " << moments << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments << ", mean abs = " << meanAbs);

        double v[]{CBasicStatistics::maximumLikelihoodVariance(moments),
                   CBasicStatistics::maximumLikelihoodVariance(leftMoments + rightMoments) +
                       MINIMUM_COEFFICIENT_OF_VARIATION * CBasicStatistics::mean(meanAbs)};
        double df[]{4.0 - 2.0, static_cast<double>(range - 4)};
        double F{(df[1] * std::max(v[0] - v[1], 0.0)) / (df[0] * v[1])};
        double pValue{CTools::oneMinusPowOneMinusX(
            CStatisticalTests::rightTailFTest(F, df[0], df[1]),
            static_cast<double>(reweighted.size() / step))};
        LOG_TRACE(<< "  p-value = " << pValue);

        if (pValue < pValueToSegment) {
            fitTopDownPiecewiseLinear(begin, begin + splitOffset, offset,
                                      startTime, pValueToSegment,
                                      outlierFraction, segmentation, reweighted);
            fitTopDownPiecewiseLinear(begin + splitOffset, end, offset + splitOffset,
                                      splitTime, pValueToSegment,
                                      outlierFraction, segmentation, reweighted);
            segmentation.push_back(offset + splitOffset);
        }
    }
}

CTimeSeriesSegmentation::TPredictor
CTimeSeriesSegmentation::fitPiecewiseLinear(const TSizeVec& segmentation,
                                            double outlierFraction,
                                            TFloatMeanAccumulatorVec& values) {
    using TRegressionVec = std::vector<TRegression>;

    TDoubleVec segmentEndTimes(segmentation.size() - 1,
                               static_cast<double>(values.size()));
    TRegressionVec models(segmentation.size() - 1);

    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        double a{static_cast<double>(segmentation[i - 1])};
        double b{static_cast<double>(segmentation[i])};
        segmentEndTimes[i - 1] = b;
        models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                       values.begin() + segmentation[i], a);
    }
    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    auto predict = [&](std::size_t i) {
        double time{static_cast<double>(i)};
        auto index = std::upper_bound(segmentEndTimes.begin(), segmentEndTimes.end(), time);
        return models[index - segmentEndTimes.begin()].predict(time);
    };

    if (CSignal::reweightOutliers(predict, outlierFraction, values)) {
        for (std::size_t i = 1; i < segmentation.size(); ++i) {
            double a{static_cast<double>(segmentation[i - 1])};
            models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                           values.begin() + segmentation[i], a);
            LOG_TRACE(<< "model = " << models[i - 1].print());
        }
    }

    return [=](double time) {
        auto index = std::upper_bound(segmentEndTimes.begin(), segmentEndTimes.end(), time);
        return models[index - segmentEndTimes.begin()].predict(time);
    };
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseLinearScaledSeasonal(
    ITR begin,
    ITR end,
    std::size_t offset,
    std::size_t minimumSegmentLength,
    const TSeasonality& model,
    double pValueToSegment,
    TSizeVec& segmentation) {

    // We want to find the partition of [begin, end) into scaled seasonal
    // models which efficiently predicts the values. To this end we maximize
    // R^2 whilst maintaining a parsimonious model. We achieve the later
    // objective by doing model selection for each candidate segment using
    // a significance test of the explained variance.

    using TDoubleItrPr = std::pair<double, ITR>;
    using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<TDoubleItrPr, 1>;
    using TMeanAccumulatorAry = std::array<TMeanAccumulator, 3>;

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};
    TMeanVarAccumulator moments{centredResidualMoments(begin, end, offset, model)};

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    if (range > 2 * minimumSegmentLength &&
        CBasicStatistics::maximumLikelihoodVariance(moments) > 0.0) {
        TMinAccumulator minResidualVariance;

        auto statistics = [offset, begin, &model](ITR begin_, ITR end_) {
            TMeanAccumulatorAry result;
            for (ITR i = begin_; i != end_; ++i) {
                double x{CBasicStatistics::mean(*i)};
                double w{CBasicStatistics::count(*i)};
                double p{model(offset + std::distance(begin, i))};
                result[0].add(x * x, w);
                result[1].add(x * p, w);
                result[2].add(p * p, w);
            }
            return result;
        };
        auto variance = [](const TMeanAccumulatorAry& l, const TMeanAccumulatorAry& r) {
            double nl{CBasicStatistics::count(l[0])};
            double nr{CBasicStatistics::count(r[0])};
            double vl{CBasicStatistics::mean(l[0]) -
                      (CBasicStatistics::mean(l[2]) == 0.0
                           ? 0.0
                           : CTools::pow2(CBasicStatistics::mean(l[1])) /
                                 CBasicStatistics::mean(l[2]))};
            double vr{CBasicStatistics::mean(r[0]) -
                      (CBasicStatistics::mean(l[2]) == 0.0
                           ? 0.0
                           : CTools::pow2(CBasicStatistics::mean(r[1])) /
                                 CBasicStatistics::mean(r[2]))};
            return (nl * vl + nr * vr) / (nl + nr);
        };

        ITR i = begin + minimumSegmentLength;

        TMeanAccumulatorAry leftStatistics{statistics(begin, i)};
        TMeanAccumulatorAry rightStatistics{statistics(i, end)};
        minResidualVariance.add({variance(leftStatistics, rightStatistics), i});

        // The following loop is the interesting part. We compute (up to constants)
        //
        //   min_j{ min_s{ sum_{i<j}{(x_i - s p_i)^2} } + min_s{ sum_{i>=j}{(x_i - s p_i)^2} } }
        //
        // We use the fact that we can maintain sufficient statistics to which
        // we can both add and remove values to compute this in O(|S||T|) for
        // S candidate splits and T sufficient statistics.

        for (/**/; i + minimumSegmentLength != end; ++i) {
            TMeanAccumulatorAry deltaStatistics{statistics(i, i + 1)};
            for (std::size_t j = 0; j < 3; ++j) {
                leftStatistics[j] += deltaStatistics[j];
                rightStatistics[j] -= deltaStatistics[j];
            }
            minResidualVariance.add({variance(leftStatistics, rightStatistics), i + 1});
        }

        ITR split{minResidualVariance[0].second};
        std::size_t splitOffset{offset + std::distance(begin, split)};
        LOG_TRACE(<< "  candidate = " << splitOffset
                  << ", variance = " << minResidualVariance[0].first);

        TMeanVarAccumulator leftMoments{centredResidualMoments(begin, split, offset, model)};
        TMeanVarAccumulator rightMoments{
            centredResidualMoments(split, end, splitOffset, model)};
        TMeanAccumulator meanAbs{std::accumulate(
            begin, end, TMeanAccumulator{}, [](auto partialMean, const auto& value) {
                partialMean.add(std::fabs(CBasicStatistics::mean(value)),
                                CBasicStatistics::count(value));
                return partialMean;
            })};
        LOG_TRACE(<< "  total moments = " << moments << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments << ", mean abs = " << meanAbs);

        double v[]{CBasicStatistics::maximumLikelihoodVariance(moments),
                   CBasicStatistics::maximumLikelihoodVariance(leftMoments + rightMoments) +
                       MINIMUM_COEFFICIENT_OF_VARIATION * CBasicStatistics::mean(meanAbs)};
        double df[]{2.0 - 1.0, static_cast<double>(range - 2)};
        double F{df[1] * std::max(v[0] - v[1], 0.0) / (df[0] * v[1])};
        double significance{CTools::oneMinusPowOneMinusX(
            CStatisticalTests::rightTailFTest(F, df[0], df[1]),
            static_cast<double>(range - 2 * minimumSegmentLength))};
        LOG_TRACE(<< "  significance = " << significance);

        if (significance < pValueToSegment) {
            fitTopDownPiecewiseLinearScaledSeasonal(begin, split, offset,
                                                    minimumSegmentLength, model,
                                                    pValueToSegment, segmentation);
            fitTopDownPiecewiseLinearScaledSeasonal(split, end, splitOffset,
                                                    minimumSegmentLength, model,
                                                    pValueToSegment, segmentation);
            segmentation.push_back(splitOffset);
        }
    }
}

void CTimeSeriesSegmentation::fitPiecewiseLinearScaledSeasonal(
    const TFloatMeanAccumulatorVec& values,
    const TSeasonality& model,
    const TSizeVec& segmentation,
    double outlierFraction,
    TFloatMeanAccumulatorVec& reweighted,
    TDoubleVec& scales) {

    scales.assign(segmentation.size() - 1, 1.0);

    auto scale = [&](std::size_t i) { return scaleAt(i, segmentation, scales); };

    auto scaledModel = [&](std::size_t i) { return scale(i) * model(i); };

    auto computeScales = [&] {
        for (std::size_t j = 1; j < segmentation.size(); ++j) {
            TMeanAccumulator projection;
            TMeanAccumulator Z;
            for (std::size_t i = segmentation[j - 1]; i < segmentation[j]; ++i) {
                double x{CBasicStatistics::mean(reweighted[i])};
                double w{CBasicStatistics::count(reweighted[i])};
                double p{model(i)};
                if (w > 0.0) {
                    projection.add(x * p, w);
                    Z.add(p * p, w);
                }
            }
            scales[j - 1] = CBasicStatistics::mean(Z) == 0.0
                                ? 1.0
                                : CBasicStatistics::mean(projection) /
                                      CBasicStatistics::mean(Z);
        }
    };

    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    // First pass to re-weight any large outliers.
    reweighted = values;
    computeScales();
    CSignal::reweightOutliers(scaledModel, outlierFraction, reweighted);

    // Fit the model adjusting for scales.
    computeScales();
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    // Re-weight outliers based on the new model.
    reweighted = values;
    if (CSignal::reweightOutliers(scaledModel, outlierFraction, reweighted)) {
        // If any re-weighting happened fine tune.
        computeScales();
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
    }
}

void CTimeSeriesSegmentation::fitPiecewiseLinearScaledSeasonal(
    const TFloatMeanAccumulatorVec& values,
    const TSeasonalComponentVec& periods,
    const TSizeVec& segmentation,
    double outlierFraction,
    TFloatMeanAccumulatorVec& reweighted,
    TDoubleVecVec& models,
    TDoubleVec& scales) {

    models.assign(periods.size(), {});
    scales.assign(segmentation.size() - 1, 1.0);

    auto model = [&](std::size_t i) {
        double result{0.0};
        for (std::size_t j = 0; j < periods.size(); ++j) {
            if (periods[j].contains(i)) {
                result += models[j][periods[j].offset(i)];
            }
        }
        return result;
    };

    auto scale = [&](std::size_t i) { return scaleAt(i, segmentation, scales); };

    auto scaledModel = [&](std::size_t i) { return scale(i) * model(i); };

    auto computeScale = [&](std::size_t begin, std::size_t end) {
        TMeanAccumulator projection;
        TMeanAccumulator Z;
        for (std::size_t i = begin; i < end; ++i) {
            double x{CBasicStatistics::mean(reweighted[i])};
            double w{CBasicStatistics::count(reweighted[i])};
            double p{model(i)};
            if (w > 0.0) {
                projection.add(x * p, w);
                Z.add(p * p, w);
            }
        }
        return CBasicStatistics::mean(Z) == 0.0
                   ? 1.0
                   : CBasicStatistics::mean(projection) / CBasicStatistics::mean(Z);
    };

    auto fitSeasonalModels = [&](const TScale& scale_) {
        for (std::size_t i = 0; i < periods.size(); ++i) {
            models[i].assign(periods[i].period(), 0.0);
        }
        std::size_t iterations(periods.size() > 1 ? 2 : 1);
        for (std::size_t i = 0; i < iterations; ++i) {
            for (std::size_t j = 0; j < periods.size(); ++j) {
                models[j].assign(periods[j].period(), 0.0);
                fitSeasonalModel(reweighted.begin(), reweighted.end(),
                                 periods[j], model, scale_, models[j]);
            }
        }
    };

    auto fitScaledSeasonalModels = [&](const TScale& scale_) {
        for (std::size_t i = 0; i < 2; ++i) {
            fitSeasonalModels(scale_);
            for (std::size_t j = 1; j < segmentation.size(); ++j) {
                scales[j - 1] = computeScale(segmentation[j - 1], segmentation[j]);
            }
        }
    };

    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    // First pass to re-weight any large outliers.
    reweighted = values;
    fitSeasonalModels([](std::size_t) { return 1.0; });
    CSignal::reweightOutliers(scaledModel, outlierFraction, reweighted);

    // Fit the model adjusting for scales.
    fitScaledSeasonalModels(scale);
    LOG_TRACE(<< "models = " << core::CContainerPrinter::print(models));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    // Re-weight outliers based on the new model.
    reweighted = values;
    if (CSignal::reweightOutliers(scaledModel, outlierFraction, reweighted)) {
        // If any re-weighting happened fine tune.
        fitScaledSeasonalModels(scale);
        LOG_TRACE(<< "model = " << core::CContainerPrinter::print(models));
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
    }
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseTimeShifted(ITR begin,
                                                             ITR end,
                                                             std::size_t offset,
                                                             std::size_t minimumSegmentLength,
                                                             const TDoubleVecVec& predictions,
                                                             double pValueToSegment,
                                                             TSizeVec& segmentation) {

    using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;

    // We want to find the partition of [begin, end) into shifted seasonal
    // models which efficiently predicts the values. To this end we maximize
    // R^2 whilst maintaining a parsimonious model. We achieve the later
    // objective by doing model selection for each candidate segment using
    // a significance test of the explained variance.

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};
    TMeanVarAccumulator totalMoments;
    std::tie(totalMoments, std::ignore) = residualMoments(begin, end, offset, predictions);
    LOG_TRACE(<< "total moments = " << totalMoments);
    double totalVariance{CBasicStatistics::maximumLikelihoodVariance(totalMoments)};

    if (range > 2 * minimumSegmentLength && totalVariance > 0.0) {

        LOG_TRACE(<< "  partitioning [" << offset << "," << offset + range << ")");
        LOG_TRACE(<< "  offset = " << offset);

        double bestSplitVariance{std::numeric_limits<double>::max()};
        std::size_t bestSplit{range};

        TMeanVarAccumulatorVec leftMoments(predictions.size());
        for (std::size_t j = 0; j + 1 < minimumSegmentLength; ++j) {
            const auto& value = *(begin + j);
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                leftMoments[i].add(CBasicStatistics::mean(value) - predictions[i][offset + j],
                                   CBasicStatistics::count(value));
            }
        }
        TMeanVarAccumulatorVec rightMoments(predictions.size());
        for (std::size_t j = minimumSegmentLength - 1; j < range; ++j) {
            const auto& value = *(begin + j);
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                rightMoments[i].add(CBasicStatistics::mean(value) -
                                        predictions[i][offset + j],
                                    CBasicStatistics::count(value));
            }
        }

        // The following loop is the interesting part. We compute
        //
        //   min_j{ min_s{ sum_{i < j}{(x_i - p_i(s))^2} } + min_s{ sum_{i >= j}{(x_i - p_i(s))^2} } }
        //
        // We use the fact that we can remove values from the central moments
        // accumulator to compute this in O(|S||T|^2) for S candidate splits
        // and T candidate splits.

        for (std::size_t split = minimumSegmentLength;
             split + minimumSegmentLength < range; ++split) {
            const auto& valueLeftOfSplit = *(begin + split - 1);
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                leftMoments[i].add(CBasicStatistics::mean(valueLeftOfSplit) -
                                       predictions[i][offset + split - 1],
                                   CBasicStatistics::count(valueLeftOfSplit));
            }
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                TMeanVarAccumulator delta;
                delta.add(CBasicStatistics::mean(valueLeftOfSplit) -
                              predictions[i][offset + split - 1],
                          CBasicStatistics::count(valueLeftOfSplit));
                rightMoments[i] -= delta;
            }

            double variance{std::numeric_limits<double>::max()};
            for (std::size_t i = 0; i < leftMoments.size(); ++i) {
                for (std::size_t j = 0; j < rightMoments.size(); ++j) {
                    double shiftsVariance{CBasicStatistics::maximumLikelihoodVariance(
                        leftMoments[i] + rightMoments[j])};
                    if (shiftsVariance < variance) {
                        variance = shiftsVariance;
                    }
                }
            }

            if (variance < bestSplitVariance) {
                std::tie(bestSplitVariance, bestSplit) = std::make_pair(variance, split);
            }
        }

        auto split = begin + bestSplit;
        std::size_t splitOffset{offset + bestSplit};

        TMeanAccumulator meanAbs{std::accumulate(
            begin, end, TMeanAccumulator{}, [](auto partialMean, const auto& value) {
                partialMean.add(std::fabs(CBasicStatistics::mean(value)),
                                CBasicStatistics::count(value));
                return partialMean;
            })};
        LOG_TRACE(<< "  total variance = " << totalVariance << ", best split = " << bestSplit
                  << ", best split variance = " << bestSplitVariance
                  << ", mean abs = " << meanAbs);

        double v[]{totalVariance, bestSplitVariance + MINIMUM_COEFFICIENT_OF_VARIATION *
                                                          CBasicStatistics::mean(meanAbs)};
        double df[]{3.0 - 2.0, static_cast<double>(range - 3)};
        double F{df[1] * std::max(v[0] - v[1], 0.0) / (df[0] * v[1])};
        double significance{CTools::oneMinusPowOneMinusX(
            CStatisticalTests::rightTailFTest(F, df[0], df[1]),
            static_cast<double>(range - 2 * minimumSegmentLength))};
        LOG_TRACE(<< "  significance = " << significance);

        if (significance < pValueToSegment) {
            fitTopDownPiecewiseTimeShifted(begin, split, offset, minimumSegmentLength,
                                           predictions, pValueToSegment, segmentation);
            fitTopDownPiecewiseTimeShifted(split, end, splitOffset, minimumSegmentLength,
                                           predictions, pValueToSegment, segmentation);
            segmentation.push_back(splitOffset);
        }
    }
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::centredResidualMoments(ITR begin, ITR end, double startTime) {
    if (begin == end) {
        return CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);
    }
    if (std::distance(begin, end) == 1) {
        return CBasicStatistics::momentsAccumulator(CBasicStatistics::count(*begin),
                                                    0.0, 0.0);
    }
    TRegression model{fitLinearModel(begin, end, startTime)};
    TMeanVarAccumulator moments{residualMoments(begin, end, startTime, model)};
    CBasicStatistics::moment<0>(moments) = 0.0;
    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::centredResidualMoments(ITR begin,
                                                ITR end,
                                                std::size_t offset,
                                                const TSeasonality& model) {
    if (begin == end) {
        return CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);
    }
    if (std::distance(begin, end) == 1) {
        return CBasicStatistics::momentsAccumulator(CBasicStatistics::count(*begin),
                                                    0.0, 0.0);
    }

    TMeanAccumulator projection;
    TMeanAccumulator Z;
    for (ITR i = begin; i != end; ++i) {
        double x{CBasicStatistics::mean(*i)};
        double w{CBasicStatistics::count(*i)};
        double p{model(offset + std::distance(begin, i))};
        if (w > 0.0) {
            projection.add(x * p, w);
            Z.add(p * p, w);
        }
    }
    double scale{CBasicStatistics::mean(Z) == 0.0
                     ? 1.0
                     : CBasicStatistics::mean(projection) / CBasicStatistics::mean(Z)};
    LOG_TRACE(<< "  scale = " << scale);

    TMeanVarAccumulator moments;
    for (ITR i = begin; i != end; ++i) {
        double x{CBasicStatistics::mean(*i)};
        double w{CBasicStatistics::count(*i)};
        double p{scale * model(offset + std::distance(begin, i))};
        if (w > 0.0) {
            moments.add(x - p, w);
        }
    }
    CBasicStatistics::moment<0>(moments) = 0.0;
    LOG_TRACE(<< "  moments = " << moments);

    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::residualMoments(ITR begin, ITR end, double startTime, const TRegression& model) {
    TMeanVarAccumulator moments;
    TRegression::TArray params;
    model.parameters(params);
    for (double time = startTime; begin != end; ++begin, time += 1.0) {
        double w{CBasicStatistics::count(*begin)};
        if (w > 0.0) {
            double x{CBasicStatistics::mean(*begin) - TRegression::predict(params, time)};
            moments.add(x, w);
        }
    }
    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulatorSizePr
CTimeSeriesSegmentation::residualMoments(ITR begin,
                                         ITR end,
                                         std::size_t offset,
                                         const TDoubleVecVec& predictions) {

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};

    TMeanVarAccumulator bestMoments;
    std::size_t bestShift{predictions.size()};
    double bestVariance{std::numeric_limits<double>::max()};

    for (std::size_t i = 0; i < predictions.size(); ++i) {
        TMeanVarAccumulator shiftMoments;
        for (std::size_t j = 0; j < range; ++j) {
            const auto& value = *(begin + j);
            shiftMoments.add(CBasicStatistics::mean(value) - predictions[i][offset + j],
                             CBasicStatistics::count(value));
        }
        if (CBasicStatistics::maximumLikelihoodVariance(shiftMoments) < bestVariance) {
            std::tie(bestMoments, bestShift, bestVariance) = std::make_tuple(
                shiftMoments, i, CBasicStatistics::maximumLikelihoodVariance(shiftMoments));
        }
    }

    return std::make_pair(bestMoments, bestShift);
}

template<typename ITR>
CTimeSeriesSegmentation::TRegression
CTimeSeriesSegmentation::fitLinearModel(ITR begin, ITR end, double startTime) {
    TRegression result;
    for (double time = startTime; begin != end; ++begin) {
        double x{CBasicStatistics::mean(*begin)};
        double w{CBasicStatistics::count(*begin)};
        if (w > 0.0) {
            result.add(time, x, w);
        }
        time += 1.0;
    }
    return result;
}

template<typename ITR>
void CTimeSeriesSegmentation::fitSeasonalModel(ITR begin,
                                               ITR end,
                                               const TSeasonalComponent& period,
                                               const TPredictor& predictor,
                                               const TScale& scale,
                                               TDoubleVec& result) {
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

    TMeanAccumulatorVec model(period.period());
    for (std::size_t i = 0; begin != end; ++i, ++begin) {
        if (period.contains(i)) {
            double x{CBasicStatistics::mean(*begin) / scale(i)};
            double w{CBasicStatistics::count(*begin) * scale(i)};
            if (w > 0.0) {
                model[period.offset(i)].add(x - predictor(static_cast<double>(i)), w);
            }
        }
    }

    for (std::size_t i = 0; i < model.size(); ++i) {
        result[i] = CBasicStatistics::mean(model[i]);
    }
}
}
}
