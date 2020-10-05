/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDE_ml_maths_CTimeSeriesSegmentation_h
#define INCLUDE_ml_maths_CTimeSeriesSegmentation_h

#include <maths/CBasicStatistics.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CSignal.h>
#include <maths/Constants.h>
#include <maths/MathsTypes.h>

#include <tuple>
#include <vector>

namespace ml {
namespace maths {

//! \brief Utility functionality to perform segmentation of a time series.
class MATHS_EXPORT CTimeSeriesSegmentation {
public:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TDoubleVecDoubleVecPr = std::pair<TDoubleVec, TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TTimeVec = std::vector<core_t::TTime>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TFloatMeanAccumulatorVecDoubleVecBoolTr =
        std::tuple<TFloatMeanAccumulatorVec, TDoubleVec, bool>;
    using TSeasonalComponent = CSignal::SSeasonalComponentSummary;
    using TSeasonalComponentVec = std::vector<TSeasonalComponent>;
    using TSeasonality = std::function<double(std::size_t)>;
    using TIndexWeight = std::function<double(std::size_t)>;
    using TModel = std::function<double(core_t::TTime)>;

    //! Perform top-down recursive segmentation with linear models.
    //!
    //! The time series is segmented using piecewise linear models in a top down
    //! fashion with each break point being chosen to maximize r-squared and model
    //! selection happening for each candidate segmentation. Model selection
    //! is achieved by thresholding the significance of the unexplained variance
    //! ratio for segmented versus non-segmented models.
    //!
    //! \param[in] values The time series values to segment. These are assumed to
    //! be equally spaced in time order.
    //! \param[in] pValueToSegment The maximum p-value of the explained variance
    //! to accept a trend segment.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when fitting the model and computing unexplained
    //! variance for model selection.
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec piecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                    double pValueToSegment,
                                    double outlierFraction);

    //! Remove the predictions of a piecewise linear model.
    //!
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \param[out] shifts If not null filled in with the average shift of each segment.
    //! \return The values minus the model predictions.
    static TFloatMeanAccumulatorVec removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                                                          const TSizeVec& segmentation,
                                                          double outlierFraction,
                                                          TDoubleVec* shifts = nullptr);

    //! Remove only the jump discontinuities in the segmented model.
    //!
    //! This removes discontinuities corresponding to the models in adjacent segments
    //! in backwards pass such that values in each preceding segment are adjusted
    //! relative to each succeeding segment.
    //!
    //! \param[in] segmentation The segmentation of \p values to use.
    //! \return The values minus discontinuities.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinearDiscontinuities(TFloatMeanAccumulatorVec values,
                                         const TSizeVec& segmentation,
                                         double outlierFraction);

    //! Perform top-down recursive segmentation of a seasonal model into segments with
    //! constant linear scale.
    //!
    //! The time series is segmented using piecewise constant linear scaled seasonal
    //! model in a top down fashion with break points being chosen to maximize r-squared
    //! and model selection happening for each candidate segmentation. Since each split
    //! generates a nested model we check the significance using the explained variance
    //! divided by segmented model's residual variance. A scaled seasonal model is
    //! defined here as:
    //! <pre class="fragment">
    //!   \f$ \sum_i 1\{t_i \leq t < t_{i+1}\} s_i f_p(t) \f$
    //! </pre>
    //! where \f$\{s_i\}\f$ are a collection of constant scales and \f$f_p(\cdot)\f$
    //! denotes a function with period \p period.
    //!
    //! \param[in] values The time series values to segment. These are assumed to be
    //! equally spaced in time order.
    //! \param[in] model A model of the seasonality to segment which returns its value
    //! for the i'th bucket of \p values.
    //! \param[in] minimumSegmentLength The minimum segment length considered.
    //! \param[in] pValueToSegment The maximum p-value of the explained variance
    //! to accept a scaling segment.
    //! \return The sorted segmentation indices. This includes the start and end
    //! indices of \p values, i.e. 0 and values.size().
    static TSizeVec piecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                  const TSeasonality& model,
                                                  std::size_t minimumSegmentLength,
                                                  double pValueToSegment);

    //! Remove the scaled predictions \p model from \p values.
    //!
    //! This fits piecewise linear scaled \p model for the segmentation \p segmentation
    //! to \p values and returns \p values minus its predictions.
    //!
    //! \param[in] model A model of the seasonality to remove which returns its value
    //! for the i'th bucket of \p values.
    //! \param[in] segmentation The segmentation of \p values into intervals with
    //! constant scale.
    //! \param[in] outlierFraction The proportion of values to treat as outliers.
    //! Outliers are re-weighted when fitting the model.
    //! \param[out] scales If not null filled in with the average scale of each segment.
    //! \return The values minus the scaled model predictions.
    static TFloatMeanAccumulatorVec
    removePiecewiseLinearScaledSeasonal(TFloatMeanAccumulatorVec values,
                                        const TSeasonality& model,
                                        const TSizeVec& segmentation,
                                        double outlierFraction,
                                        TDoubleVec* scales = nullptr);

    //! Rescale the piecewise linear scaled seasonal component of \p values with
    //! period \p period to its mean scale.
    //!
    //! \param[in] segmentation The segmentation of \p values into intervals with
    //! constant scale.
    //! \param[in] indexWeight A function used to weight indices of \p segmentation.
    //! \return The values with the mean scaled seasonal component.
    static TFloatMeanAccumulatorVec
    meanScalePiecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                           const TSeasonalComponentVec& periods,
                                           const TSizeVec& segmentation,
                                           const TIndexWeight& indexWeight,
                                           double outlierFraction);

    //! Compute the weighted mean scale for the piecewise linear \p scales on
    //! \p segmentation.
    //!
    //! \param[in] segmentation The segmentation into intervals with constant scale.
    //! \param[in] scales The piecewise constant linear scales.
    //! \param[in] indexWeight A function used to weight indices of \p segmentation.
    static double meanScale(const TSizeVec& segmentation,
                            const TDoubleVec& scales,
                            const TIndexWeight& indexWeight);

    //! Compute the scale to use at \p index for the piecewise linear \p scales on
    //! \p segmentation.
    //!
    //! \param[in] index The index at which to compute the scale.
    //! \param[in] segmentation The segmentation into intervals with constant scale.
    //! \param[in] scales The piecewise constant linear scales to apply.
    //! \return The scale at \p index.
    static double scaleAt(std::size_t index, const TSizeVec& segmentation, const TDoubleVec& scales);

    //! Perform top-down recursive segmentation of a seasonal model into segments with
    //! constant time shift.
    //!
    //! \param[in] bucketLength The time bucket length of \p values.
    //! \param[in] candidateShifts The time shifts we'll consider applying.
    //! \param[in] model A model of the seasonality to segment which returns its value
    //! for the i'th bucket of \p values.
    //! \param[in] minimumSegmentLength The minimum segment length considered.
    //! \param[in] pValueToSegment The maximum p-value of the explained variance
    //! to accept a scaling segment.
    //! \param[out] shifts If not null filled in with the shift for each segment.
    static TSizeVec piecewiseTimeShifted(const TFloatMeanAccumulatorVec& values,
                                         core_t::TTime bucketLength,
                                         const TTimeVec& candidateShifts,
                                         const TModel& model,
                                         std::size_t minimumSegmentLength,
                                         double pValueToSegment,
                                         TTimeVec* shifts = nullptr);

    //! Compute the time shift to use at \p index for the piecewise constant \p shifts
    //! on \p segmentation.
    //!
    //! \param[in] index The index at which to compute the shift.
    //! \param[in] segmentation The segmentation into intervals with constant time shifts.
    //! \param[in] shifts The piecewise constant time shift to apply.
    //! \return The shift at \p index.
    static core_t::TTime
    shiftAt(std::size_t index, const TSizeVec& segmentation, const TTimeVec& shifts);

private:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulatorSizePr = std::pair<TMeanVarAccumulator, std::size_t>;
    using TRegression = CLeastSquaresOnlineRegression<1, double>;
    using TPredictor = std::function<double(double)>;
    using TScale = std::function<double(std::size_t)>;

private:
    //! Implements top-down recursive segmentation of [\p begin, \p end) to minimise
    //! square residuals for a linear model.
    template<typename ITR>
    static void fitTopDownPiecewiseLinear(ITR begin,
                                          ITR end,
                                          std::size_t offset,
                                          double startTime,
                                          double pValueToSegment,
                                          double outliersFraction,
                                          TSizeVec& segmentation,
                                          TFloatMeanAccumulatorVec& values);

    //! Fit a piecewise linear model to \p values for the segmentation \p segmentation.
    static TPredictor fitPiecewiseLinear(const TSizeVec& segmentation,
                                         double outlierFraction,
                                         TFloatMeanAccumulatorVec& values);

    //! Implements top-down recursive segmentation of [\p begin, \p end) to minimise
    //! square residuals for linear scales of \p model.
    template<typename ITR>
    static void fitTopDownPiecewiseLinearScaledSeasonal(ITR begin,
                                                        ITR end,
                                                        std::size_t offset,
                                                        std::size_t minimumSegmentLength,
                                                        const TSeasonality& model,
                                                        double pValueToSegment,
                                                        TSizeVec& segmentation);

    //! Fit \p model with piecewise constant linear scaling to \p values for the
    //! segmentation \p segmentation.
    static void fitPiecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                 const TSeasonality& model,
                                                 const TSizeVec& segmentation,
                                                 double outlierFraction,
                                                 TFloatMeanAccumulatorVec& reweighted,
                                                 TDoubleVec& scales);

    //! Fit a seasonal model with piecewise constant linear scaling to \p values
    //! for the segmentation \p segmentation.
    static void fitPiecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                 const TSeasonalComponentVec& periods,
                                                 const TSizeVec& segmentation,
                                                 double outlierFraction,
                                                 TFloatMeanAccumulatorVec& reweighted,
                                                 TDoubleVecVec& model,
                                                 TDoubleVec& scales);

    //! Implements top-down recursive segmentation of [\p begin, \p end) to minimise
    //! square residuals for time shifts of a base model \p predictions.
    template<typename ITR>
    static void fitTopDownPiecewiseTimeShifted(ITR begin,
                                               ITR end,
                                               std::size_t offset,
                                               std::size_t minimumSegmentLength,
                                               const TDoubleVecVec& predictions,
                                               double pValueToSegment,
                                               TSizeVec& segmentation);

    //! Compute the residual moments of a least squares linear model fit to
    //! [\p begin, \p end).
    template<typename ITR>
    static TMeanVarAccumulator centredResidualMoments(ITR begin, ITR end, double startTime);

    //! Compute the residual moments of a least squares scaled seasonal model fit to
    //! [\p begin, \p end).
    template<typename ITR>
    static TMeanVarAccumulator
    centredResidualMoments(ITR begin, ITR end, std::size_t offset, const TSeasonality& model);

    //! Compute the moments of the values in [\p begin, \p end) after subtracting
    //! the predictions of \p model.
    template<typename ITR>
    static TMeanVarAccumulator
    residualMoments(ITR begin, ITR end, double startTime, const TRegression& model);

    //! Compute the moments of the values in [\p begin, \p end) after subtracting
    //! \p predictions minimising variance over the outer index.
    template<typename ITR>
    static TMeanVarAccumulatorSizePr
    residualMoments(ITR begin, ITR end, std::size_t offset, const TDoubleVecVec& predictions);

    //! Fit a linear model to the values in [\p begin, \p end).
    template<typename ITR>
    static TRegression fitLinearModel(ITR begin, ITR end, double startTime);

    //! Fit a seasonal model of period \p period to the values [\p begin, \p end).
    template<typename ITR>
    static void fitSeasonalModel(ITR begin,
                                 ITR end,
                                 const TSeasonalComponent& period,
                                 const TPredictor& predictor,
                                 const TScale& scale,
                                 TDoubleVec& result);
};
}
}

#endif // INCLUDE_ml_maths_CTimeSeriesSegmentation_h
