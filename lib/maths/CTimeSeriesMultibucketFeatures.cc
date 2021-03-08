/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesMultibucketFeatures.h>

#include <core/CPersistUtils.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CSolvers.h>
#include <maths/CTypeTraits.h>
#include <maths/MathsTypes.h>

#include <boost/circular_buffer.hpp>

#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ml {
namespace maths {
template<typename T, typename STORAGE, typename PREDICTOR>
class CTimeSeriesMultibucketMeanImpl {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using Type = T;
    using TFeature = CTimeSeriesMultibucketFeature<T, PREDICTOR>;
    using TType1Vec = typename TFeature::TType1Vec;
    using TWeightsAry1Vec = typename TFeature::TWeightsAry1Vec;
    using TType1VecTWeightAry1VecPr = typename TFeature::TType1VecTWeightAry1VecPr;
    using TFloatMeanAccumulator = typename CBasicStatistics::SSampleMean<STORAGE>::TAccumulator;
    using TValueFunc = std::function<STORAGE(core_t::TTime, const TFloatMeanAccumulator&)>;
    using TWeightFunc = std::function<double(const TFloatMeanAccumulator&)>;

public:
    explicit CTimeSeriesMultibucketMeanImpl(std::size_t length = 0)
        : m_SlidingWindow(length) {}

    TType1VecTWeightAry1VecPr value(const TValueFunc& value) const {
        if (4 * m_SlidingWindow.size() >= 3 * m_SlidingWindow.capacity()) {
            return {{this->mean(value)}, {maths_t::countWeight(this->count())}};
        }
        return {{}, {}};
    }

    double correlationWithBucketValue() const {
        // This follows from the weighting applied to values in the window and
        // linearity of expectation.
        double r{WINDOW_GEOMETRIC_WEIGHT * WINDOW_GEOMETRIC_WEIGHT};
        double length{static_cast<double>(m_SlidingWindow.size())};
        return length == 0.0 ? 0.0 : std::sqrt((1.0 - r) / (1.0 - std::pow(r, length)));
    }

    void clear() { m_SlidingWindow.clear(); }

    void add(core_t::TTime time,
             core_t::TTime bucketLength,
             const TType1Vec& values,
             const TWeightsAry1Vec& weights) {

        // Remove any old samples.
        core_t::TTime cutoff{time - this->windowInterval(bucketLength)};
        while (m_SlidingWindow.size() > 0 && m_SlidingWindow.front().first < cutoff) {
            m_SlidingWindow.pop_front();
        }
        if (values.size() > 0) {
            using TStorage1Vec = core::CSmallVector<STORAGE, 1>;

            // Get the scales to apply to each value.
            TStorage1Vec scales;
            scales.reserve(weights.size());
            for (const auto& weight : weights) {
                using std::sqrt;
                scales.push_back(sqrt(STORAGE{maths_t::countVarianceScale(weight)} *
                                      STORAGE{maths_t::seasonalVarianceScale(weight)}));
            }

            // Add the next sample.
            TFloatMeanAccumulator next{this->conformable(STORAGE(values[0]), 0.0f)};
            for (std::size_t i = 0; i < values.size(); ++i) {
                auto weight = maths_t::countForUpdate(weights[i]);
                next.add(STORAGE(values[i]) / scales[i], this->minweight(weight));
            }
            m_SlidingWindow.push_back({time, next});
        }
    }

    std::uint64_t checksum(std::uint64_t seed = 0) const {
        seed = CChecksum::calculate(seed, m_SlidingWindow.capacity());
        return CChecksum::calculate(seed, m_SlidingWindow);
    }

    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        core::CMemoryDebug::dynamicSize("m_SlidingWindow", m_SlidingWindow, mem);
    }

    std::size_t memoryUsage() const {
        return core::CMemory::dynamicSize(m_SlidingWindow);
    }

    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name{traverser.name()};
            RESTORE_SETUP_TEARDOWN(CAPACITY_TAG, std::size_t capacity,
                                   core::CStringUtils::stringToType(traverser.value(), capacity),
                                   m_SlidingWindow.set_capacity(capacity))
            RESTORE(SLIDING_WINDOW_TAG,
                    core::CPersistUtils::restore(SLIDING_WINDOW_TAG, m_SlidingWindow, traverser))
        } while (traverser.next());
        return true;
    }
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertValue(CAPACITY_TAG, m_SlidingWindow.capacity());
        core::CPersistUtils::persist(SLIDING_WINDOW_TAG, m_SlidingWindow, inserter);
    }

private:
    using TDoubleMeanAccumulator = typename SPromoted<TFloatMeanAccumulator>::Type;
    using TTimeFloatMeanAccumulatorPr = std::pair<core_t::TTime, TFloatMeanAccumulator>;
    using TTimeFloatMeanAccumulatorPrCBuf = boost::circular_buffer<TTimeFloatMeanAccumulatorPr>;

private:
    //! The geometric weight applied to the window.
    constexpr static const double WINDOW_GEOMETRIC_WEIGHT = 0.9;
    //! The state tags.
    static const std::string CAPACITY_TAG;
    static const std::string SLIDING_WINDOW_TAG;

private:
    //! Get the length of the window given \p bucketLength.
    core_t::TTime windowInterval(core_t::TTime bucketLength) const {
        return static_cast<core_t::TTime>(m_SlidingWindow.capacity()) * bucketLength;
    }

    //! Compute the weighted mean of the values in the window.
    Type mean(const TValueFunc& value) const {
        return this->evaluateOnWindow(value, [](const TFloatMeanAccumulator& x) {
            return static_cast<double>(CBasicStatistics::count(x));
        });
    }

    //! Compute the weighted count of the values in the window.
    Type count() const {
        return this->evaluateOnWindow(
            [&](core_t::TTime, const TFloatMeanAccumulator& x) {
                return this->conformable(CBasicStatistics::mean(x),
                                         static_cast<double>(CBasicStatistics::count(x)));
            },
            [](const TFloatMeanAccumulator&) { return 1.0; });
    }

    //! Compute the weighted mean of \p value on the sliding window.
    Type evaluateOnWindow(const TValueFunc& value, const TWeightFunc& weight) const {
        double latest;
        double earliest;
        std::tie(earliest, latest) = this->range();
        double n{static_cast<double>(m_SlidingWindow.size())};
        double scale{(n - 1.0) * (latest == earliest ? 1.0 : 1.0 / (latest - earliest))};
        auto i = m_SlidingWindow.begin();
        TDoubleMeanAccumulator mean{this->conformable(value(i->first, i->second), 0.0)};
        for (double last{earliest}; i != m_SlidingWindow.end(); ++i) {
            double dt{static_cast<double>(i->first) - last};
            last = static_cast<double>(i->first);
            mean.age(std::pow(WINDOW_GEOMETRIC_WEIGHT, scale * dt));
            mean.add(value(i->first, i->second), weight(i->second));
        }
        return this->toVector(CBasicStatistics::mean(mean));
    }

    //! Compute the time range of window.
    TDoubleDoublePr range() const {
        auto range = std::accumulate(m_SlidingWindow.begin(), m_SlidingWindow.end(),
                                     CBasicStatistics::CMinMax<double>(),
                                     [](CBasicStatistics::CMinMax<double> partial,
                                        const TTimeFloatMeanAccumulatorPr& value) {
                                         partial.add(static_cast<double>(value.first));
                                         return partial;
                                     });
        return {range.min(), range.max()};
    }

    //! Return weight.
    static CFloatStorage minweight(double weight) { return weight; }
    //! Return the minimum weight.
    template<std::size_t N>
    static CFloatStorage minweight(const core::CSmallVector<double, N>& weight) {
        return *std::min_element(weight.begin(), weight.end());
    }

    //! Univariate implementation returns \p value.
    template<typename U, typename V>
    static V conformable(const U& /*x*/, V value) {
        return value;
    }
    //! Multivariate implementation returns the \p value scalar multiple of the
    //! one vector which is conformable with \p x.
    template<typename U, typename V>
    static CVector<V> conformable(const CVector<U>& x, V value) {
        return CVector<V>(x.dimension(), value);
    }

    //! Univariate implementation returns 1.
    template<typename U>
    static std::size_t dimension(const U& /*x*/) {
        return 1;
    }
    //! Multivariate implementation returns the dimension of \p x.
    template<typename U>
    static std::size_t dimension(const CVector<U>& x) {
        return x.dimension();
    }

    //! Univariate implementation returns \p x.
    template<typename U>
    static double toVector(const U& x) {
        return x;
    }
    //! Multivariate implementation returns \p x as the VECTOR type.
    template<typename U>
    static Type toVector(const CVector<U>& x) {
        return x.template toVector<Type>();
    }

private:
    //! The window values.
    TTimeFloatMeanAccumulatorPrCBuf m_SlidingWindow;
};

template<typename T, typename STORAGE, typename PREDICTOR>
const std::string CTimeSeriesMultibucketMeanImpl<T, STORAGE, PREDICTOR>::CAPACITY_TAG{"a"};
template<typename T, typename STORAGE, typename PREDICTOR>
const std::string CTimeSeriesMultibucketMeanImpl<T, STORAGE, PREDICTOR>::SLIDING_WINDOW_TAG{"b"};

CTimeSeriesMultibucketScalarMean::CTimeSeriesMultibucketScalarMean(std::size_t length)
    : m_Impl{std::make_unique<TImpl>(length)} {
}

CTimeSeriesMultibucketScalarMean::CTimeSeriesMultibucketScalarMean(const CTimeSeriesMultibucketScalarMean& other)
    : m_Impl{std::make_unique<TImpl>(*other.m_Impl)} {
}

CTimeSeriesMultibucketScalarMean::~CTimeSeriesMultibucketScalarMean() = default;
CTimeSeriesMultibucketScalarMean::CTimeSeriesMultibucketScalarMean(
    CTimeSeriesMultibucketScalarMean&&) noexcept = default;

CTimeSeriesMultibucketScalarMean& CTimeSeriesMultibucketScalarMean::
operator=(CTimeSeriesMultibucketScalarMean&&) noexcept = default;
CTimeSeriesMultibucketScalarMean& CTimeSeriesMultibucketScalarMean::
operator=(const CTimeSeriesMultibucketScalarMean& other) {
    if (this != &other) {
        CTimeSeriesMultibucketScalarMean tmp{*this};
        *this = std::move(tmp);
    }
    return *this;
}

CTimeSeriesMultibucketScalarMean::TPtr CTimeSeriesMultibucketScalarMean::clone() const {
    return std::make_unique<CTimeSeriesMultibucketScalarMean>(*this);
}

CTimeSeriesMultibucketScalarMean::TType1VecTWeightAry1VecPr
CTimeSeriesMultibucketScalarMean::value(core_t::TTime maximumShift,
                                        const TPredictor& predictor) const {
    core_t::TTime timeShift{this->likelyShift(maximumShift, predictor)};
    auto value = [&](core_t::TTime time, const TImpl::TFloatMeanAccumulator& mean) {
        double x{CBasicStatistics::mean(mean)};
        return x - predictor(time + timeShift);
    };
    return m_Impl->value(value);
}

double CTimeSeriesMultibucketScalarMean::correlationWithBucketValue() const {
    return m_Impl->correlationWithBucketValue();
}

void CTimeSeriesMultibucketScalarMean::clear() {
    m_Impl->clear();
}

void CTimeSeriesMultibucketScalarMean::add(core_t::TTime time,
                                           core_t::TTime bucketLength,
                                           const TType1Vec& values,
                                           const TWeightsAry1Vec& weights) {
    m_Impl->add(time, bucketLength, values, weights);
}

std::uint64_t CTimeSeriesMultibucketScalarMean::checksum(std::uint64_t seed) const {
    return m_Impl->checksum(seed);
}

void CTimeSeriesMultibucketScalarMean::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesMultibucketScalarMean");
    core::CMemoryDebug::dynamicSize("m_Impl", m_Impl, mem);
}

std::size_t CTimeSeriesMultibucketScalarMean::staticSize() const {
    return sizeof(*this);
}

std::size_t CTimeSeriesMultibucketScalarMean::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Impl);
}

bool CTimeSeriesMultibucketScalarMean::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    return m_Impl->acceptRestoreTraverser(traverser);
}

void CTimeSeriesMultibucketScalarMean::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    m_Impl->acceptPersistInserter(inserter);
}

core_t::TTime CTimeSeriesMultibucketScalarMean::likelyShift(core_t::TTime maximumShift,
                                                            const TPredictor& predictor) const {
    std::array<double, 6> times;
    double range{2 * static_cast<double>(maximumShift)};
    double step{range / static_cast<double>(times.size() - 1)};
    times[0] = -range / 2.0;
    for (std::size_t i = 1; i < times.size(); ++i) {
        times[i] = times[i - 1] + step;
    }

    auto loss = [&](double time) {
        return std::fabs(predictor(static_cast<core_t::TTime>(time + 0.5)));
    };

    double shiftedTime;
    double lossAtShiftedTime;
    CSolvers::globalMinimize(times, loss, shiftedTime, lossAtShiftedTime);
    LOG_TRACE(<< "shift = " << static_cast<core_t::TTime>(shiftedTime + 0.5)
              << ", loss(shift) = " << lossAtShiftedTime);

    return static_cast<core_t::TTime>(shiftedTime + 0.5);
}

CTimeSeriesMultibucketVectorMean::CTimeSeriesMultibucketVectorMean(std::size_t length)
    : m_Impl{std::make_unique<TImpl>(length)} {
}

CTimeSeriesMultibucketVectorMean::CTimeSeriesMultibucketVectorMean(const CTimeSeriesMultibucketVectorMean& other)
    : m_Impl{std::make_unique<TImpl>(*other.m_Impl)} {
}

CTimeSeriesMultibucketVectorMean::~CTimeSeriesMultibucketVectorMean() = default;
CTimeSeriesMultibucketVectorMean::CTimeSeriesMultibucketVectorMean(
    CTimeSeriesMultibucketVectorMean&&) noexcept = default;

CTimeSeriesMultibucketVectorMean& CTimeSeriesMultibucketVectorMean::
operator=(CTimeSeriesMultibucketVectorMean&&) noexcept = default;
CTimeSeriesMultibucketVectorMean& CTimeSeriesMultibucketVectorMean::
operator=(const CTimeSeriesMultibucketVectorMean& other) {
    if (this != &other) {
        CTimeSeriesMultibucketVectorMean tmp{*this};
        *this = std::move(tmp);
    }
    return *this;
}

CTimeSeriesMultibucketVectorMean::TPtr CTimeSeriesMultibucketVectorMean::clone() const {
    return std::make_unique<CTimeSeriesMultibucketVectorMean>(*this);
}

CTimeSeriesMultibucketVectorMean::TType1VecTWeightAry1VecPr
CTimeSeriesMultibucketVectorMean::value(core_t::TTime maximumShift,
                                        const TPredictor& predictor) const {
    core_t::TTime timeShift{this->likelyShift(maximumShift, predictor)};
    auto value = [&](core_t::TTime time, const TImpl::TFloatMeanAccumulator& mean) {
        auto x = CBasicStatistics::mean(mean);
        return x - predictor(time + timeShift);
    };
    return m_Impl->value(value);
}

double CTimeSeriesMultibucketVectorMean::correlationWithBucketValue() const {
    return m_Impl->correlationWithBucketValue();
}

void CTimeSeriesMultibucketVectorMean::clear() {
    m_Impl->clear();
}

void CTimeSeriesMultibucketVectorMean::add(core_t::TTime time,
                                           core_t::TTime bucketLength,
                                           const TType1Vec& values,
                                           const TWeightsAry1Vec& weights) {
    m_Impl->add(time, bucketLength, values, weights);
}

std::uint64_t CTimeSeriesMultibucketVectorMean::checksum(std::uint64_t seed) const {
    return m_Impl->checksum(seed);
}

void CTimeSeriesMultibucketVectorMean::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTimeSeriesMultibucketVectorMean");
    core::CMemoryDebug::dynamicSize("m_Impl", m_Impl, mem);
}

std::size_t CTimeSeriesMultibucketVectorMean::staticSize() const {
    return sizeof(*this);
}

std::size_t CTimeSeriesMultibucketVectorMean::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Impl);
}

bool CTimeSeriesMultibucketVectorMean::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    return m_Impl->acceptRestoreTraverser(traverser);
}

void CTimeSeriesMultibucketVectorMean::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    m_Impl->acceptPersistInserter(inserter);
}

core_t::TTime CTimeSeriesMultibucketVectorMean::likelyShift(core_t::TTime maximumShift,
                                                            const TPredictor& predictor) const {
    std::array<double, 6> times;
    double range{2 * static_cast<double>(maximumShift)};
    double step{range / static_cast<double>(times.size() - 1)};
    times[0] = -range / 2.0;
    for (std::size_t i = 1; i < times.size(); ++i) {
        times[i] = times[i - 1] + step;
    }

    auto loss = [&](double time) {
        return predictor(static_cast<core_t::TTime>(time + 0.5)).L1();
    };

    double shiftedTime;
    double lossAtShiftedTime;
    CSolvers::globalMinimize(times, loss, shiftedTime, lossAtShiftedTime);
    LOG_TRACE(<< "shift = " << static_cast<core_t::TTime>(shiftedTime + 0.5)
              << ", loss(shift) = " << lossAtShiftedTime);

    return static_cast<core_t::TTime>(shiftedTime + 0.5);
}
}
}
