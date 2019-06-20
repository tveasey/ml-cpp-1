/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFrameUtils.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPackedBitVector.h>
#include <core/Concurrency.h>

#include <maths/CBasicStatistics.h>
#include <maths/CMathsFuncs.h>
#include <maths/CMic.h>
#include <maths/CPRNG.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>

#include <vector>

namespace ml {
namespace maths {
using TRowItr = core::CDataFrame::TRowItr;
namespace {
const std::size_t NUMBER_SAMPLES_TO_COMPUTE_MIC{10000};
}

bool CDataFrameUtils::standardizeColumns(std::size_t numberThreads, core::CDataFrame& frame) {

    using TMeanVarAccumulatorVec =
        std::vector<CBasicStatistics::SSampleMeanVar<double>::TAccumulator>;

    if (frame.numberRows() == 0 || frame.numberColumns() == 0) {
        return true;
    }

    auto computeColumnMoments = core::bindRetrievableState(
        [](TMeanVarAccumulatorVec& moments, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                    if (isMissing((*row)[i]) == false) {
                        moments[i].add((*row)[i]);
                    }
                }
            }
        },
        TMeanVarAccumulatorVec(frame.numberColumns()));

    auto results = frame.readRows(numberThreads, computeColumnMoments);
    if (results.second == false) {
        LOG_ERROR(<< "Failed to standardise columns");
        return false;
    }

    TMeanVarAccumulatorVec moments(frame.numberColumns());
    for (const auto& result : results.first) {
        for (std::size_t i = 0; i < moments.size(); ++i) {
            moments[i] += result.s_FunctionState[i];
        }
    }

    TDoubleVec mean(moments.size());
    TDoubleVec scale(moments.size());
    for (std::size_t i = 0; i < moments.size(); ++i) {
        double variance{CBasicStatistics::variance(moments[i])};
        mean[i] = CBasicStatistics::mean(moments[i]);
        scale[i] = variance == 0.0 ? 1.0 : 1.0 / std::sqrt(variance);
    }

    LOG_TRACE(<< "means = " << core::CContainerPrinter::print(mean));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scale));

    auto standardiseColumns = [&mean, &scale](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                row->writeColumn(i, scale[i] * ((*row)[i] - mean[i]));
            }
        }
    };

    return frame.writeColumns(numberThreads, standardiseColumns).second;
}

bool CDataFrameUtils::columnQuantiles(std::size_t numberThreads,
                                      const core::CDataFrame& frame,
                                      const core::CPackedBitVector& rowMask,
                                      const TSizeVec& columnMask,
                                      const CQuantileSketch& sketch,
                                      TQuantileSketchVec& result,
                                      TWeightFunction weight) {
    result.assign(columnMask.size(), sketch);

    auto results = frame.readRows(
        numberThreads, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](TQuantileSketchVec& quantiles, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (std::size_t i = 0; i < columnMask.size(); ++i) {
                        if (isMissing((*row)[columnMask[i]]) == false) {
                            quantiles[i].add((*row)[columnMask[i]], weight(*row));
                        }
                    }
                }
            },
            std::move(result)),
        &rowMask);

    if (results.second == false) {
        LOG_ERROR(<< "Failed to compute column quantiles");
        return false;
    }

    result = std::move(results.first[0].s_FunctionState);

    for (std::size_t i = 1; i < results.first.size(); ++i) {
        for (std::size_t j = 0; j < columnMask.size(); ++j) {
            result[j] += results.first[i].s_FunctionState[j];
        }
    }

    return true;
}

CDataFrameUtils::TDoubleVec CDataFrameUtils::micWithColumn(const core::CDataFrame& frame,
                                                           const TSizeVec& columnMask,
                                                           std::size_t targetColumn) {

    using TFloatVec = std::vector<CFloatStorage>;
    using TFloatVecVec = std::vector<TFloatVec>;
    using TRowSampler = CSampling::CRandomStreamSampler<TRowRef>;

    TDoubleVec mics(frame.numberColumns(), 0.0);

    if (targetColumn >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: target column out of bounds '"
                     << targetColumn << " >= " << frame.numberColumns()
                     << "'. Please report this problem.");
        return mics;
    }

    std::size_t n{std::min(NUMBER_SAMPLES_TO_COMPUTE_MIC, frame.numberRows())};

    CPRNG::CXorOShiro128Plus rng;
    TFloatVecVec samples;

    if (frame.inMainMemory()) {

        for (auto i : columnMask) {

            auto onSample = [&](std::size_t slot, const TRowRef& row) {
                if (slot >= samples.size()) {
                    samples.resize(slot + 1, TFloatVec(2));
                }
                samples[slot][0] = row[i];
                samples[slot][1] = row[targetColumn];
            };
            TRowSampler sampler{n, onSample};

            auto result = frame.readRows(
                1, core::bindRetrievableState(
                       [&](std::size_t& missing, TRowItr beginRows, TRowItr endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               if (isMissing((*row)[i])) {
                                   ++missing;
                               } else if (isMissing((*row)[targetColumn]) == false) {
                                   sampler.sample(*row);
                               }
                           }
                       },
                       std::size_t{0}));
            LOG_TRACE(<< "# samples = " << samples.size());

            double fractionMissing{static_cast<double>(result.first[0].s_FunctionState) /
                                   static_cast<double>(frame.numberRows())};
            LOG_TRACE(<< "Fraction missing = " << fractionMissing);

            CMic mic;
            mic.reserve(samples.size());
            for (const auto& sample : samples) {
                mic.add(sample[0], sample[1]);
            }

            mics[i] = (1.0 - fractionMissing) * mic.compute();
            samples.clear();
        }

    } else {

        auto onSample = [&](std::size_t slot, const TRowRef& row) {
            if (slot >= samples.size()) {
                samples.resize(slot + 1, TFloatVec(row.numberColumns()));
            }
            row.copyTo(samples[slot].begin());
        };
        TRowSampler sampler{n, onSample};

        auto results = frame.readRows(
            1, core::bindRetrievableState(
                   [&](TSizeVec& missing, TRowItr beginRows, TRowItr endRows) {
                       for (auto row = beginRows; row != endRows; ++row) {
                           for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                               missing[i] += isMissing((*row)[i]) ? 1 : 0;
                           }
                           if (isMissing((*row)[targetColumn]) == false) {
                               sampler.sample(*row);
                           }
                       }
                   },
                   TSizeVec(frame.numberColumns(), 0)));
        LOG_TRACE(<< "# samples = " << samples.size());

        TDoubleVec fractionMissing(frame.numberColumns());
        for (std::size_t i = 0; i < fractionMissing.size(); ++i) {
            for (const auto& result : results.first) {
                fractionMissing[i] += static_cast<double>(result.s_FunctionState[i]) /
                                      static_cast<double>(frame.numberRows());
            }
        }
        LOG_TRACE(<< "Fraction missing = " << core::CContainerPrinter::print(fractionMissing));

        for (auto i : columnMask) {
            if (i != targetColumn) {
                CMic mic;
                mic.reserve(samples.size());
                for (const auto& sample : samples) {
                    if (isMissing(sample[i]) == false) {
                        mic.add(sample[i], sample[targetColumn]);
                    }
                }
                mics[i] = (1.0 - fractionMissing[i]) * mic.compute();
            }
        }
    }

    return mics;
}

CDataFrameUtils::TDoubleVecVec
CDataFrameUtils::categoryFrequencies(std::size_t numberThreads,
                                     const core::CDataFrame& frame,
                                     const TSizeVec& columnMask) {

    TDoubleVecVec result(frame.numberColumns());

    if (frame.numberRows() == 0 || frame.numberColumns() == 0) {
        return result;
    }

    const auto& columnIsCategorical = frame.columnIsCategorical();
    if (std::find_if(columnMask.begin(), columnMask.end(), [&](std::size_t i) {
            return columnIsCategorical[i];
        }) == columnMask.end()) {
        return result;
    }

    auto results = frame.readRows(
        numberThreads,
        core::bindRetrievableState(
            [&](TDoubleVecVec& counts, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (std::size_t i : columnMask) {
                        if (columnIsCategorical[i]) {
                            std::size_t id{static_cast<std::size_t>((*row)[i])};
                            counts[i].resize(std::max(counts[i].size(), id + 1), 0.0);
                            counts[i][id] += 1.0;
                        }
                    }
                }
            },
            TDoubleVecVec(frame.numberColumns())));

    if (results.second == false) {
        HANDLE_FATAL(<< "Internal error: failed to calculate category"
                     << " frequencies. Please report this problem.");
        return result;
    }

    for (const auto& counts_ : results.first) {
        for (std::size_t i = 0; i < counts_.s_FunctionState.size(); ++i) {
            result[i].resize(counts_.s_FunctionState[i].size());
        }
    }
    for (const auto& counts_ : results.first) {
        for (std::size_t i = 0; i < counts_.s_FunctionState.size(); ++i) {
            for (std::size_t j = 0; j < counts_.s_FunctionState[i].size(); ++j) {
                result[i][j] += counts_.s_FunctionState[i][j] /
                                static_cast<double>(frame.numberRows());
            }
        }
    }

    return result;
}

CDataFrameUtils::TDoubleVecVec
CDataFrameUtils::meanValueOfTargetForCategories(std::size_t numberThreads,
                                                const core::CDataFrame& frame,
                                                const TSizeVec& columnMask,
                                                std::size_t targetColumn) {

    TDoubleVecVec result(frame.numberColumns());

    if (targetColumn >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: target column out of bounds '"
                     << targetColumn << " >= " << frame.numberColumns()
                     << "'. Please report this problem.");
        return result;
    }

    if (frame.numberRows() == 0 || frame.numberColumns() == 0) {
        return result;
    }

    using TMeanAccumulatorVec = std::vector<CBasicStatistics::SSampleMean<double>::TAccumulator>;
    using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;

    const auto& columnIsCategorical = frame.columnIsCategorical();
    if (std::find_if(columnMask.begin(), columnMask.end(), [&](std::size_t i) {
            return columnIsCategorical[i];
        }) == columnMask.end()) {
        return result;
    }

    auto results = frame.readRows(
        numberThreads,
        core::bindRetrievableState(
            [&](TMeanAccumulatorVecVec& means, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (std::size_t i : columnMask) {
                        if (columnIsCategorical[i]) {
                            std::size_t id{static_cast<std::size_t>((*row)[i])};
                            means[i].resize(std::max(means[i].size(), id + 1));
                            means[i][id].add((*row)[targetColumn]);
                        }
                    }
                }
            },
            TMeanAccumulatorVecVec(frame.numberColumns())));

    if (results.second == false) {
        HANDLE_FATAL(<< "Internal error: failed to calculate mean target value"
                     << " for categories. Please report this problem.");
        return result;
    }

    TMeanAccumulatorVecVec means(frame.numberColumns());
    for (const auto& means_ : results.first) {
        for (std::size_t i = 0; i < means_.s_FunctionState.size(); ++i) {
            means[i].resize(means_.s_FunctionState[i].size());
        }
    }
    for (const auto& means_ : results.first) {
        for (std::size_t i = 0; i < means_.s_FunctionState.size(); ++i) {
            for (std::size_t j = 0; j < means_.s_FunctionState[i].size(); ++j) {
                means[i][j] += means_.s_FunctionState[i][j];
            }
        }
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i].resize(means[i].size());
        for (std::size_t j = 0; j < means[i].size(); ++j) {
            result[i][j] = CBasicStatistics::mean(means[i][j]);
        }
    }

    return result;
}

bool CDataFrameUtils::isMissing(double x) {
    return CMathsFuncs::isFinite(x) == false;
}

double CDataFrameUtils::unitWeight(const TRowRef&) {
    return 1.0;
}
}
}
