/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBoostedTreeUtils.h>

#include <maths/CBoostedTree.h>
#include <maths/CBoostedTreeLoss.h>

namespace ml {
namespace maths {
namespace boosted_tree_detail {
using namespace boosted_tree;

TSizeAlignmentPrVec extraColumns(std::size_t numberLossParameters) {
    return {{numberLossParameters, core::CAlignment::E_Unaligned},
            {numberLossParameters, core::CAlignment::E_Aligned16},
            {numberLossParameters * numberLossParameters, core::CAlignment::E_Unaligned},
            {1, core::CAlignment::E_Unaligned}};
}

void zeroPrediction(const TRowDataRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Prediction] + i, 0.0);
    }
}

void zeroLossGradient(const TRowDataRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Gradient] + i, 0.0);
    }
}

void writeLossGradient(const TRowDataRef& row,
                       const TSizeVec& extraColumns,
                       const CLoss& loss,
                       const TMemoryMappedFloatVector& prediction,
                       double actual,
                       double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Gradient] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.gradient(prediction, actual,
                  [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

void zeroLossCurvature(const TRowDataRef& row,
                       const TSizeVec& extraColumns,
                       std::size_t numberLossParameters) {
    for (std::size_t i = 0, size = lossHessianUpperTriangleSize(numberLossParameters);
         i < size; ++i) {
        row.writeColumn(extraColumns[E_Curvature] + i, 0.0);
    }
}

void writeLossCurvature(const TRowDataRef& row,
                        const TSizeVec& extraColumns,
                        const CLoss& loss,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Curvature] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.curvature(prediction, actual,
                   [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}
}
}
}
