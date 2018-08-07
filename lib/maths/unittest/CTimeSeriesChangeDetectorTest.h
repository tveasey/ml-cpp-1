/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CTimeSeriesChangeDetectorTest_h
#define INCLUDED_CTimeSeriesChangeDetectorTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <core/CoreTypes.h>

#include <maths/CTimeSeriesChangeDetector.h>

class CTimeSeriesChangeDetectorTest : public CppUnit::TestFixture {
public:
    void testNoChange();
    void testLevelShift();
    void testLinearScale();
    void testTimeShift();
    void testPersist();

    static CppUnit::Test* suite();

private:
    using TDouble2Vec = ml::core::CSmallVector<double, 2>;
    using TGenerator = std::function<double(ml::core_t::TTime)>;
    using TGeneratorVec = std::vector<TGenerator>;
    using TChange = std::function<double(TGenerator generator, ml::core_t::TTime)>;
    using TExtractValue = std::function<double(const ml::maths::SChangeDescription&)>;

private:
    void testChange(const TGeneratorVec& trends,
                    ml::maths::SChangeDescription::EDescription description,
                    TChange applyChange,
                    TExtractValue extractValue,
                    double expectedChange,
                    double maximumFalseNegatives,
                    double maximumMeanBucketsToDetectChange);
};

#endif // INCLUDED_CTimeSeriesChangeDetectorTest_h
