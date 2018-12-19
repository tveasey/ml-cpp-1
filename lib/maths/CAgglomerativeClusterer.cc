/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CAgglomerativeClusterer.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/Concurrency.h>
#include <core/CoreTypes.h>

#include <maths/COrderings.h>
#include <maths/CSetTools.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

namespace {

using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TDoubleSizeSizePrPr = std::pair<double, TSizeSizePr>;
using TDoubleSizeSizePrPrVec = std::vector<TDoubleSizeSizePrPr>;
using TSizeVec = CAgglomerativeClusterer::TSizeVec;
using TDoubleVec = CAgglomerativeClusterer::TDoubleVec;
using TDoubleVecVec = CAgglomerativeClusterer::TDoubleVecVec;
using TNode = CAgglomerativeClusterer::CNode;
using TNodeVec = CAgglomerativeClusterer::TNodeVec;
using TNodePtrVec = std::vector<TNode*>;

const double INF{std::numeric_limits<double>::max()};

//! Get the distance between node \p i and \p j.
inline double& distance(TDoubleVecVec& distanceMatrix, std::size_t i, std::size_t j) {
    if (j > i) {
        std::swap(i, j);
    }
    return distanceMatrix[i][j];
}

//! Get the distance between node \p i and \p j.
inline double distance(const TDoubleVecVec& distanceMatrix, std::size_t i, std::size_t j) {
    if (j > i) {
        std::swap(i, j);
    }
    return distanceMatrix[i][j];
}

//! \brief Complete update distance update function.
//!
//! The distance between clusters is given by
//! <pre class="fragment">
//!   \f$\displaystyle \max_{a \in A, b \in B}{d[a,b]}\f$
//! </pre>
struct SComplete {
    void operator()(const TDoubleVec& /*sizes*/,
                    std::size_t x,
                    std::size_t a,
                    std::size_t b,
                    TDoubleVecVec& distanceMatrix) const {
        distance(distanceMatrix, b, x) = std::max(distance(distanceMatrix, a, x),
                                                  distance(distanceMatrix, b, x));
    }
};

//! \brief Average objective distance update function.
//!
//! The distance between clusters is given by
//! <pre class="fragment">
//!   \f$\displaystyle \frac{1}{|A||B|}\sum_{a \in A, b \in B}{d[a,b]}\f$
//! </pre>
struct SAverage {
    void operator()(const TDoubleVec& sizes,
                    std::size_t x,
                    std::size_t a,
                    std::size_t b,
                    TDoubleVecVec& distanceMatrix) const {
        double sa{sizes[a]};
        double sb{sizes[b]};
        distance(distanceMatrix, b, x) = (sa * distance(distanceMatrix, a, x) +
                                          sb * distance(distanceMatrix, b, x)) /
                                         (sa + sb);
    }
};

//! \brief Weighted objective distance update function.
struct SWeighted {
    void operator()(const TDoubleVec /*sizes*/,
                    std::size_t x,
                    std::size_t a,
                    std::size_t b,
                    TDoubleVecVec& distanceMatrix) const {
        distance(distanceMatrix, b, x) =
            (distance(distanceMatrix, a, x) + distance(distanceMatrix, b, x)) / 2.0;
    }
};

//! \brief Ward objective distance update function.
//!
//! See https://en.wikipedia.org/wiki/Ward%27s_method.
struct SWard {
    void operator()(const TDoubleVec sizes,
                    std::size_t x,
                    std::size_t a,
                    std::size_t b,
                    TDoubleVecVec& distanceMatrix) const {
        double sa{sizes[a]};
        double sb{sizes[b]};
        double sx{sizes[x]};
        distance(distanceMatrix, b, x) =
            std::sqrt((sa + sx) * distance(distanceMatrix, a, x) +
                      (sb + sx) * distance(distanceMatrix, b, x) -
                      sx * distance(distanceMatrix, a, b)) /
            (sa + sb + sx);
    }
};

//! MST-LINKAGE algorithm due to Rolhf and given in Mullner.
//!
//! This lifts the subset of the minimum spanning tree algorithm needed for building
//! hierarchical clusterings.
//!
//! For details see http://arxiv.org/pdf/1109.2378.pdf.
//!
//! \param[in] distanceMatrix the matrices of distances between the points to cluster.
//! \param[in] L Filled in with the unsorted dendrogram.
void mstCluster(const TDoubleVecVec& distanceMatrix, TDoubleSizeSizePrPrVec& L) {

    L.clear();
    std::size_t N{distanceMatrix.size()};

    if (N <= 1) {
        return;
    }

    L.reserve(N - 1);

    TSizeVec S(N);
    TDoubleVec D(N, INF);

    std::iota(S.begin(), S.end(), 0);

    std::size_t c{S[N - 1]};
    while (S.size() > 1) {
        S.erase(std::lower_bound(S.begin(), S.end(), c));

        std::size_t n{0};
        double d{INF};

        // Implements
        //
        // for (std::size_t x : S) {
        //     D[x] = std::min(D[x], distance(distanceMatrix, x, c));
        //     if (D[x] < d) {
        //         n = x;
        //         d = D[x];
        //     }
        // }
        auto results = core::parallel_for_each(
            S.begin(), S.end(),
            core::bindRetrievableState(
                [&](std::pair<std::size_t, double>& nearest, std::size_t x) {
                    D[x] = std::min(D[x], distance(distanceMatrix, x, c));
                    if (D[x] < nearest.second) {
                        nearest.first = x;
                        nearest.second = D[x];
                    }
                },
                std::make_pair(n, d)));
        for (const auto& result : results) {
            if (result.s_FunctionState.second < d) {
                n = result.s_FunctionState.first;
                d = result.s_FunctionState.second;
            }
        }

        L.emplace_back(d, std::make_pair(std::min(c, n), std::max(c, n)));
        c = n;
    }
}

//! The NN-CHAIN-LINKAGE cluster algorithm due to Murtagh and given in Mullner.
//!
//! This makes use of the fact that reciprocal nearest neighbours eventually get
//! clustered together by some node of the stepwise dendrogram for a certain class
//! of objective function.
//!
//! For details see http://arxiv.org/pdf/1109.2378.pdf.
//!
//! \param[in,out] distanceMatrix the matrices of distances between the points to
//! cluster.
//! \param[in] update The distance update function which varies based on the
//! objective function.
//! \param[in] L Filled in with the unsorted dendrogram.
//! \note This has worst case O(N^2) complexity.
//! \note For maximum efficiency modifications are made in place to \p distanceMatrix.
template<typename UPDATE>
void nnCluster(TDoubleVecVec& distanceMatrix, UPDATE update, TDoubleSizeSizePrPrVec& L) {
    // In departure from the scheme given by Mullner we make all our updates in-place
    // by using a direct address table from n -> max(a, b), where n is the new node
    // index and a and b are reciprocal nearest neighbours. It is still possible to
    // build the stepwise dendrogram from the resulting unsorted dendrogram, we just
    // need to keep track of the highest node created for each index when building
    // the tree. See buildTree for details.

    L.clear();
    std::size_t N{distanceMatrix.size()};

    if (N <= 1) {
        return;
    }

    L.reserve(N - 1);

    TSizeVec S(N);
    TSizeVec chain;
    TDoubleVec size(N, 1.0);
    TSizeVec rightmost;

    std::iota(S.begin(), S.end(), 0);
    chain.reserve(N);
    rightmost.reserve(2 * N - 1);
    rightmost.resize(N);
    std::iota(rightmost.begin(), rightmost.end(), 0);

    std::size_t a{0};
    std::size_t b{1};
    std::size_t p{N - 1};

    while (S.size() > 1) {
        std::size_t m{chain.size()};
        if (m <= 3) {
            a = S[0];
            b = S[1];
            chain.clear();
            chain.push_back(a);
            m = 1;
        } else {
            a = chain[m - 4];
            b = chain[m - 3];
            // Cut the tail.
            chain.pop_back();
            chain.pop_back();
            chain.pop_back();
            m -= 3;
        }

        LOG_TRACE(<< "chain = " << core::CContainerPrinter::print(chain));
        LOG_TRACE(<< "a = " << a << ", b = " << b << ", m = " << m);

        double d;
        do {
            std::size_t ra{rightmost[a]};

            std::size_t c{0};
            d = INF;

            auto pos = std::lower_bound(S.begin(), S.end(), b);
            if (pos != S.end() && *pos == b) {
                c = *pos;
                d = distance(distanceMatrix, ra, rightmost[*pos]);
            }

            // Implements
            //
            // for (std::size_t x : S) {
            //     if (a != x) {
            //     	std::size_t rx{rightmost[x]};
            //         double dx{distance(distanceMatrix, ra, rx)};
            //         if (dx < d) {
            //             c = x;
            //             d = dx;
            //         }
            //     }
            // }
            auto results = core::parallel_for_each(
                S.begin(), S.end(),
                core::bindRetrievableState(
                    [&](std::pair<std::size_t, double>& nearest, std::size_t x) {
                        if (a != x) {
                            std::size_t rx{rightmost[x]};
                            double dx{distance(distanceMatrix, ra, rx)};
                            if (dx < nearest.second) {
                                nearest = std::make_pair(x, dx);
                            }
                        }
                    },
                    std::make_pair(c, d)));
            for (const auto& result : results) {
                if (result.s_FunctionState.second < d) {
                    c = result.s_FunctionState.first;
                    d = result.s_FunctionState.second;
                }
            }

            b = a;
            a = c;
            chain.push_back(a);
            ++m;
        } while (m <= 3 || a != chain[m - 3]);

        if (a > b) {
            std::swap(a, b);
        }
        std::size_t ra{rightmost[a]};
        std::size_t rb{rightmost[b]};

        LOG_TRACE(<< "chain = " << core::CContainerPrinter::print(chain));
        LOG_TRACE(<< "d = " << d << ", a = " << a << ", b = " << b << ", rightmost a = "
                  << ra << ", rightmost b " << rb << ", m = " << m);

        // a and b are reciprocal nearest neighbors.
        L.emplace_back(d, std::make_pair(ra, rb));

        // Update the index set, the distance matrix, the sizes and the rightmost
        // direct address table.
        std::size_t merged[]{a, b};
        CSetTools::inplace_set_difference(S, merged, merged + 2);
        for (std::size_t x : S) {
            update(size, rightmost[x], ra, rb, distanceMatrix);
        }
        size[rb] += size[ra];
        S.push_back(++p);
        rightmost[p] = rb;
    }
}

//! Add a node to the end of the tree with height \p height.
TNode& addNode(TNodeVec& tree, double height) {
    tree.emplace_back(tree.size(), height);
    return tree.back();
}

//! Build the binary hierarchical clustering tree from the unsorted dendrogram
//! representation in \p heights.
//!
//! \param[in,out] heights The nodes which are merged and the level at which they
//! are merged. This can contain repeated node indices, in which case the later
//! indices refer to the last node created at that index. Note that these are
//! (stably) sorted.
//! \param[out] tree A binary tree representing the stepwise dendrogram.
void buildTree(TDoubleSizeSizePrPrVec& heights, TNodeVec& tree) {
    tree.clear();

    std::size_t n{heights.size()};
    if (n == 0) {
        return;
    }

    tree.reserve(2 * n + 1);
    for (std::size_t i = 0; i <= n; ++i) {
        tree.emplace_back(i, 0.0);
    }

    std::stable_sort(heights.begin(), heights.end(), COrderings::SFirstLess());
    LOG_TRACE(<< "heights = " << core::CContainerPrinter::print(heights));

    for (const auto& height : heights) {
        double h{height.first};
        std::size_t j{height.second.first};
        std::size_t k{height.second.second};
        LOG_TRACE(<< "Joining " << j << " and " << k << " at height " << h);
        TNode& parent = addNode(tree, h);
        parent.addChild(tree[j].root());
        parent.addChild(tree[k].root());
    }
}
}

bool CAgglomerativeClusterer::initialize(TDoubleVecVec& distanceMatrix) {
    // Check that the matrix is square.
    std::size_t n{distanceMatrix.size()};
    for (std::size_t i = 0; i < n; ++i) {
        LOG_TRACE(<< "D = " << core::CContainerPrinter::print(distanceMatrix[i]));
        if (distanceMatrix[i].size() != i + 1) {
            LOG_ERROR(<< "Distance matrix isn't upper triangular");
            return false;
        }
    }
    m_DistanceMatrix.swap(distanceMatrix);

    m_Pi.resize(n);
    m_Lambda.resize(n, INF);
    m_M.resize(n);

    std::iota(m_Pi.begin(), m_Pi.end(), 0);

    return true;
}

void CAgglomerativeClusterer::run(EObjective objective, TNodeVec& tree) {
    if (m_DistanceMatrix.empty()) {
        return;
    }

    TDoubleSizeSizePrPrVec heights;

    switch (objective) {
    case E_Single:
        mstCluster(m_DistanceMatrix, heights);
        break;
    case E_Complete:
        nnCluster(m_DistanceMatrix, SComplete(), heights);
        break;
    case E_Average:
        nnCluster(m_DistanceMatrix, SAverage(), heights);
        break;
    case E_Weighted:
        nnCluster(m_DistanceMatrix, SWeighted(), heights);
        break;
    case E_Ward:
        nnCluster(m_DistanceMatrix, SWard(), heights);
        break;
    }

    buildTree(heights, tree);
}

////// CNode //////

CAgglomerativeClusterer::CNode::CNode(std::size_t index, double height)
    : m_Index{index}, m_Height{height} {
}

bool CAgglomerativeClusterer::CNode::addChild(CNode& child) {
    if (m_LeftChild == nullptr) {
        m_LeftChild = &child;
        child.m_Parent = this;
        return true;
    }
    if (m_RightChild == nullptr) {
        m_RightChild = &child;
        child.m_Parent = this;
        return true;
    }

    LOG_ERROR(<< "Trying to add third child");

    return false;
}

std::size_t CAgglomerativeClusterer::CNode::index() const {
    return m_Index;
}

double CAgglomerativeClusterer::CNode::height() const {
    return m_Height;
}

TNode& CAgglomerativeClusterer::CNode::root() {
    CNode* result{this};
    for (CNode* parent = m_Parent; parent; parent = parent->m_Parent) {
        result = parent;
    }
    return *result;
}

void CAgglomerativeClusterer::CNode::points(TSizeVec& result) const {
    if (m_LeftChild == nullptr && m_RightChild == nullptr) {
        result.push_back(m_Index);
    }
    if (m_LeftChild != nullptr) {
        m_LeftChild->points(result);
    }
    if (m_RightChild != nullptr) {
        m_RightChild->points(result);
    }
}

void CAgglomerativeClusterer::CNode::clusters(TDoubleSizeVecPrVec& result) const {
    if (m_LeftChild != nullptr && m_RightChild != nullptr) {
        TSizeVec points;
        this->points(points);
        result.emplace_back(m_Height, points);
    }
    if (m_LeftChild != nullptr) {
        m_LeftChild->clusters(result);
    }
    if (m_RightChild != nullptr) {
        m_RightChild->clusters(result);
    }
}

void CAgglomerativeClusterer::CNode::clusteringAt(double height, TSizeVecVec& result) const {
    if (height >= m_Height) {
        result.emplace_back();
        this->points(result.back());
    } else {
        if (m_LeftChild != nullptr && height < m_LeftChild->height()) {
            m_LeftChild->clusteringAt(height, result);
        } else if (m_LeftChild != nullptr) {
            result.emplace_back();
            m_LeftChild->points(result.back());
        }
        if (m_RightChild != nullptr && height < m_RightChild->height()) {
            m_RightChild->clusteringAt(height, result);
        } else if (m_RightChild != nullptr) {
            result.emplace_back();
            m_RightChild->points(result.back());
        }
    }
}

std::string CAgglomerativeClusterer::CNode::print(const std::string& indent) const {
    std::string result;
    result += "height = " + core::CStringUtils::typeToStringPretty(m_Height);
    if (m_LeftChild != nullptr) {
        result += core_t::LINE_ENDING + indent + m_LeftChild->print(indent + "  ");
    }
    if (m_RightChild != nullptr) {
        result += core_t::LINE_ENDING + indent + m_RightChild->print(indent + "  ");
    }
    if (m_LeftChild == nullptr && m_RightChild == nullptr) {
        result += ", point = " + core::CStringUtils::typeToStringPretty(m_Index);
    }
    return result;
}
}
}
