/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CRandomProjectionClusterer_h
#define INCLUDED_ml_maths_CRandomProjectionClusterer_h

#include <maths/CAgglomerativeClusterer.h>
#include <maths/CBasicStatistics.h>
#include <maths/CGramSchmidt.h>
#include <maths/CKMeans.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CNaturalBreaksClassifier.h>
#include <maths/CSampling.h>
#include <maths/CXMeans.h>

#include <boost/array.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ml {
namespace maths {

//! \brief Common functionality for random projection clustering.
//!
//! DESCRIPTION:\n
//! This implements the core functionality for clustering via
//! random projections.
//!
//! The idea is to construct a set of subspaces of low dimension
//! by projecting the data points onto an orthogonalisation of
//! randomly generated vectors. Specifically, this generates a
//! collection of random vectors \f$[x]_i ~ N(0,1)\f$ and then
//! constructs an orthonormal basis by the Gram-Schmidt process.
//!
//! Having generated a number of different random projections
//! of the data, this measures the similarity of the i'th and
//! j'th point by looking at the average probability they belong
//! to the same cluster over the ensemble of clusterings. This
//! step is achieved by associating a generative model, specifically,
//! a weighted mixture of normals with each clustering.
//!
//! Finally, hierarchical agglomerative clustering is performed
//! on the resulting similarity matrix together with model selection
//! to choose the number of clusters.
//!
//! For more details see http://people.ee.duke.edu/~lcarin/random-projection-for-high.pdf
template<std::size_t N>
class CRandomProjectionClusterer {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;

public:
    virtual ~CRandomProjectionClusterer() = default;

    //! Set up the projections.
    virtual bool initialise(std::size_t numberProjections, std::size_t dimension) {
        m_Dimension = dimension;
        if (this->generateProjections(numberProjections) == false) {
            LOG_ERROR(<< "Failed to generate projections");
            return false;
        }
        return true;
    }

protected:
    using TVector = CVector<double>;
    using TVectorVec = std::vector<TVector>;
    using TVectorArray = boost::array<TVector, N>;
    using TVectorArrayVec = std::vector<TVectorArray>;

protected:
    //! Get the random number generator.
    CPRNG::CXorShift1024Mult& rng() const { return m_Rng; }

    //! Get the projections.
    const TVectorArrayVec& projections() const { return m_Projections; }

    //! Generate \p b random projections.
    bool generateProjections(std::size_t b) {
        m_Projections.clear();

        if (b == 0) {
            return true;
        }

        if (m_Dimension <= N) {
            m_Projections.resize(1);
            TVectorArray& projection = m_Projections[0];
            for (std::size_t i = 0u; i < N; ++i) {
                projection[i].extend(m_Dimension, 0.0);
                if (i < m_Dimension) {
                    projection[i](i) = 1.0;
                }
            }
            return true;
        }

        m_Projections.resize(b);

        TDoubleVec components;
        CSampling::normalSample(m_Rng, 0.0, 1.0, b * N * m_Dimension, components);
        for (std::size_t i = 0u; i < b; ++i) {
            TVectorArray& projection = m_Projections[i];
            for (std::size_t j = 0u; j < N; ++j) {
                projection[j].assign(&components[(i * N + j) * m_Dimension],
                                     &components[(i * N + j + 1) * m_Dimension]);
            }

            if (!CGramSchmidt::basis(projection)) {
                LOG_ERROR(<< "Failed to construct basis");
                return false;
            }
        }

        return true;
    }

    //! Extend the projections for an increase in data dimension to \p dimension.
    bool extendProjections(std::size_t dimension) {
        using TDoubleVecArray = boost::array<TDoubleVec, N>;

        if (dimension <= m_Dimension) {
            return true;
        } else if (dimension <= N) {
            TVectorArray& projection = m_Projections[0];
            for (std::size_t i = m_Dimension; i < dimension; ++i) {
                projection[i](i) = 1.0;
            }
            return true;
        }

        std::size_t b = m_Projections.size();
        std::size_t d = dimension - m_Dimension;
        double alpha = static_cast<double>(m_Dimension) / static_cast<double>(dimension);
        double beta = 1.0 - alpha;

        TDoubleVecArray extension;
        TDoubleVec components;
        CSampling::normalSample(m_Rng, 0.0, 1.0, b * N * d, components);
        for (std::size_t i = 0u; i < b; ++i) {
            for (std::size_t j = 0u; j < N; ++j) {
                extension[j].assign(&components[(i * N + j) * d],
                                    &components[(i * N + j + 1) * d]);
            }

            if (!CGramSchmidt::basis(extension)) {
                LOG_ERROR(<< "Failed to construct basis");
                return false;
            }

            for (std::size_t j = 0u; j < N; ++j) {
                scale(extension[j], beta);
                TVector& projection = m_Projections[i][j];
                projection *= alpha;
                projection.reserve(dimension);
                projection.extend(extension[j].begin(), extension[j].end());
            }
        }

        return true;
    }

private:
    //! Scale the values in the vector \p x by \p scale.
    void scale(TDoubleVec& x, double scale) {
        for (std::size_t i = 0u; i < x.size(); ++i) {
            x[i] *= scale;
        }
    }

private:
    //! The random number generator.
    mutable CPRNG::CXorShift1024Mult m_Rng;

    //! The dimension of the data to project.
    std::size_t m_Dimension;

    //! The projections.
    TVectorArrayVec m_Projections;
};

//! \brief Implements random projection clustering for batches of data points.
template<std::size_t N>
class CRandomProjectionClustererBatch : public CRandomProjectionClusterer<N> {
public:
    using TDoubleVec = typename CRandomProjectionClusterer<N>::TDoubleVec;
    using TSizeVec = typename CRandomProjectionClusterer<N>::TSizeVec;
    using TVector = typename CRandomProjectionClusterer<N>::TVector;
    using TVectorVec = typename CRandomProjectionClusterer<N>::TVectorVec;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TSizeUSet = boost::unordered_set<std::size_t>;
    using TVectorNx1 = CVectorNx1<double, N>;
    using TEigenVectorNx1 = typename SDenseVector<TVectorNx1>::Type;
    using TVectorNx1Vec = std::vector<TVectorNx1>;
    using TVectorNx1VecVec = std::vector<TVectorNx1Vec>;
    using TSymmetricMatrixNxN = CSymmetricMatrixNxN<double, N>;
    using TSvdNxN =
        typename SJacobiSvd<typename SDenseMatrix<TSymmetricMatrixNxN>::Type>::Type;
    using TSvdNxNVec = std::vector<TSvdNxN>;
    using TSvdNxNVecVec = std::vector<TSvdNxNVec>;
    using TSymmetricMatrix = CSymmetricMatrix<double>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;

public:
    CRandomProjectionClustererBatch(double compression)
        : m_Compression(compression) {}

    virtual ~CRandomProjectionClustererBatch() = default;

    //! Create the \p numberProjections random projections.
    //!
    //! \param[in] numberProjections The number of projections to create.
    //! \param[in] dimension The dimension of the space to project.
    virtual bool initialise(std::size_t numberProjections, std::size_t dimension) {
        m_ProjectedData.resize(numberProjections);
        return this->CRandomProjectionClusterer<N>::initialise(numberProjections, dimension);
    }

    //! Reserve space for \p n data points.
    void reserve(std::size_t n) {
        for (auto& projection : m_ProjectedData) {
            projection.reserve(n);
        }
    }

    //! Add projected data for \p x.
    void add(const TVector& x) {
        for (std::size_t i = 0; i < this->projections().size(); ++i) {
            TVectorNx1 px;
            for (std::size_t j = 0; j < N; ++j) {
                px(j) = this->projections()[i][j].inner(x);
            }
            m_ProjectedData[i].push_back(px);
        }
    }

    //! Compute the clusters.
    //!
    //! \param[in] clusterer The object responsible for clustering the projected
    //! data points.
    //! \return The final agglomerative clustering of the different projections.
    template<typename CLUSTERER>
    TSizeVecVec run(CLUSTERER clusterer) const {
        if (m_ProjectedData.empty()) {
            return;
        }

        std::size_t numberBags{m_ProjectedData.size()};

        // Filled in with the weights of the clusterings.
        TDoubleVecVec W(numberBags);
        // Filled in with the sample means of the clusterings.
        TVectorNx1VecVec M(numberBags);
        // Filled in with the SVDs of the sample covariances of the clusterings.
        TSvdNxNVecVec C(numberBags);
        // Filled in with the sample points indices.
        TSizeUSet I;

        // Compute the projected clusterings and sampling.
        this->clusterProjections(clusterer, W, M, C, I);

        // Compute the sample neighbourhoods.
        TSizeVecVec H{this->neighbourhoods(I)};

        // Compute the neighbourhood similarities.
        TSymmetricMatrix S{this->similarities(W, M, C, H)};

        // Run agglomerative clustering and choose number of clusters.
        return this->clusterNeighbourhoods(std::move(S), H);
    }

protected:
    //! Compute the projected clusterings and find a good sampling of the points
    //! on which to perform agglomerative clustering.
    //!
    //! \param[in] clusterer The object responsible for clustering the projected
    //! data points.
    //! \param[out] W Filled in with the cluster weights.
    //! \param[out] M Filled in with the cluster sample means.
    //! \param[out] C Filled in with the SVD of cluster sample covariance matrices.
    //! \param[out] I Filled in with the indices of distinct sampled points.
    template<typename CLUSTERER>
    void clusterProjections(CLUSTERER clusterer,
                            TDoubleVecVec& W,
                            TVectorNx1VecVec& M,
                            TSvdNxNVecVec& C,
                            TSizeUSet& I) const {
        using TTaggedVectorNx1 = CAnnotatedVector<TVectorNx1, std::size_t>;
        using TTaggedVectorNx1Vec = std::vector<TTaggedVectorNx1>;
        using TSampleCovariancesNxN = CBasicStatistics::SSampleCovariances<TVectorNx1>;

        std::size_t numberBags{m_ProjectedData.size()};
        std::size_t numberPoints{this->numberPoints()};

        // A placeholder for copy of i'th projected data.
        TTaggedVectorNx1Vec projection;
        // Filled in with the probabilities sampling each point in a single cluster
        // from one bag.
        TDoubleVec probabilities;
        // Filled in with a mapping from each point in a single cluster of one bag
        // to the index of their corresponding data point.
        TSizeVec indices;
        // Filled in with the samples of a single cluster from one bag.
        TSizeVec samples;

        projection.reserve(numberPoints);

        for (std::size_t bag = 0; bag < numberBags; ++bag) {
            LOG_TRACE(<< "projection " << bag);

            projection.clear();
            for (const auto& point : m_ProjectedData[bag]) {
                projection.emplace_back(point, projection.size());
            }

            // Cluster the projected data.
            clusterer.setPoints(projection);
            clusterer.run();
            const auto& clusters{clusterer.clusters()};
            double numberClusters{static_cast<double>(clusters.size())};
            LOG_TRACE(<< "# clusters = " << numberClusters);

            for (const auto& cluster_ : clusters) {
                const TTaggedVectorNx1Vec& cluster{cluster_.points()};
                LOG_TRACE(<< "# points = " << cluster.size());

                // Compute the number of points to sample from this cluster.
                double weight{static_cast<double>(cluster.size()) /
                              static_cast<double>(numberPoints)};
                std::size_t numberSamples{static_cast<std::size_t>(
                    std::max(m_Compression * weight * numberClusters, 1.0))};
                LOG_TRACE(<< "weight = " << weight << ", numberSamples = " << numberSamples);

                // Compute the cluster sample mean and covariance matrix.
                TSampleCovariancesNxN covariances(N);
                covariances.add(cluster);
                TVectorNx1 mb{CBasicStatistics::mean(covariances)};
                TSvdNxN Cb(toDenseMatrix(CBasicStatistics::covariances(covariances)),
                           Eigen::ComputeFullU | Eigen::ComputeFullV);

                // Compute the likelihood that a sample from the cluster is a given
                // point in the cluster.
                probabilities.clear();
                indices.clear();
                probabilities.reserve(cluster.size());
                indices.reserve(cluster.size());
                for (const auto& point : cluster) {
                    if (I.count(point.annotation()) == 0) {
                        TEigenVectorNx1 x{toDenseVector(point - mb)};
                        probabilities.push_back(-0.5 * x.transpose() * Cb.solve(x));
                        indices.push_back(point.annotation());
                    }
                }

                if (probabilities.size() > 0) {
                    this->normalizeLikelihoods(probabilities);
                    LOG_TRACE(<< "probabilities = "
                              << core::CContainerPrinter::print(probabilities));

                    // Sample the cluster.
                    CSampling::categoricalSampleWithoutReplacement(
                        this->rng(), probabilities, numberSamples, samples);
                    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));

                    // Save the relevant data for the i'th clustering.
                    for (const auto& sample : samples) {
                        I.insert(indices[sample]);
                    }
                }
                W[bag].push_back(weight);
                M[bag].push_back(mb);
                C[bag].push_back(Cb);
            }
        }
    }

    //! Construct the neighbourhoods of each of the seed point.
    //!
    //! \param[in] seeds The indices of distinct sampled points which define
    //! the neighbourhoods. In particular, a neighbourhood comprises the set
    //! of points closest to a particular seed point.
    //! \return The neighbourhoods of each point in \p seeds, i.e. the indices
    //! of the closest points.
    TSizeVecVec neighbourhoods(const TSizeUSet& seeds) const {

        using TTaggedVector = CAnnotatedVector<TVector, std::size_t>;
        using TTaggedVectorVec = std::vector<TTaggedVector>;

        LOG_TRACE(<< "seeds = " << core::CContainerPrinter::print(seeds));
        std::size_t numberBags{m_ProjectedData.size()};
        std::size_t numberPoints{this->numberPoints()};

        // Create a k-d tree of the concatination of the projections of the
        // seed data points.
        TTaggedVectorVec S;
        S.reserve(seeds.size());
        TVector concat(numberBags * N);
        for (auto seed : seeds) {
            for (std::size_t bag = 0; bag < numberBags; ++bag) {
                for (std::size_t i = 0; i < N; ++i) {
                    concat(N * bag + i) = m_ProjectedData[bag][seed](i);
                }
            }
            LOG_TRACE(<< "concat = " << concat);
            S.emplace_back(concat, S.size());
        }
        CKdTree<TTaggedVector> lookup;
        lookup.build(std::move(S));

        // Compute the neighbourhoods.
        TSizeVecVec neighbourhoods(seeds.size());
        for (std::size_t point = 0; point < numberPoints; ++point) {
            for (std::size_t bag = 0; bag < numberBags; ++bag) {
                for (std::size_t i = 0; i < N; ++i) {
                    concat(N * bag + i) = m_ProjectedData[bag][point](i);
                }
            }
            const TTaggedVector* nn{lookup.nearestNeighbour(concat)};
            if (nn == nullptr) {
                LOG_ERROR(<< "No nearest neighbour of " << concat);
            } else {
                LOG_TRACE(<< "nn = " << *nn);
                neighbourhoods[nn->annotation()].push_back(point);
            }
        }
        LOG_TRACE(<< "neighbourhoods = " << core::CContainerPrinter::print(neighbourhoods));

        return neighbourhoods;
    }

    //! Compute the similarities between neighbourhoods.
    //!
    //! \param[in] W The cluster weights.
    //! \param[in] M The cluster sample means.
    //! \param[in] C The SVD of cluster sample covariance matrices.
    //! \param[in] H The neighbourhoods of each point in \p I, i.e. the indices
    //! of the closest points.
    //! \return The mean similarities between neighbourhoods over the different
    //! clusterings.
    TSymmetricMatrix similarities(const TDoubleVecVec& W,
                                  const TVectorNx1VecVec& M,
                                  const TSvdNxNVecVec& C,
                                  const TSizeVecVec& H) const {

        std::size_t numberBags{m_ProjectedData.size()};
        std::size_t numberNeighbourhoods{H.size()};

        TSymmetricMatrix similarities(numberNeighbourhoods);

        // The probabilities each neighbourhood is from each cluster.
        TVectorVec P(numberNeighbourhoods);

        for (std::size_t bag = 0; bag < numberBags; ++bag) {
            const TVectorNx1Vec& X = m_ProjectedData[bag];
            const TDoubleVec& Wb = W[bag];
            const TVectorNx1Vec& Mb = M[bag];
            const TSvdNxNVec& Cb = C[bag];
            LOG_TRACE(<< "W[bag] = " << core::CContainerPrinter::print(Wb));
            LOG_TRACE(<< "M[bag] = " << core::CContainerPrinter::print(Mb));

            std::size_t numberClusters{Mb.size()};
            std::fill_n(P.begin(), numberNeighbourhoods, TVector(numberClusters));

            // Compute the log likelihood each neighbourhood is from each cluster.
            for (std::size_t cluster = 0; cluster < numberClusters; ++cluster) {
                double Z{std::log(Wb[cluster]) - 0.5 * this->logDeterminant(Cb[cluster])};
                LOG_TRACE(<< "  Z(" << projection << "," << cluster << ") = " << Z);
                for (std::size_t h = 0; h < numberNeighbourhoods; ++h) {
                    auto& likelihood = P[h];
                    likelihood(cluster) = static_cast<double>(H[h].size()) * Z;
                    for (auto point : H[h]) {
                        TEigenVectorNx1 x{toDenseVector(X[point] - Mb[cluster])};
                        likelihood(cluster) -= 0.5 * x.transpose() * Cb[cluster].solve(x);
                    }
                    LOG_TRACE(<< "    P(" << h << "," << cluster
                              << ") = " << likelihood(cluster));
                }
            }

            // Compute the probabilities each neighbourhood is from each cluster.
            for (std::size_t h = 0; h < numberNeighbourhoods; ++h) {
                auto& probabilities = P[h];
                this->normalizeLikelihoods(probabilities);
                LOG_TRACE(<< "  P(" << h << ") = " << probability);
            }

            // Update the similarities with the results from this clustering.
            for (std::size_t i = 0; i < numberNeighbourhoods; ++i) {
                for (std::size_t j = 0; j <= i; ++j) {
                    similarities(i, j) +=
                        -CTools::fastLog(std::max(
                            P[i].inner(P[j]), std::numeric_limits<double>::min())) /
                        static_cast<double>(numberBags);
                }
            }
        }

        return similarities;
    }

    //! Extract the clustering of the neighbourhoods based on their similarities.
    //!
    //! \param[in] S The similarities between neighbourhoods.
    //! \param[in] H The neighbourhoods.
    //! \return The clustering of the underlying points.
    TSizeVecVec clusterNeighbourhoods(TSymmetricMatrix S, const TSizeVecVec& H) const {
        using TNode = CAgglomerativeClusterer::CNode;
        using TDoubleTuple = CNaturalBreaksClassifier::TDoubleTuple;
        using TDoubleTupleVec = CNaturalBreaksClassifier::TDoubleTupleVec;

        // Compute an average linkage agglomerative clustering of the neighbourhoods.
        CAgglomerativeClusterer agglomerative;
        agglomerative.initialize(std::move(S));
        CAgglomerativeClusterer::TNodeVec tree;
        agglomerative.run(CAgglomerativeClusterer::E_Average, tree);

        TSizeVecVec clustering;

        // Get the natural break in the agglomerative clustering node heights.
        TDoubleTupleVec heights;
        heights.reserve(tree.size());
        for (const auto& node : tree) {
            heights.push_back(TDoubleTuple());
            heights.back().add(node.height());
        }
        LOG_TRACE(<< "heights = " << core::CContainerPrinter::print(heights));
        TSizeVec split;
        if (CNaturalBreaksClassifier::naturalBreaks(
                heights,
                2, // Number splits
                0, // Minimum cluster size
                CNaturalBreaksClassifier::E_TargetDeviation, split)) {

            double height{CBasicStatistics::mean(heights[split[0] - 1])};
            LOG_TRACE(<< "split = " << core::CContainerPrinter::print(splits)
                      << ", height = " << height);

            const TNode& root{tree.back()};
            root.clusteringAt(height, clustering);

            for (std::size_t cluster = 0; cluster < clustering.size(); ++cluster) {
                TSizeVec expanded;
                std::size_t size{0};
                for (auto i : clustering[cluster]) {
                    size += H[i].size();
                }
                expanded.reserve(size);

                for (auto i : clustering[cluster]) {
                    expanded.insert(expanded.end(), H[i].begin(), H[i].end());
                }
                clustering[cluster] = std::move(expanded);
            }
        } else {
            LOG_ERROR(<< "Failed to cluster " << core::CContainerPrinter::print(heights));
        }

        return clustering;
    }

    //! Get the projected data points.
    const TVectorNx1VecVec& projectedData() const { return m_ProjectedData; }

private:
    std::size_t numberPoints() const {
        return m_ProjectedData.empty() ? 0 : m_ProjectedData[0].size();
    }

    double logDeterminant(const TSvdNxN& svd) const {
        double result{0.0};
        for (std::size_t i = 0, rank = static_cast<std::size_t>(svd.rank()); i < rank; ++i) {
            result += CTools::fastLog(svd.singularValues()[i]);
        }
        return result;
    }

    template<typename VECTOR>
    void normalizeLikelihoods(VECTOR& likelihoods) const {
        double Z{0.0};
        double lmax{*std::max_element(likelihoods.begin(), likelihoods.end())};
        for (auto& l : likelihoods) {
            l = std::exp(l - lmax);
            Z += l;
        }
        for (auto& l : likelihoods) {
            l /= Z;
        }
    }

private:
    //! Controls the amount of compression in sampling points for computing the
    //! hierarchical clustering. Larger numbers equate to more sampled points so
    //! less compression.
    double m_Compression;

    //! The projected data points.
    TVectorNx1VecVec m_ProjectedData;
};

//! \brief Adapts x-means for use by the random projection clusterer.
template<std::size_t N, typename COST>
class CRandomProjectionXMeansClusterer {
public:
    using TTaggedVectorNx1 = CAnnotatedVector<CVectorNx1<double, N>, std::size_t>;
    using TTaggedVectorNx1Vec = std::vector<TTaggedVectorNx1>;
    using TClusterer = CXMeans<TTaggedVectorNx1, COST>;
    using TClusterVec = typename TClusterer::TClusterVec;

public:
    CRandomProjectionXMeansClusterer(std::size_t kmax,
                                     std::size_t improveParamsKmeansIterations,
                                     std::size_t improveStructureClusterSeeds,
                                     std::size_t improveStructureKmeansIterations)
        : m_XMeans(kmax),
          m_ImproveParamsKmeansIterations(improveParamsKmeansIterations),
          m_ImproveStructureClusterSeeds(improveStructureClusterSeeds),
          m_ImproveStructureKmeansIterations(improveStructureKmeansIterations) {}

    //! Set the points to cluster.
    void setPoints(TTaggedVectorNx1Vec& points) { m_XMeans.setPoints(points); }

    //! Cluster the points.
    void run() {
        m_XMeans.run(m_ImproveParamsKmeansIterations, m_ImproveStructureClusterSeeds,
                     m_ImproveStructureKmeansIterations);
    }

    //! Get the clusters (should only be called after run).
    const TClusterVec& clusters() const { return m_XMeans.clusters(); }

private:
    //! The x-means implementation.
    TClusterer m_XMeans;
    //! The number of iterations to use in k-means for a single
    //! round of improve parameters.
    std::size_t m_ImproveParamsKmeansIterations;
    //! The number of random seeds to try when initializing k-means
    //! for a single round of improve structure.
    std::size_t m_ImproveStructureClusterSeeds;
    //! The number of iterations to use in k-means for a single
    //! round of improve structure.
    std::size_t m_ImproveStructureKmeansIterations;
};

//! Makes an x-means adapter for random projection clustering.
template<std::size_t N, typename COST>
CRandomProjectionXMeansClusterer<N, COST>
randomProjectionXMeansClusterer(std::size_t kmax,
                                std::size_t improveParamsKmeansIterations,
                                std::size_t improveStructureClusterSeeds,
                                std::size_t improveStructureKmeansIterations) {
    return CRandomProjectionXMeansClusterer<N, COST>(
        kmax, improveParamsKmeansIterations, improveStructureClusterSeeds,
        improveStructureKmeansIterations);
}

//! \brief Adapts k-means for use by the random projection clusterer.
template<std::size_t N>
class CRandomProjectionKMeansClusterer {
public:
    using TTaggedVectorNx1 = CAnnotatedVector<CVectorNx1<double, N>, std::size_t>;
    using TTaggedVectorNx1Vec = std::vector<TTaggedVectorNx1>;
    using TClusterer = CKMeans<TTaggedVectorNx1>;
    using TClusterVec = typename TClusterer::TClusterVec;

public:
    CRandomProjectionKMeansClusterer(std::size_t k, std::size_t maxIterations)
        : m_K(k), m_MaxIterations(maxIterations) {}

    //! Set the points to cluster.
    void setPoints(TTaggedVectorNx1Vec& points) {
        m_KMeans.setPoints(points);
        TTaggedVectorNx1Vec centres;
        CKMeansPlusPlusInitialization<TTaggedVectorNx1, CPRNG::CXorShift1024Mult> seedCentres(m_Rng);
        seedCentres.run(points, m_K, centres);
        m_KMeans.setCentres(centres);
    }

    //! Cluster the points.
    void run() { m_KMeans.run(m_MaxIterations); }

    //! Get the clusters (should only be called after run).
    const TClusterVec& clusters() const {
        m_KMeans.clusters(m_Clusters);
        return m_Clusters;
    }

private:
    //! The random number generator.
    CPRNG::CXorShift1024Mult m_Rng;
    //! The k-means implementation.
    TClusterer m_KMeans;
    //! The number of clusters to use.
    std::size_t m_K;
    //! The number of iterations to use in k-means.
    std::size_t m_MaxIterations;
    //! The clusters.
    mutable TClusterVec m_Clusters;
};

//! Makes a k-means adapter for random projection clustering.
template<std::size_t N>
CRandomProjectionKMeansClusterer<N>
randomProjectionKMeansClusterer(std::size_t k, std::size_t maxIterations) {
    return CRandomProjectionKMeansClusterer<N>(k, maxIterations);
}
}
}

#endif // INCLUDED_ml_maths_CRandomProjectionClusterer_h
