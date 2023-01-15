#pragma once
#include <torch/torch.h>
#include "utils.H"

namespace geometry
{

template<typename Coord, typename Delta>
struct CellFaceStarGraph {
    const int num_nodes; // number of nodes
    const int num_edges; // number of edges
    const int num_neighbors; // number of neighbors
    int64_t reference_cid;
    std::vector<FaceIdInfo> neighbor_faceIdInfos;
    at::Tensor edge_index;
    at::Tensor edge_attr;

    CellFaceStarGraph(): num_nodes(0), num_edges(0) {}

    CellFaceStarGraph(const int64_t reference_cid,
                      const std::vector<FaceIdInfo> &neighbor_faceIdInfos,
                      const Coord &node_coords, 
                      const Delta &node_deltas,
                      const double Ne = 10.0 // normalising factor
        ): 
        num_nodes(neighbor_faceIdInfos.size() + 1), 
        num_edges(2 * (num_nodes - 1)), 
        num_neighbors(num_nodes - 1), 
        reference_cid(reference_cid), 
        neighbor_faceIdInfos(neighbor_faceIdInfos)
    {
        assert((int64_t)neighbor_faceIdInfos.size() + 1 == (int64_t)node_coords.size());
        assert((int64_t)neighbor_faceIdInfos.size() + 1 == (int64_t)node_deltas.size());

        init_tensors(); // initialize tensors

        // edge attr (pseudo coordinates)
        auto ei_accessor = edge_index.accessor<int64_t, 2>(); // to access faster
        auto ea_accessor = edge_attr.accessor<float, 2>();

        for (int i = 0; i < num_edges; ++i) {
            int64_t ei_in = ei_accessor[0][i];
            int64_t ei_out = ei_accessor[1][i];

            double attr_x = node_coords[ei_in].x() - node_coords[ei_out].x();
            double attr_y = node_coords[ei_in].y() - node_coords[ei_out].y();
            double attr_z = node_coords[ei_in].z() - node_coords[ei_out].z();

            double attr_delta;
            attr_delta = node_deltas[0]; // delta at the reference point
            attr_x = attr_x / (Ne * attr_delta) + 0.5;
            attr_y = attr_y / (Ne * attr_delta) + 0.5;
            attr_z = attr_z / (Ne * attr_delta) + 0.5;

            assert(0 <= attr_x && attr_x <= 1.0);
            assert(0 <= attr_y && attr_y <= 1.0);
            assert(0 <= attr_z && attr_z <= 1.0);
            ea_accessor[i][0] = (float)attr_x;
            ea_accessor[i][1] = (float)attr_y;
            ea_accessor[i][2] = (float)attr_z;
        }

        assert(edge_attr.size(0) == num_edges && edge_attr.size(1) == 3);
    }

    void init_tensors() {
        auto options_Long = at::TensorOptions().dtype(torch::kInt64)
                                .device(torch::kCPU).requires_grad(false);
        auto options_Float = at::TensorOptions().dtype(torch::kFloat32)
                                .device(torch::kCPU).requires_grad(false);

        // edge index
        at::Tensor sta_ind = at::cat({at::arange(num_neighbors, options_Long) + 1, 
                                        at::zeros(num_neighbors, options_Long)}, 0);
        at::Tensor end_ind = at::cat({at::zeros(num_neighbors, options_Long), 
                                        at::arange(num_neighbors, options_Long) + 1}, 0);
        edge_index = at::cat({at::unsqueeze(sta_ind, 0), 
                                at::unsqueeze(end_ind, 0)}, 0);
        assert(edge_index.size(0) == 2 && edge_index.size(1) == num_edges);

        // edge attr (pseudo coordinates)
        edge_attr = at::empty({num_edges, 3}, options_Float);
    }
};


template<typename CellFaceStarGraphType>
struct CellFaceStarGraphData {
    at::Tensor x;
    at::Tensor edge_index;
    at::Tensor edge_attr;
    at::Tensor batch;
    std::vector<CellFaceStarGraphType> G;
    std::vector<int64_t> node_slice_index;
    std::vector<int64_t> edge_slice_index;
    int64_t num_nodes; // number of nodes
    int num_features;

    CellFaceStarGraphData(): num_nodes(0), num_features(0) {}

    CellFaceStarGraphData(std::vector<CellFaceStarGraphType> &G_, 
                            const int64_t num_nodes_, 
                            const int num_features_): 
        G(G_), 
        num_nodes(num_nodes_), 
        num_features(num_features_)
    {
        assert((int64_t)G.size() == num_nodes);
        auto options_Long = at::TensorOptions().dtype(torch::kInt64)
                                .device(torch::kCPU).requires_grad(false);
        auto options_Float = at::TensorOptions().dtype(torch::kFloat32)
                                .device(torch::kCPU).requires_grad(false);

        int64_t sum_n = 0, sum_ne = 0;
        for (auto &g : G) {
            sum_n += g.num_nodes;
            sum_ne += g.num_edges;
        }
        x = torch::zeros({sum_n, num_features}, options_Float);

        edge_index = at::empty({2, sum_ne}, options_Long);
        edge_attr = at::empty({sum_ne, 3}, options_Float);
        batch = at::empty({sum_n}, options_Long);
        auto edge_index_accessor = edge_index.accessor<int64_t, 2>(); // to access faster
        auto edge_attr_accessor = edge_attr.accessor<float, 2>();
        auto batch_accessor = batch.accessor<int64_t, 1>();

        node_slice_index.emplace_back(0);
        edge_slice_index.emplace_back(0);
        int64_t now_n = 0, now_ne = 0;
        for (int64_t i = 0; i < (int64_t)G.size(); ++i) {
            CellFaceStarGraphType &g = G[i];
            assert(i == g.reference_cid);

            auto ei_accessor = g.edge_index.template accessor<int64_t, 2>(); 
            auto ea_accessor = g.edge_attr.template accessor<float, 2>();

            for (int64_t j = 0; j < g.num_edges; ++j) {
                // edge index
                edge_index_accessor[0][now_ne + j] = ei_accessor[0][j] + now_n;
                edge_index_accessor[1][now_ne + j] = ei_accessor[1][j] + now_n;
                // edge_attr
                edge_attr_accessor[now_ne + j][0] = ea_accessor[j][0];
                edge_attr_accessor[now_ne + j][1] = ea_accessor[j][1];
                edge_attr_accessor[now_ne + j][2] = ea_accessor[j][2];
            }

            for (int64_t j = 0; j < g.num_nodes; ++j) {
                // batch
                batch_accessor[now_n + j] = i;
            }

            now_n += g.num_nodes;
            now_ne += g.num_edges;
            node_slice_index.emplace_back(now_n);
            edge_slice_index.emplace_back(now_ne);
        }
    }


    // volScalarField, surfaceScalarField, volScalarField::Boundary
    template<typename VSF, typename SSF, typename VSFB> 
    void map_field(const VSF& field, const SSF& surfaceField, const VSFB& boundaryField, 
                    const double maxCoeff, const int k) {
        assert(k < num_features);
        auto x_accessor = x.accessor<float, 2>();

        int64_t now_n = 0;
        for (int64_t i = 0; i < (int64_t)G.size(); ++i) {
            const CellFaceStarGraphType &g = G[i];

            assert(g.reference_cid < field.size());
            assert(now_n < x.size(0));
            x_accessor[now_n][k] = (float)(field[g.reference_cid] / maxCoeff);

            for (int64_t j = 0; j < (int64_t)g.neighbor_faceIdInfos.size(); ++j) {
                const FaceIdInfo &fii = g.neighbor_faceIdInfos[j];
                assert(now_n + (j+1) < x.size(0));

                if (fii.is_internal) {
                    assert(fii.faceid < surfaceField.size());
                    x_accessor[now_n + (j+1)][k] = (float)(surfaceField[fii.faceid] / maxCoeff);
                }
                else {
                    assert(fii.pFacei < boundaryField[fii.patchi].size());
                    x_accessor[now_n + (j+1)][k] 
                        = (float)(boundaryField[fii.patchi][fii.pFacei] / maxCoeff);
                }
            }

            now_n += g.num_nodes;
        }
        assert(now_n == x.size(0));
    }

    // volTensorField, surfaceTensorField, volTensorField::Boundary
    // volVectorField, surfaceVectorField, volVectorField::Boundary
    template<typename VSF, typename SSF, typename VSFB> 
    void map_field(const VSF& field, const SSF& surfaceField, const VSFB& boundaryField, 
                    const int d, const double maxCoeff, const int k) {
        assert(k < num_features);
        auto x_accessor = x.accessor<float, 2>();

        int64_t now_n = 0;
        for (int64_t i = 0; i < (int64_t)G.size(); ++i) {
            const CellFaceStarGraphType &g = G[i];

            assert(g.reference_cid < field.size());
            assert(now_n < x.size(0));
            x_accessor[now_n][k] = (float)(field[g.reference_cid].component(d) / maxCoeff);

            for (int64_t j = 0; j < (int64_t)g.neighbor_faceIdInfos.size(); ++j) {
                const FaceIdInfo &fii = g.neighbor_faceIdInfos[j];
                assert(now_n + (j+1) < x.size(0));

                if (fii.is_internal) {
                    assert(fii.faceid < surfaceField.size());
                    x_accessor[now_n + (j+1)][k] 
                        = (float)(surfaceField[fii.faceid].component(d) / maxCoeff);
                }
                else {
                    assert(fii.pFacei < boundaryField[fii.patchi].size());
                    x_accessor[now_n + (j+1)][k] 
                        = (float)(boundaryField[fii.patchi][fii.pFacei].component(d) / maxCoeff);
                }
            }

            now_n += g.num_nodes;
        }
        assert(now_n == x.size(0));
    }
};

} // namespace geometry
