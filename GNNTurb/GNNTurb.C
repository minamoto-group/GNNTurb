/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2016 OpenFOAM Foundation
    Copyright (C) 2019-2022 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "GNNTurb.H"
#include "fvOptions.H"
#include "wallDist.H"


namespace Foam 
{
namespace LESModels 
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template <class BasicTurbulenceModel>
void GNNTurb<BasicTurbulenceModel>::correctTauij() {
    if (debug)
    {
        InfoInFunction << this->runTime_.timeName() << endl;
    }

    Info << "GNNTurb:  " << "Processing input features" << endl;

    //* calculate uDelta
    correctUDelta();

    //* calculate UFlucDelta
    correctUFlucDelta();

    //* calculate nondimensionalized features
    this->expDWallDelta_.primitiveFieldRef() = exp(-wallDist::New(this->mesh_).y() / this->adjustedDelta());
    volScalarField& expDWallDelta = this->expDWallDelta_;

    // OpenFOAM's grad(U) is different from solid mechanics.
    // https://cfd.direct/openfoam/tensor-mathematics/
    volTensorField nondimGradU = T(fvc::grad(this->U_)) * this->adjustedDelta() / this->uDelta_; 
    volVectorField nondimUFlucDelta = this->UFlucDelta_ / this->uDelta_; 

    expDWallDelta.correctBoundaryConditions();
    nondimGradU.correctBoundaryConditions();
    nondimUFlucDelta.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(expDWallDelta);
    fv::options::New(this->mesh_).correct(nondimGradU);
    fv::options::New(this->mesh_).correct(nondimUFlucDelta);

    // get surface fields
    const surfaceScalarField expDWallDeltaSF = fvc::interpolate(expDWallDelta); 
    const surfaceTensorField nondimGradUSF = fvc::interpolate(nondimGradU); 
    const surfaceVectorField nondimUFlucDeltaSF = fvc::interpolate(nondimUFlucDelta); 

    Info << "GNNTurb:  " << "Mapping input features to graph data" << endl;
    geometry::CellFaceStarGraphData<CellFaceStarGraphType>& data = this->Data_;
    volSymmTensorField& tauij = this->tauij_;

    forAll (inputNames_, fieldi)
    {
        const word name = this->inputNames_[fieldi];

        if (name == "expDWallDelta") {
            data.map_field<volScalarField, surfaceScalarField, volScalarField::Boundary>
                (expDWallDelta, expDWallDeltaSF, expDWallDelta.boundaryField(), 1.0, fieldi);
        }
        else if (this->gradUComponents_.found(name)) {
            data.map_field<volTensorField, surfaceTensorField, volTensorField::Boundary>
                (nondimGradU, nondimGradUSF, nondimGradU.boundaryField(),
                this->gradUComponents_[name], this->inputMaxs_[fieldi], fieldi);
        }
        else if (this->UFlucDeltaComponents_.found(name)) {
            data.map_field<volVectorField, surfaceVectorField, volVectorField::Boundary>
                (nondimUFlucDelta, nondimUFlucDeltaSF, nondimUFlucDelta.boundaryField(),
                this->UFlucDeltaComponents_[name], this->inputMaxs_[fieldi], fieldi); 
        }
        else {
            FatalErrorInFunction << name << " is not defined in GNNTurb"
                << nl << exit(FatalError);
        }
    }

    Info << "GNNTurb:  " << "Predicting tau_ij" << endl; //* predict tau_{ij}^{SGS}
    at::Tensor output_ten = this->model_.forward({
            data.x.to(this->device_), 
            data.edge_index.to(this->device_), 
            data.edge_attr.to(this->device_), 
            data.batch.to(this->device_)
        }).toTensor();

    output_ten = output_ten.to(torch::kCPU);
    auto out_accessor = output_ten.accessor<float, 2>(); // for faster access

    //* process outputs (re-dimensionalize & scaling)
    Info << "GNNTurb:  " << "Processing output features" << endl;
    forAll(tauij, celli)
    {
        tauij[celli].xx() = (scalar)out_accessor[celli][0] * pow(this->uDelta_[celli], 2) * this->outputMaxs_[0];
        tauij[celli].xy() = (scalar)out_accessor[celli][1] * pow(this->uDelta_[celli], 2) * this->outputMaxs_[1];
        tauij[celli].xz() = (scalar)out_accessor[celli][2] * pow(this->uDelta_[celli], 2) * this->outputMaxs_[2];
        tauij[celli].yy() = (scalar)out_accessor[celli][3] * pow(this->uDelta_[celli], 2) * this->outputMaxs_[3];
        tauij[celli].yz() = (scalar)out_accessor[celli][4] * pow(this->uDelta_[celli], 2) * this->outputMaxs_[4];
        tauij[celli].zz() = (scalar)out_accessor[celli][5] * pow(this->uDelta_[celli], 2) * this->outputMaxs_[5];
    }

    this->tauij_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->tauij_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template <class BasicTurbulenceModel>
GNNTurb<BasicTurbulenceModel>::GNNTurb
(
    const alphaField &alpha,
    const rhoField &rho,
    const volVectorField &U,
    const surfaceScalarField &alphaRhoPhi,
    const surfaceScalarField &phi,
    const transportModel &transport,
    const word &propertiesName,
    const word &type
)
: 
    LESDirectTauij<BasicTurbulenceModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    device_(torch::kCPU),

    inputNames_(this->coeffDict_.lookup("inputNames")),
    inputMaxs_(inputNames_.size()),
    outputMaxs_(
        {
        readScalar(this->coeffDict_.subDict("scalingCoeffs").lookup("tau11")),
        readScalar(this->coeffDict_.subDict("scalingCoeffs").lookup("tau12")),
        readScalar(this->coeffDict_.subDict("scalingCoeffs").lookup("tau13")),
        readScalar(this->coeffDict_.subDict("scalingCoeffs").lookup("tau22")),
        readScalar(this->coeffDict_.subDict("scalingCoeffs").lookup("tau23")),
        readScalar(this->coeffDict_.subDict("scalingCoeffs").lookup("tau33"))
        }
    ),
    Ne_(readScalar(this->coeffDict_.lookup("Ne"))),

    adjustedDelta_(
        IOobject
        (
            IOobject::groupName("adjustedDelta", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("adjustedDelta", dimLength, 1e-6),
        extrapolatedCalculatedFvPatchScalarField::typeName
    ),

    expDWallDelta_(
        IOobject
        (
            IOobject::groupName("expDWallDelta", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    uDelta_(
        IOobject
        (
            IOobject::groupName("uDelta", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("uDelta", dimVelocity, 1e-6),
        extrapolatedCalculatedFvPatchScalarField::typeName
    ),

    UFlucDelta_(
        IOobject
        (
            IOobject::groupName("UFlucDelta", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedVector("UFlucDelta", dimVelocity, vector::zero)
    )
{
    if (type == typeName) {
        this->printCoeffs(type);
    }

    forAll(this->inputMaxs_, i) {
        this->inputMaxs_[i] 
            = readScalar(this->coeffDict_.subDict("scalingCoeffs").lookup(inputNames_[i]));
    }

    this->gradUComponents_.set("dudx", tensor::XX);
    this->gradUComponents_.set("dudy", tensor::XY);
    this->gradUComponents_.set("dudz", tensor::XZ);
    this->gradUComponents_.set("dvdx", tensor::YX);
    this->gradUComponents_.set("dvdy", tensor::YY);
    this->gradUComponents_.set("dvdz", tensor::YZ);
    this->gradUComponents_.set("dwdx", tensor::ZX);
    this->gradUComponents_.set("dwdy", tensor::ZY);
    this->gradUComponents_.set("dwdz", tensor::ZZ);
    this->UFlucDeltaComponents_.set("u_fluc_delta", vector::X);
    this->UFlucDeltaComponents_.set("v_fluc_delta", vector::Y);
    this->UFlucDeltaComponents_.set("w_fluc_delta", vector::Z);

    torch::manual_seed(8);
    at::globalContext().setBenchmarkCuDNN(true);
    at::globalContext().setDeterministicCuDNN(false);
    torch::autograd::AutoGradMode guard(false);
    torch::NoGradGuard no_grad; // ensures that autograd is off

    if (torch::cuda::is_available()) 
    {
        Info << "GNNTurb:  " << "CUDA is available. ";
        if (Switch(this->coeffDict_.lookup("useCUDA"))) {
            this->device_ = torch::kCUDA;
            Info << "device <- CUDA" << endl;
        }
        else {
            Info << "device <- CPU" << endl;
        }
    }
    else 
    {
        Info << "GNNTurb:  " << "CUDA is not available. "
            << "device <- CPU" << endl;
    }

    // read model path from "turbulencePropaties".
    string modelPath = string(this->coeffDict_.lookup("modelPath"));
    try 
    {
        this->model_ = torch::jit::load(modelPath);
        this->model_.eval();
        this->model_.to(this->device_);
    }
    catch (const c10::Error& e) 
    {
        FatalErrorInFunction
            << "GNNTurb:  " << "Failed to load the model: "
            << modelPath << nl 
            << e.what_without_backtrace() << exit(FatalError);
    }
    Info << "GNNTurb:  " << "Succeeded in loading the model: " 
        << modelPath << nl << endl;


    //* construct graph data
    buildGraph(); 
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template <class BasicTurbulenceModel>
bool GNNTurb<BasicTurbulenceModel>::read() {
    if (LESModel<BasicTurbulenceModel>::read())
    {
        return true;
    }
    else
    {
        return false;
    }
}


template <class BasicTurbulenceModel>
void GNNTurb<BasicTurbulenceModel>::correct() {
    LESModel<BasicTurbulenceModel>::correct();
    correctTauij(); 
}


template <class BasicTurbulenceModel>
const volScalarField& GNNTurb<BasicTurbulenceModel>::adjustedDelta() {
    adjustedDelta_.primitiveFieldRef() = this->delta().primitiveField();
    adjustedDelta_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(adjustedDelta_);
    return adjustedDelta_;
}


template <class BasicTurbulenceModel>
void GNNTurb<BasicTurbulenceModel>::buildGraph() {
    const scalarField& delta = this->adjustedDelta().primitiveField();
    const surfaceScalarField surface_delta = fvc::interpolate(this->adjustedDelta()); // get surface values
    
    std::vector<CellFaceStarGraphType> G;
    
    Info << "GNNTurb:  " << "Constructing graph data..." << endl;

    forAll(this->mesh_.cells(), celli) {
        const cell &c = this->mesh_.cells()[celli];

        std::vector<geometry::FaceIdInfo> neighbor_faceIdInfos;
        List<vector> node_coords(c.size() + 1);
        List<scalar> node_deltas(c.size() + 1);
        node_coords[0] = this->mesh_.cellCentres()[celli]; 
        node_deltas[0] = delta[celli];

        forAll(c, j)
        {
            label facei = c[j];
            if (this->mesh_.isInternalFace(facei)) {
                neighbor_faceIdInfos.emplace_back(geometry::FaceIdInfo(facei)); 
                node_deltas[j + 1] = surface_delta[facei];
            }
            else 
            {
                label patchi = this->mesh_.boundaryMesh().whichPatch(facei);
                const polyPatch &pp = this->mesh_.boundaryMesh()[patchi];
                label pFacei = facei - pp.start();

                neighbor_faceIdInfos.emplace_back(geometry::FaceIdInfo(facei, patchi, pFacei));
                node_deltas[j + 1] = this->delta().boundaryField()[patchi][pFacei];
            }
            node_coords[j + 1] = this->mesh_.faceCentres()[facei]; 
        }

        CellFaceStarGraphType g(celli, neighbor_faceIdInfos, 
                                node_coords, node_deltas, this->Ne_);
        G.emplace_back(g);
    }

    this->Data_ = CellFaceStarGraphData(G, this->mesh_.cells().size(), inputNames_.size());

    if (debug) {
        Info << "Number of neighbor cells(celli = 0): " 
            << (G[0].num_nodes-1) << endl;
        Info << "Number of cells: " << this->mesh_.cells().size() << endl;
        Info << "Data_.x.size(0): " << this->Data_.x.size(0) << endl;
    }
    Info << "GNNTurb:  " << "Finish!" << nl << endl;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam