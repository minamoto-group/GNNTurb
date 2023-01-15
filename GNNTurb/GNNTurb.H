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

Class
    Foam::LESModels::GNNTurb

Description
    The GNN based SGS model.

SourceFiles
    GNNTurb.C

\*---------------------------------------------------------------------------*/

#ifndef GNNTurb_H
#define GNNTurb_H

#include "LESModel.H"
#include "LESDirectTauij/LESDirectTauij.H"
#include "geometry/cell_face_graph.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam 
{
namespace LESModels
{

template <class BasicTurbulenceModel>
class GNNTurb
: 
    public LESDirectTauij<BasicTurbulenceModel> 
{

protected:

    // Protected data

        // Model and constants

            torch::Device device_;
            torch::jit::script::Module model_; // created by pytorch
            const List<word> inputNames_; // names of input features
            List<scalar> inputMaxs_; // maximum values of input features
            const List<scalar> outputMaxs_; // maximum values of output features
            const double Ne_; // normalising factor of edge attr

        // Fields

            volScalarField adjustedDelta_; // filter width with non-zero value on the walls
            volScalarField expDWallDelta_; // exp(-d_wall / \Delta)
            volScalarField uDelta_; // u_\Delta
            volVectorField UFlucDelta_; // (u_i - <u_i>_\Delta)

        // Internal variables

            HashTable<int, word> gradUComponents_;
            HashTable<int, word> UFlucDeltaComponents_;

            List<labelList> strictNeighborCells_;

            using CellFaceStarGraphType = geometry::CellFaceStarGraph<List<vector>, List<scalar>>;
            using CellFaceStarGraphData = geometry::CellFaceStarGraphData<CellFaceStarGraphType>;
            CellFaceStarGraphData Data_;


    // Protected Member Functions

        //- Calculate the SGS strain tensor
        virtual void correctTauij();

        //- Calculate u_\Delta at a cell
        virtual void correctUDelta();

        //- Calculate (u_i - <u_i>_\Delta) at a cell
        virtual void correctUFlucDelta();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("GNNTurb");


    // Constructors

        //- Construct from components
        GNNTurb(
            const alphaField &alpha,
            const rhoField &rho,
            const volVectorField &U,
            const surfaceScalarField &alphaRhoPhi,
            const surfaceScalarField &phi,
            const transportModel &transport,
            const word &propertiesName = turbulenceModel::propertiesName,
            const word &type = typeName);

        //- Disallow default bitwise copy construction
        GNNTurb(const GNNTurb &) = delete;


    //- Destructor
    virtual ~GNNTurb() 
    {}


    // Member Functions

        //- Read model coefficients if they have changed
        virtual bool read();

        //- Correct tau_ij and related properties
        virtual void correct();

        //- Access function to filter width with non-zero value on the wall
        const volScalarField& adjustedDelta();

        //- Construct graph data
        virtual void buildGraph();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const GNNTurb &) = delete;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
  #include "GNNTurb.C"
  #include "UFlucDelta.C"
  #include "uDelta.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
