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
void GNNTurb<BasicTurbulenceModel>::correctUFlucDelta() {
    const volVectorField& U = this->U_;
    const surfaceVectorField U_surfaceField = fvc::interpolate(U);
    const volVectorField::Boundary& U_boundaryField = U.boundaryField();
    
    volVectorField& UFlucDelta = this->UFlucDelta_;
    
    forAll(this->mesh_.cells(), celli) {
        const CellFaceStarGraphType& g = this->Data_.G[celli];
        assert(g.reference_cid == celli);

        std::vector<scalar> ui_with_nbr(g.num_nodes);

        for (direction d=0; d<vector::nComponents; d++)
        {
            ui_with_nbr[0] = U[celli].component(d);

            for (int64_t j = 0; j < (int64_t)g.neighbor_faceIdInfos.size(); ++j) {
                const geometry::FaceIdInfo &fii = g.neighbor_faceIdInfos[j];

                if (fii.is_internal) {
                    assert(fii.faceid < U_surfaceField.size());
                    ui_with_nbr[j + 1] = U_surfaceField[fii.faceid].component(d);
                }
                else {
                    assert(fii.pFacei < U_boundaryField[fii.patchi].size());
                    ui_with_nbr[j + 1] = U_boundaryField[fii.patchi][fii.pFacei].component(d);
                }
            }

            scalar ui_bar = std::accumulate(ui_with_nbr.begin(), ui_with_nbr.end(), scalar(0.)) / g.num_nodes;

            UFlucDelta[celli].replace(d, U[celli].component(d) - ui_bar);
        }
    }

    UFlucDelta.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(UFlucDelta);
}

} // End namespace LESModels
} // End namespace Foam