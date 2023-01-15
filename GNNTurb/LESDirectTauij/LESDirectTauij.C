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

#include "LESDirectTauij.H"
#include "fvc.H"
#include "fvm.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
LESDirectTauij<BasicTurbulenceModel>::LESDirectTauij
(
    const word& modelName,
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    LESModel<BasicTurbulenceModel>
    (
        modelName,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    dummy_nut_
    (
        IOobject
        (
            IOobject::groupName("nut", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
            "nut", 
            dimensionSet(0,2,-1,0,0,0,0), 
            0.0
        )
    ),

    tauij_( // \tilde{u_i u_j} - \tilde{u_i} \tilde{u_j}
        IOobject
        (
            IOobject::groupName("tauij", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedSymmTensor
        (
            "tauij", 
            dimVelocity*dimVelocity, 
            symmTensor::zero
        )
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool LESDirectTauij<BasicTurbulenceModel>::read()
{
    return BasicTurbulenceModel::read();
}


template<class BasicTurbulenceModel>
tmp<volSymmTensorField>
LESDirectTauij<BasicTurbulenceModel>::devRhoReff() const
{
    return volSymmTensorField::New
    (
        IOobject::groupName("devRhoReff", this->alphaRhoPhi_.group()),
        (-(this->alpha_*this->rho_*this->nu()))
       *dev(twoSymm(fvc::grad(this->U_)))
        + (this->alpha_*this->rho_) * dev(this->tauij())
    );
}


template<class BasicTurbulenceModel>
tmp<fvVectorMatrix>
LESDirectTauij<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U
) const
{
    return 
    (
    - fvc::div((this->alpha_*this->rho_*this->nu()) * dev2(T(fvc::grad(U))))
    - fvm::laplacian(this->alpha_*this->rho_*this->nu(), U)
    + fvc::div(this->alpha_*this->rho_ * dev(this->tauij()))
    );
}


template<class BasicTurbulenceModel>
tmp<fvVectorMatrix>
LESDirectTauij<BasicTurbulenceModel>::divDevRhoReff
(
    const volScalarField& rho,
    volVectorField& U
) const
{
    return 
    (
    - fvc::div((this->alpha_*rho*this->nu()) * dev2(T(fvc::grad(U))))
    - fvm::laplacian(this->alpha_*rho*this->nu(), U)
    + fvc::div(this->alpha_*rho * dev(this->tauij()))
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> LESDirectTauij<BasicTurbulenceModel>::k() const
{
    return volScalarField::New
    (
        IOobject::groupName("k", this->alphaRhoPhi_.group()),
        tr(this->tauij_) / 2.0
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> LESDirectTauij<BasicTurbulenceModel>::epsilon() const
{
    volScalarField k(this->k());
    scalar Ce = 1.048; // default value of smagorinsky

    return volScalarField::New
    (
        IOobject::groupName("epsilon", this->alphaRhoPhi_.group()),
        Ce*k*sqrt(k)/this->delta()
    );
}


template<class BasicTurbulenceModel>
tmp<volSymmTensorField>
LESDirectTauij<BasicTurbulenceModel>::R() const 
{
    tmp<volScalarField> tk(k());

    // Get list of patchField type names from k
    wordList patchFieldTypes(tk().boundaryField().types());

    // For k patchField types which do not have an equivalent for symmTensor
    // set to calculated
    forAll(patchFieldTypes, i)
    {
        if
        (
           !fvPatchField<symmTensor>::patchConstructorTablePtr_
                ->found(patchFieldTypes[i])
        )
        {
            patchFieldTypes[i] = calculatedFvPatchField<symmTensor>::typeName;
        }
    }

    return volSymmTensorField::New
    (
        IOobject::groupName("R", this->alphaRhoPhi_.group()),
        this->tauij_,
        patchFieldTypes
    );
}


template<class BasicTurbulenceModel>
void LESDirectTauij<BasicTurbulenceModel>::validate()
{
    correctTauij();
}


template<class BasicTurbulenceModel>
void LESDirectTauij<BasicTurbulenceModel>::correct()
{
    BasicTurbulenceModel::correct();
}


template <class BasicTurbulenceModel>
tmp<volSymmTensorField> 
LESDirectTauij<BasicTurbulenceModel>::tauij() const 
{
    return volSymmTensorField::New
    (
        IOobject::groupName("tauij", this->alphaRhoPhi_.group()),
        this->tauij_
    );
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //