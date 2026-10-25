// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Events/ObserveFluxHorizon.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Math.hpp"

namespace GrSelfForce::Events::detail {

ComplexDataVector psi0_on_horizon(const tnsr::aa<ComplexDataVector, 3>& field,
                                  const tnsr::aa<ComplexDataVector, 3>& d_field,
                                  const ComplexDataVector& d2_field_vv,
                                  const DataVector& z, const int m_mode,
                                  const double mode_frequency,
                                  const double black_hole_mass) {
  ASSERT(m_mode >= 1, "psi0_on_horizon requires m >= 1, got m = " << m_mode);
  const std::complex<double> imag_unit{0., 1.};
  const double m = static_cast<double>(m_mode);
  const double M = black_hole_mass;
  const std::complex<double> horizon_factor =
      1. + 4. * imag_unit * M * mode_frequency;

  const DataVector one_minus_z = 1. - z;
  const DataVector sin_squared = 1. - square(z);
  // psi^{7,8,9} carry two extra powers of sin(theta) for m = 1
  const DataVector psi_mm_weight =
      m_mode == 1 ? sin_squared : DataVector(z.size(), 1.);

  // h_ll terms
  ComplexDataVector result =
      m * (m - 1.) * square(one_minus_z) * get<0, 0>(field) +
      2. * m * one_minus_z * sin_squared * get<0, 0>(d_field) +
      square(sin_squared) * d2_field_vv;
  // h_lm and h_mm terms
  const ComplexDataVector psi_lm =
      get<0, 2>(field) - imag_unit * get<0, 3>(field);
  const ComplexDataVector d_psi_lm =
      get<0, 2>(d_field) - imag_unit * get<0, 3>(d_field);
  const ComplexDataVector psi_mm =
      imag_unit * (get<2, 2>(field) - get<3, 3>(field)) +
      2. * get<2, 3>(field);
  result += horizon_factor *
            ComplexDataVector{(m * one_minus_z + 2. * z) * psi_lm +
                              sin_squared * d_psi_lm +
                              M * mode_frequency * psi_mm_weight * psi_mm};
  const DataVector prefactor =
      pow(sin_squared, 0.5 * (m - 2.)) / (32. * cube(M));
  return ComplexDataVector{prefactor * result};
}

std::pair<double, double> extract_flux_horizon(
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const Mesh<2>& mesh,
    const Scalar<DataVector>& face_jacobian,
    const tnsr::I<DataVector, 2, Frame::Inertial>& face_coords) {
  const double r0 = circular_orbit.orbital_radius();
  const double M = circular_orbit.black_hole_mass();
  const double spin = circular_orbit.black_hole_spin();
  const int m_mode = circular_orbit.m_mode_number();
  if (spin != 0.) {
    ERROR("ObserveFluxHorizon is only implemented for Schwarzschild, but the "
          "black hole spin is "
          << spin);
  }
  if (circular_orbit.version() != 3 or
      not circular_orbit.penetrating_horizon() or
      not circular_orbit.compactification()) {
    ERROR("ObserveFluxHorizon requires Version 3, PenetratingHorizon and "
          "Compactification, but got Version "
          << circular_orbit.version() << ", PenetratingHorizon "
          << circular_orbit.penetrating_horizon() << ", Compactification "
          << circular_orbit.compactification());
  }
  const double r_plus = 2. * M;
  const DataVector& face_radius = get<0>(face_coords);
  if (max(abs(face_radius - r_plus)) > 1.e-10 * M) {
    ERROR("ObserveFluxHorizon must be evaluated at the horizon r_+ = "
          << r_plus << ", but the inner face is at r in [" << min(face_radius)
          << ", " << max(face_radius) << "]");
  }

  const Mesh<1> face_mesh = mesh.slice_away(0);
  const double surface_area =
      definite_integral(get(face_jacobian), face_mesh);
  // A stationary mode does not carry energy through the horizon
  if (m_mode == 0) {
    return {0., surface_area};
  }
  const double orbital_frequency = 1. / sqrt(cube(r0) / M);
  const double mode_frequency = m_mode * orbital_frequency;

  // z-derivatives on the face
  const auto field_on_face =
      dg::project_tensor_to_boundary(field, mesh, Direction<2>::lower_xi());
  const Index<1>& face_extents = face_mesh.extents();
  const std::array<Matrix, 1> logical_derivative{
      {Spectral::differentiation_matrix(face_mesh)}};
  const DataVector& z_on_face = get<1>(face_coords);
  const DataVector dz_dxi =
      apply_matrices(logical_derivative, z_on_face, face_extents);
  const auto d_dz = [&logical_derivative, &face_extents,
                     &dz_dxi](const ComplexDataVector& u) {
    return ComplexDataVector{
        apply_matrices(logical_derivative, u, face_extents) / dz_dxi};
  };
  tnsr::aa<ComplexDataVector, 3> d_field_on_face{
      face_mesh.number_of_grid_points()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i; j < 4; ++j) {
      d_field_on_face.get(i, j) = d_dz(field_on_face.get(i, j));
    }
  }
  const ComplexDataVector d2_field_vv_on_face =
      d_dz(get<0, 0>(d_field_on_face));

  // Evaluate psi0 on Gauss points, which exclude the poles z = +-1 where the
  // individual terms of psi0 diverge for m = 1
  const Mesh<1> gauss_mesh{face_mesh.extents(0), Spectral::Basis::Legendre,
                           Spectral::Quadrature::Gauss};
  const std::array<Matrix, 1> to_gauss{{Spectral::interpolation_matrix(
      face_mesh, Spectral::collocation_points(gauss_mesh))}};
  const auto interpolate = [&to_gauss, &face_extents](const auto& u) {
    return apply_matrices(to_gauss, u, face_extents);
  };
  tnsr::aa<ComplexDataVector, 3> field_on_gauss{
      gauss_mesh.number_of_grid_points()};
  tnsr::aa<ComplexDataVector, 3> d_field_on_gauss{
      gauss_mesh.number_of_grid_points()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i; j < 4; ++j) {
      field_on_gauss.get(i, j) = interpolate(field_on_face.get(i, j));
      d_field_on_gauss.get(i, j) = interpolate(d_field_on_face.get(i, j));
    }
  }
  const ComplexDataVector psi0 = psi0_on_horizon(
      field_on_gauss, d_field_on_gauss, interpolate(d2_field_vv_on_face),
      interpolate(z_on_face), m_mode, mode_frequency, M);

  // Barack & Sago (2007) Eq. (53), with psi0 in the Hartle-Hawking tetrad and
  // the phi integral done. The face measure dz = sin(theta) dtheta.
  const double energy_flux =
      32. * square(square(M)) / (1. + 16. * square(M * mode_frequency)) *
      definite_integral(
          real(psi0 * conj(psi0)) * interpolate(get(face_jacobian)),
          gauss_mesh);
  return {energy_flux, surface_area};
}

}  // namespace GrSelfForce::Events::detail
