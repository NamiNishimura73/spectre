// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Events/ObserveFlux.hpp"

#include <cmath>
#include <complex>
#include <tuple>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Math.hpp"

namespace GrSelfForce::Events::detail {

std::pair<double, double> extract_flux(
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const Mesh<2>& mesh,
    const Scalar<DataVector>& face_jacobian,
    const tnsr::I<DataVector, 2, Frame::Inertial>& face_coords) {
  const double r0 = circular_orbit.orbital_radius();
  const double M = circular_orbit.black_hole_mass();
  const double spin = circular_orbit.black_hole_spin();
  const double a = M * spin;
  const int m_mode = circular_orbit.m_mode_number();
  const double omega = 1. / (a + sqrt(cube(r0) / M));
  const int version_ = circular_orbit.version();
  const auto field_on_face =
      dg::project_tensor_to_boundary(field, mesh, Direction<2>::upper_xi());
  const bool penetrating_horizon = circular_orbit.penetrating_horizon();
  DataVector sin_theta;
  const size_t num_face_pts = get<0>(face_coords).size();
  DataVector integrand_multiplier{num_face_pts};
  if (penetrating_horizon) {
    sin_theta = sqrt(1. - square(get<1>(face_coords)));
    integrand_multiplier = 1.0;
  } else {
    sin_theta = sin(get<1>(face_coords));
    integrand_multiplier = sin_theta;
  }
  int n_psi7{};
  int n_psi8{};
  int n_psi9{};
  if (version_ == 3) {
    if (m_mode == 0) {
      n_psi7 = -2;
      n_psi8 = 0;
      n_psi9 = 2;
    } else if (m_mode == 1) {
      n_psi7 = -1;
      n_psi8 = 1;
      n_psi9 = 3;
    } else {
      n_psi7 = -2 + (m_mode - 2);
      n_psi8 = (m_mode - 2);
      n_psi9 = 2 + (m_mode - 2);
    }
  } else {
    n_psi7 = -2;
    n_psi8 = 0;
    n_psi9 = 2;
  }
  const double energy_flux =
      square(m_mode * omega) * 0.03125 *
      definite_integral(
          real(square(
                   abs(get<2, 2>(field_on_face) * pow(sin_theta, n_psi7 + 2))) +
               4. * square(abs(get<2, 3>(field_on_face) *
                               pow(sin_theta, n_psi8))) +
               square(
                   abs(get<3, 3>(field_on_face) * pow(sin_theta, n_psi9 - 2))) -
               get<2, 2>(field_on_face) * pow(sin_theta, n_psi7 + 2) *
                   conj(get<3, 3>(field_on_face) * pow(sin_theta, n_psi9 - 2)) -
               conj(get<2, 2>(field_on_face) * pow(sin_theta, n_psi7 + 2)) *
                   get<3, 3>(field_on_face) * pow(sin_theta, n_psi9 - 2)) *
              get(face_jacobian) * integrand_multiplier,
          mesh.slice_away(0));
  const double surface_area = definite_integral(
      integrand_multiplier * get(face_jacobian), mesh.slice_away(0));
  return {energy_flux, surface_area};
}

std::tuple<double, double, double> extract_flux(
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const DerivMMode& deriv_field,
    const Mesh<2>& mesh, const Scalar<DataVector>& face_jacobian,
    const tnsr::I<DataVector, 2, Frame::Inertial>& face_coords) {
  const double r0 = circular_orbit.orbital_radius();
  const double M = circular_orbit.black_hole_mass();
  const double spin = circular_orbit.black_hole_spin();
  const double a = M * spin;
  const int m_mode = circular_orbit.m_mode_number();
  const double omega = 1. / (a + sqrt(cube(r0) / M));
  const int version_ = circular_orbit.version();
  int n_psi7{};
  int n_psi8{};
  int n_psi9{};
  if (version_ == 3) {
    if (m_mode == 0) {
      n_psi7 = -2;
      n_psi8 = 0;
      n_psi9 = 2;
    } else if (m_mode == 1) {
      n_psi7 = -1;
      n_psi8 = 1;
      n_psi9 = 3;
    } else {
      n_psi7 = -2 + (m_mode - 2);
      n_psi8 = (m_mode - 2);
      n_psi9 = 2 + (m_mode - 2);
    }
  } else {
    n_psi7 = -2;
    n_psi8 = 0;
    n_psi9 = 2;
  }
  const auto field_on_face =
      dg::project_tensor_to_boundary(field, mesh, Direction<2>::upper_xi());
  const auto dfield_on_face = dg::project_tensor_to_boundary(
      deriv_field, mesh, Direction<2>::upper_xi());

  // r at the face
  const DataVector& r_face = get<0>(face_coords);

  // factor A for each component (dpsi/dr is the i=0 slice of deriv_field)
  const auto A_22 =
      get<2, 2>(field_on_face) + 2. * r_face * get<0, 2, 2>(dfield_on_face);
  const auto A_23 =
      get<2, 3>(field_on_face) + 2. * r_face * get<0, 2, 3>(dfield_on_face);
  const auto A_33 =
      get<3, 3>(field_on_face) + 2. * r_face * get<0, 3, 3>(dfield_on_face);

  const bool penetrating_horizon = circular_orbit.penetrating_horizon();
  DataVector sin_theta;
  const size_t num_face_pts = get<0>(face_coords).size();
  DataVector integrand_multiplier{num_face_pts};
  if (penetrating_horizon) {
    sin_theta = sqrt(1. - square(get<1>(face_coords)));
    integrand_multiplier = 1.0;
  } else {
    sin_theta = sin(get<1>(face_coords));
    integrand_multiplier = sin_theta;
  }
  const double energy_flux =
      square(m_mode * omega) * 0.03125 *
      definite_integral(
          real(square(
                   abs(get<2, 2>(field_on_face) * pow(sin_theta, n_psi7 + 2))) +
               4. * square(abs(get<2, 3>(field_on_face) *
                               pow(sin_theta, n_psi8))) +
               square(
                   abs(get<3, 3>(field_on_face) * pow(sin_theta, n_psi9 - 2))) -
               get<2, 2>(field_on_face) * pow(sin_theta, n_psi7 + 2) *
                   conj(get<3, 3>(field_on_face) * pow(sin_theta, n_psi9 - 2)) -
               conj(get<2, 2>(field_on_face) * pow(sin_theta, n_psi7 + 2)) *
                   get<3, 3>(field_on_face) * pow(sin_theta, n_psi9 - 2)) *
              get(face_jacobian) * integrand_multiplier,
          mesh.slice_away(0));
  const double energy_flux_fit =
      square(m_mode * omega) * 0.03125 *
      definite_integral(real(square(abs(A_22) * pow(sin_theta, n_psi7 + 2)) +
                             4. * square(abs(A_23) * pow(sin_theta, n_psi8)) +
                             square(abs(A_33 * pow(sin_theta, n_psi9 - 2))) -
                             A_22 * pow(sin_theta, n_psi7 + 2) *
                                 conj(A_33 * pow(sin_theta, n_psi9 - 2)) -
                             conj(A_22 * pow(sin_theta, n_psi7 + 2)) * A_33 *
                                 pow(sin_theta, n_psi9 - 2)) *
                            get(face_jacobian) * integrand_multiplier,
                        mesh.slice_away(0));
  const double surface_area = definite_integral(
      integrand_multiplier * get(face_jacobian), mesh.slice_away(0));
  return {energy_flux, energy_flux_fit, surface_area};
}

}  // namespace GrSelfForce::Events::detail
