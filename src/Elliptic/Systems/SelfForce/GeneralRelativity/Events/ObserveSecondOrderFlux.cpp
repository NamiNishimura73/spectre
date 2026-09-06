// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Events/ObserveSecondOrderFlux.hpp"

#include <cmath>
#include <complex>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

namespace GrSelfForce::Events::detail {

namespace {
const spectre::Exporter::PointwiseInterpolator<2, Frame::Inertial>&
get_first_order_interpolator(const std::string& first_order_data_directory) {
  static const spectre::Exporter::PointwiseInterpolator<2, Frame::Inertial>
      interpolator(first_order_data_directory + "/*Volume[0-9]*.h5",
                   "VolumeData", spectre::Exporter::ObservationStep(-1),
                   {"Re(MMode_yy)", "Im(MMode_yy)", "Re(MMode_zy)",
                    "Im(MMode_zy)", "Re(MMode_zz)", "Im(MMode_zz)"});
  return interpolator;
}
}  // namespace

void ensure_first_order_interpolator_loaded(
    const std::string& first_order_data_directory) {
  get_first_order_interpolator(first_order_data_directory);
}

std::pair<double, double> extract_second_order_flux(
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const Mesh<2>& mesh,
    const Scalar<DataVector>& face_jacobian,
    const tnsr::I<DataVector, 2, Frame::Inertial>& face_coords,
    const std::string& first_order_data_directory) {
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

  const auto& first_order_interpolator =
      get_first_order_interpolator(first_order_data_directory);
  std::vector<DataVector> interpolated;
  first_order_interpolator.interpolate_to_points(make_not_null(&interpolated),
                                                 face_coords);

  const ComplexDataVector psi7_1st =
      interpolated[0] + std::complex<double>(0., 1.) * interpolated[1];
  const ComplexDataVector psi8_1st =
      interpolated[2] + std::complex<double>(0., 1.) * interpolated[3];
  const ComplexDataVector psi9_1st =
      interpolated[4] + std::complex<double>(0., 1.) * interpolated[5];

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

  const ComplexDataVector psi7_2nd = get<2, 2>(field_on_face);
  const ComplexDataVector psi8_2nd = get<2, 3>(field_on_face);
  const ComplexDataVector psi9_2nd = get<3, 3>(field_on_face);

  const double energy_flux =
      square(m_mode * omega) * 0.125 *
      definite_integral(real(0.25 *
                                 (psi7_1st * pow(sin_theta, n_psi7 + 2) -
                                  psi9_1st * pow(sin_theta, n_psi9 - 2)) *
                                 conj(psi7_2nd * pow(sin_theta, n_psi7 + 2) -
                                      psi9_2nd * pow(sin_theta, n_psi9 - 2)) +
                             2.0 * (psi8_1st * pow(sin_theta, n_psi8)) *
                                 conj(psi8_2nd * pow(sin_theta, n_psi8)) +
                             0.25 *
                                 (psi9_1st * pow(sin_theta, n_psi9 - 2) -
                                  psi7_1st * pow(sin_theta, n_psi7 + 2)) *
                                 conj(psi9_2nd * pow(sin_theta, n_psi9 - 2) -
                                      psi7_2nd * pow(sin_theta, n_psi7 + 2))) *
                            get(face_jacobian) * integrand_multiplier,
                        mesh.slice_away(0));
  // Diagnostic: check for cancellation between the psi7 and psi9
  // contributions to the (psi7*s^(n7+2) - psi9*s^(n9-2)) combination that
  // enters the flux. If max|A| is much smaller than max|psi7 term| and
  // max|psi9 term|, the flux is disproportionately sensitive to ordinary
  // numerical noise in psi7/psi9 individually.
  {
    const ComplexDataVector psi7_1st_term =
        psi7_1st * pow(sin_theta, n_psi7 + 2);
    const ComplexDataVector psi9_1st_term =
        psi9_1st * pow(sin_theta, n_psi9 - 2);
    const ComplexDataVector a_1st = psi7_1st_term - psi9_1st_term;
    const ComplexDataVector psi7_2nd_term =
        psi7_2nd * pow(sin_theta, n_psi7 + 2);
    const ComplexDataVector psi9_2nd_term =
        psi9_2nd * pow(sin_theta, n_psi9 - 2);
    const ComplexDataVector a_2nd = psi7_2nd_term - psi9_2nd_term;
    Parallel::printf(
        "psi7/psi9 cancellation check (1st order): max|psi7*s^(n7+2)|=%e, "
        "max|psi9*s^(n9-2)|=%e, max|A|=%e\n",
        max(abs(psi7_1st_term)), max(abs(psi9_1st_term)), max(abs(a_1st)));
    Parallel::printf(
        "psi7/psi9 cancellation check (2nd order): max|psi7*s^(n7+2)|=%e, "
        "max|psi9*s^(n9-2)|=%e, max|A|=%e\n",
        max(abs(psi7_2nd_term)), max(abs(psi9_2nd_term)), max(abs(a_2nd)));
  }
  const double surface_area = definite_integral(
      integrand_multiplier * get(face_jacobian), mesh.slice_away(0));
  return {energy_flux, surface_area};
}

}  // namespace GrSelfForce::Events::detail
