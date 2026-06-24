// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/LorenzGaugeConditions.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace GrSelfForce {

void lorenz_gauge_condition(
    const gsl::not_null<tnsr::a<ComplexDataVector, 3>*> result,
    const elliptic::analytic_data::Background& background,
    const tnsr::aa<ComplexDataVector, 3>& field, const DerivMMode& deriv_field,
    const tnsr::I<DataVector, 2, Frame::Inertial>& x,
    const bool zero_out
) {
  const auto* co_ptr =
      dynamic_cast<const AnalyticData::CircularOrbit*>(&background);
  const auto* nd_ptr =
      co_ptr ? nullptr
             : dynamic_cast<const AnalyticData::NumericData*>(&background);
  ASSERT(co_ptr != nullptr or nd_ptr != nullptr,
         "Background must be CircularOrbit or NumericData");
  const AnalyticData::CircularOrbit& circular_orbit =
      co_ptr ? *co_ptr : nd_ptr->circular_orbit();

  const double r0 = circular_orbit.orbital_radius();
  const double M = circular_orbit.black_hole_mass();
  const double spin = circular_orbit.black_hole_spin();
  const double a = M * spin;
  const int m_mode = circular_orbit.m_mode_number();
  const double omega = 1. / (a + sqrt(cube(r0) / M));  // \Omega
  (void)circular_orbit.version();

  set_number_of_grid_points(result, x);
  if (zero_out) {
    for (auto& component : *result) {
      component = std::complex<double>(0., 0.);
    }
    return;
  }

  const double r_plus = M * (1. + sqrt(1. - square(spin)));
  const auto& r_star_or_r = get<0>(x);
  const auto& theta_or_cos_theta = get<1>(x);
  const DataVector H = circular_orbit.hyperboloidal_boost_function(r_star_or_r);
  const bool penetrating_horizon = circular_orbit.penetrating_horizon();
  DataVector r;
  DataVector r_star;
  DataVector r_minus_r_plus;
  DataVector theta;
  DataVector cos_theta;
  DataVector sin_theta;
  DataVector cot_theta;
  DataVector csc_theta;
  if (penetrating_horizon) {
    // NOLINTNEXTLINE
    r.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = r - r_plus;
    r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
        r_minus_r_plus, M, spin);
    // NOLINTNEXTLINE
    cos_theta.set_data_ref(const_cast<DataVector*>(&theta_or_cos_theta));
    theta = acos(cos_theta);
    sin_theta = sin(theta);
    cot_theta = cos_theta / sin_theta;
    csc_theta = 1.0 / sin_theta;
  } else {
    // NOLINTNEXTLINE
    r_star.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus =
        gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M, spin);
    r = r_minus_r_plus + r_plus;
    // NOLINTNEXTLINE
    theta.set_data_ref(const_cast<DataVector*>(&theta_or_cos_theta));
    cos_theta = cos(theta);
    sin_theta = sin(theta);
    cot_theta = cos_theta / sin_theta;
    csc_theta = 1.0 / sin_theta;
  }
  // Use explicit scalar loops to avoid blaze SIMD issues with complex
  // expression template evaluation.
  const size_t num_pts = r.size();
  if (get<0, 0>(field).size() != num_pts or
      get<0, 0, 0>(deriv_field).size() != num_pts) {
    return;  // sizes inconsistent (observer mesh mismatch or uninit field)
  }

  const std::complex<double> im_m(0., m_mode);
  for (size_t i = 0; i < num_pts; ++i) {
    const std::complex<double> r_i = r[i];
    const std::complex<double> cos_i = cos_theta[i];
    const std::complex<double> sin_i = sin_theta[i];
    // const std::complex<double> cot_i = cot_theta[i];
    // const std::complex<double> csc_i = csc_theta[i];

    // version 2 * \sin \theta (to cancel out 0/sin_i at the pole)
    get<0>(*result)[i] =
        (cos_i * get<0, 2>(field)[i] + im_m * get<0, 3>(field)[i] +
         (1.0 - im_m * r_i * omega) * get<0, 1>(field)[i] * sin_i +
         get<0, 0>(field)[i] * sin_i -
         sin_i * get<1, 0, 2>(deriv_field)[i] * sin_i +
         (r_i - 2.0 * M) * get<0, 0, 1>(deriv_field)[i] * sin_i +
         (1.0 + H[i]) * im_m * omega * r_i * get<0, 1>(field)[i] * sin_i +
         r_i * get<0, 0, 0>(deriv_field)[i] * sin_i +
         (1.0 + H[i]) * im_m * omega * r_i * r_i / (r_i - 2.0 * M) *
             get<0, 0>(field)[i] * sin_i) /
        (r_i * r_i);

    // version 2 * \sin \theta (to cancel out 0/sin_i at the pole)
    get<1>(*result)[i] =
        (r_i * get<0, 1>(field)[i] * sin_i +
         (1. + r_i - im_m * r_i * r_i * omega) * get<1, 1>(field)[i] * sin_i +
         r_i * (cos_i * get<1, 2>(field)[i] + im_m * get<1, 3>(field)[i] -
                get<2, 2>(field)[i] * sin_i - get<3, 3>(field)[i] * sin_i -
                sin_i * get<1, 1, 2>(deriv_field)[i] * sin_i +
                r_i * get<0, 0, 1>(deriv_field)[i] * sin_i +
                (im_m * omega * (1. + H[i]) * r_i / (r_i - 2.)) * r_i *
                    get<0, 1>(field)[i] * sin_i +
                (-2. + r_i) * get<0, 1, 1>(deriv_field)[i] * sin_i +
                (im_m * omega * (1. + H[i]) * r_i / (r_i - 2.)) * (-2. + r_i) *
                    get<1, 1>(field)[i] * sin_i)) /
        (r_i * r_i * r_i);

    // version 2 *  \sin \theta (to cancel out 0/sin_i at the pole)
    get<2>(*result)[i] =
        (2. * r_i * get<0, 2>(field)[i] * sin_i +
         (-2. + r_i * (2. - im_m * r_i * omega)) * get<1, 2>(field)[i] * sin_i +
         r_i * (im_m * get<2, 3>(field)[i] +
                cos_i * (get<2, 2>(field)[i] - get<3, 3>(field)[i]) -
                sin_i * get<1, 2, 2>(deriv_field)[i] * sin_i +
                r_i * get<0, 0, 2>(deriv_field)[i] * sin_i +
                (im_m * omega * (1. + H[i]) * r_i / (r_i - 2.)) * r_i *
                    get<0, 2>(field)[i] * sin_i +
                (-2. + r_i) * get<0, 1, 2>(deriv_field)[i] * sin_i +
                (im_m * omega * (1. + H[i]) * r_i / (r_i - 2.)) * (-2. + r_i) *
                    get<1, 2>(field)[i] * sin_i

                )

             ) /
        (r_i * r_i);

    // this is working KEEP THIS
    get<3>(*result)[i] =
        (r_i * cos_i * get<2, 3>(field)[i] + im_m * r_i * get<3, 3>(field)[i] +
         2.0 * (r_i - M) * sin_i * get<1, 3>(field)[i] -
         im_m * r_i * r_i * omega * sin_i * get<1, 3>(field)[i] +
         2.0 * r_i * sin_i * get<0, 3>(field)[i] +
         r_i * cos_i * get<2, 3>(field)[i] -
         r_i * sin_i * sin_i * get<1, 2, 3>(deriv_field)[i] +
         (r_i * r_i - 2.0 * r_i) * sin_i * get<0, 1, 3>(deriv_field)[i] +
         (H[i] + 1.0) * im_m * omega * r_i * r_i * sin_i * get<1, 3>(field)[i] +
         r_i * r_i * sin_i * get<0, 0, 3>(deriv_field)[i] +
         (H[i] + 1.0) * im_m * omega * r_i * r_i * r_i * r_i /
             (r_i * r_i - 2.0 * M * r_i) * sin_i * get<0, 3>(field)[i]) /
        (r_i * r_i);
  }
}

}  // namespace GrSelfForce
