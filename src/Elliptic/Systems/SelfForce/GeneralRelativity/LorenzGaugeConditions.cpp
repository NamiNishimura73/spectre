// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/LorenzGaugeConditions.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace GrSelfForce {

namespace {
// Coefficients of the Lorenz gauge condition g^{al mu} nabla_al hbar_{mu nu}
// for version 3 (vrz) variables, Kerr (M = 1), m >= 2, vtu slicing:
//   LG_nu = sum_k c[nu][k] psi_k + dr[nu][k] d_r psi_k + dz[nu][k] d_z psi_k
// with psi_k in tnsr::aa storage order {vv,vr,vz,vphi,rr,rz,rphi,zz,zphi,phiphi}
// and om = m * Omega. nu = (v, r, z, phi); the z component is multiplied by
// (1 - z^2) to remove its simple pole at z = +-1.
// Generated with Mathematica: LorenzGaugeCond.m (Kerr section) ->
// LG_coeffs_vrz.txt. Do not edit by hand.
// NOLINTBEGIN
void lorenz_gauge_coefficients_vrz(
    const double m, const double a, const double om, const double H,
    const double r, const double z,
    std::array<std::array<std::complex<double>, 10>, 4>& c,
    std::array<std::array<std::complex<double>, 10>, 4>& dr,
    std::array<std::array<std::complex<double>, 10>, 4>& dz) {
  for (size_t nu = 0; nu < 4; ++nu) {
    c[nu].fill(0.);
    dr[nu].fill(0.);
    dz[nu].fill(0.);
  }
  c[0][0] = (std::complex<double>(0.,-1.)*(-1. + pow(z,2))*(pow(a,3)*m*r + a*m*(-2. + r)*pow(r,2) + pow(r,3)*(std::complex<double>(0.,2.) - std::complex<double>(0.,1.)*r + (1. + H)*om*pow(r,2)) + pow(a,4)*(std::complex<double>(0.,1.) + H*om*r + om*r*pow(z,2)) + pow(a,2)*r*(std::complex<double>(0.,-2.) + om*r*(2. + r + 2.*H*r - 2.*pow(z,2) + r*pow(z,2)))))/(pow(r,2)*(pow(a,2) + (-2. + r)*r)*(pow(r,2) + pow(a,2)*pow(z,2)));
  c[0][1] = (std::complex<double>(0.,-1.)*(a*m*r + pow(r,2)*(std::complex<double>(0.,-1.) + H*om*r) + pow(a,2)*(std::complex<double>(0.,1.) + H*om*r))*(-1. + pow(z,2)))/(pow(r,4) + pow(a,2)*pow(r,2)*pow(z,2));
  c[0][2] = (-2.*z)/(pow(r,2) + pow(a,2)*pow(z,2));
  c[0][3] = (std::complex<double>(0.,-1.)*(-1.*pow(a,2)*m - 1.*m*(-2. + r)*r + pow(a,3)*H*om*(-1. + pow(z,2)) + a*om*r*(2. + H*r)*(-1. + pow(z,2))))/((pow(a,2) + (-2. + r)*r)*(pow(r,2) + pow(a,2)*pow(z,2)));
  dr[0][0] = (-1.*(pow(a,2) + pow(r,2))*(-1. + pow(z,2)))/(pow(r,3) + pow(a,2)*r*pow(z,2));
  dr[0][1] = (-1.*(pow(a,2) + (-2. + r)*r)*(-1. + pow(z,2)))/(pow(r,3) + pow(a,2)*r*pow(z,2));
  dr[0][3] = (a - 1.*a*pow(z,2))/(pow(r,2) + pow(a,2)*pow(z,2));
  dz[0][2] = (1. - 1.*pow(z,2))/(pow(r,2) + pow(a,2)*pow(z,2));
  c[1][0] = (-1.*pow(a,2)*pow(-1. + pow(z,2),2))/pow(pow(r,2) + pow(a,2)*pow(z,2),2);
  c[1][1] = (std::complex<double>(0.,-1.)*(-1. + pow(z,2))*(a*m*(-2. + r)*pow(r,4) + pow(r,5)*(std::complex<double>(0.,2.) - std::complex<double>(0.,1.)*r + (1. + H)*om*pow(r,2)) + pow(a,5)*m*r*pow(z,2) + pow(a,3)*m*pow(r,2)*(r - 2.*pow(z,2) + r*pow(z,2)) + pow(a,6)*pow(z,2)*(std::complex<double>(0.,1.) + H*om*r + om*r*pow(z,2)) + pow(a,4)*r*(std::complex<double>(0.,-2.)*pow(z,2) + om*pow(r,2)*(H + 2.*pow(z,2) + 2.*H*pow(z,2) + pow(z,4)) + r*(std::complex<double>(0.,3.) + 2.*(std::complex<double>(0.,-1.) + om)*pow(z,2) - 2.*om*pow(z,4))) + pow(a,2)*pow(r,3)*(std::complex<double>(0.,6.)*(-1. + pow(z,2)) + r*(std::complex<double>(0.,2.) - std::complex<double>(0.,3.)*pow(z,2) - 2.*om*(-1. + pow(z,2))) + om*pow(r,2)*(1. + 2.*pow(z,2) + H*(2. + pow(z,2))))))/((pow(a,2) + (-2. + r)*r)*pow(pow(r,3) + pow(a,2)*r*pow(z,2),2));
  c[1][3] = (2.*a*r*(-1. + pow(z,2)))/pow(pow(r,2) + pow(a,2)*pow(z,2),2);
  c[1][4] = (std::complex<double>(0.,-1.)*(-1. + pow(z,2))*(a*m*pow(r,3) + pow(r,3)*(std::complex<double>(0.,-1.) - std::complex<double>(0.,1.)*r + H*om*pow(r,2)) + pow(a,3)*m*r*pow(z,2) + pow(a,4)*(std::complex<double>(0.,1.) + H*om*r)*pow(z,2) + pow(a,2)*r*(std::complex<double>(0.,1.)*pow(z,2) - std::complex<double>(0.,2.)*r*(-1. + pow(z,2)) + H*om*pow(r,2)*(1. + pow(z,2)))))/pow(pow(r,3) + pow(a,2)*r*pow(z,2),2);
  c[1][5] = (-2.*z)/(pow(r,2) + pow(a,2)*pow(z,2));
  c[1][6] = (std::complex<double>(0.,-1.)*(-1.*m*(-2. + r)*pow(r,3) - 1.*pow(a,4)*m*pow(z,2) + a*pow(r,2)*(std::complex<double>(0.,-4.) + 2.*(std::complex<double>(0.,1.) + om)*r + H*om*pow(r,2))*(-1. + pow(z,2)) + pow(a,5)*H*om*pow(z,2)*(-1. + pow(z,2)) - 1.*pow(a,2)*m*r*(r - 2.*pow(z,2) + r*pow(z,2)) + pow(a,3)*r*(-1. + pow(z,2))*(std::complex<double>(0.,2.) + 2.*om*pow(z,2) + H*om*r*(1. + pow(z,2)))))/((pow(a,2) + (-2. + r)*r)*pow(pow(r,2) + pow(a,2)*pow(z,2),2));
  c[1][7] = (-1.*pow(r,2))/pow(pow(r,2) + pow(a,2)*pow(z,2),2);
  c[1][9] = (-1.*pow(r,2))/pow(pow(r,2) + pow(a,2)*pow(z,2),2);
  dr[1][1] = (-1.*(pow(a,2) + pow(r,2))*(-1. + pow(z,2)))/(pow(r,3) + pow(a,2)*r*pow(z,2));
  dr[1][4] = (-1.*(pow(a,2) + (-2. + r)*r)*(-1. + pow(z,2)))/(pow(r,3) + pow(a,2)*r*pow(z,2));
  dr[1][6] = (a - 1.*a*pow(z,2))/(pow(r,2) + pow(a,2)*pow(z,2));
  dz[1][5] = (1. - 1.*pow(z,2))/(pow(r,2) + pow(a,2)*pow(z,2));
  c[2][0] = (-1.*pow(a,2)*(pow(a,2) + pow(r,2))*z*pow(-1. + pow(z,2),2))/(r*pow(pow(r,2) + pow(a,2)*pow(z,2),2));
  c[2][1] = (-2.*pow(a,2)*(pow(a,2) + pow(r,2))*z*pow(-1. + pow(z,2),2))/(r*pow(pow(r,2) + pow(a,2)*pow(z,2),2));
  c[2][2] = (std::complex<double>(0.,-1.)*(-1. + pow(z,2))*(pow(a,3)*m + a*m*(-2. + r)*r + pow(r,2)*(std::complex<double>(0.,4.) - std::complex<double>(0.,2.)*r + (1. + H)*om*pow(r,2)) + pow(a,4)*om*(H + pow(z,2)) + pow(a,2)*r*(std::complex<double>(0.,-2.) + om*(2. + r + 2.*H*r - 2.*pow(z,2) + r*pow(z,2)))))/((pow(a,2) + (-2. + r)*r)*(pow(r,2) + pow(a,2)*pow(z,2)));
  c[2][3] = (-2.*pow(a,3)*z*pow(-1. + pow(z,2),2))/pow(pow(r,2) + pow(a,2)*pow(z,2),2);
  c[2][4] = (-1.*pow(a,2)*(pow(a,2) + (-2. + r)*r)*z*pow(-1. + pow(z,2),2))/(r*pow(pow(r,2) + pow(a,2)*pow(z,2),2));
  c[2][5] = (std::complex<double>(0.,-1.)*(std::complex<double>(0.,2.) + a*m + pow(a,2)*H*om - std::complex<double>(0.,2.)*r + H*om*pow(r,2))*(-1. + pow(z,2)))/(pow(r,2) + pow(a,2)*pow(z,2));
  c[2][6] = (-2.*pow(a,3)*z*pow(-1. + pow(z,2),2))/pow(pow(r,2) + pow(a,2)*pow(z,2),2);
  c[2][7] = (-1.*r*(pow(a,2) + pow(r,2))*z)/pow(pow(r,2) + pow(a,2)*pow(z,2),2);
  c[2][8] = (std::complex<double>(0.,1.)*pow(a,2)*m*r + std::complex<double>(0.,1.)*m*(-2. + r)*pow(r,2) - std::complex<double>(0.,1.)*pow(a,3)*(std::complex<double>(0.,-1.) + H*om*r)*(-1. + pow(z,2)) + a*r*(2. + (-1. - std::complex<double>(0.,2.)*om)*r - std::complex<double>(0.,1.)*H*om*pow(r,2))*(-1. + pow(z,2)))/((pow(a,2) + (-2. + r)*r)*(pow(r,2) + pow(a,2)*pow(z,2)));
  c[2][9] = (r*z*(pow(r,2) + pow(a,2)*(-1. + 2.*pow(z,2))))/pow(pow(r,2) + pow(a,2)*pow(z,2),2);
  dr[2][2] = (-1.*(pow(a,2) + pow(r,2))*(-1. + pow(z,2)))/(pow(r,2) + pow(a,2)*pow(z,2));
  dr[2][5] = ((pow(a,2) + (-2. + r)*r)*(1. - 1.*pow(z,2)))/(pow(r,2) + pow(a,2)*pow(z,2));
  dr[2][8] = (-1.*a*r*(-1. + pow(z,2)))/(pow(r,2) + pow(a,2)*pow(z,2));
  dz[2][7] = (r - 1.*r*pow(z,2))/(pow(r,2) + pow(a,2)*pow(z,2));
  c[3][3] = (std::complex<double>(0.,-1.)*(-1. + pow(z,2))*(pow(a,3)*m + a*m*(-2. + r)*r + pow(r,2)*(std::complex<double>(0.,4.) - std::complex<double>(0.,2.)*r + (1. + H)*om*pow(r,2)) + pow(a,4)*om*(H + pow(z,2)) + pow(a,2)*r*(std::complex<double>(0.,-2.) + om*(2. + r + 2.*H*r - 2.*pow(z,2) + r*pow(z,2)))))/((pow(a,2) + (-2. + r)*r)*(pow(r,2) + pow(a,2)*pow(z,2)));
  c[3][6] = (std::complex<double>(0.,-1.)*(std::complex<double>(0.,2.) + a*m + pow(a,2)*H*om - std::complex<double>(0.,2.)*r + H*om*pow(r,2))*(-1. + pow(z,2)))/(pow(r,2) + pow(a,2)*pow(z,2));
  c[3][8] = (-2.*r*z)/(pow(r,2) + pow(a,2)*pow(z,2));
  c[3][9] = (std::complex<double>(0.,1.)*pow(a,2)*m*r + std::complex<double>(0.,1.)*m*(-2. + r)*pow(r,2) - std::complex<double>(0.,1.)*pow(a,3)*(std::complex<double>(0.,-1.) + H*om*r)*(-1. + pow(z,2)) + a*r*(2. + (-1. - std::complex<double>(0.,2.)*om)*r - std::complex<double>(0.,1.)*H*om*pow(r,2))*(-1. + pow(z,2)))/((pow(a,2) + (-2. + r)*r)*(pow(r,2) + pow(a,2)*pow(z,2)));
  dr[3][3] = (-1.*(pow(a,2) + pow(r,2))*(-1. + pow(z,2)))/(pow(r,2) + pow(a,2)*pow(z,2));
  dr[3][6] = (-1.*(pow(a,2) + (-2. + r)*r)*(-1. + pow(z,2)))/(pow(r,2) + pow(a,2)*pow(z,2));
  dr[3][9] = (-1.*a*r*(-1. + pow(z,2)))/(pow(r,2) + pow(a,2)*pow(z,2));
  dz[3][8] = (r - 1.*r*pow(z,2))/(pow(r,2) + pow(a,2)*pow(z,2));
}
// NOLINTEND

// (a, b) index pair of psi_k, k = 0..9
constexpr std::array<std::array<size_t, 2>, 10> psi_index_pairs{
    {{{0, 0}}, {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 1}}, {{1, 2}}, {{1, 3}},
     {{2, 2}}, {{2, 3}}, {{3, 3}}}};
}  // namespace

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

  set_number_of_grid_points(result, x);
  if (zero_out) {
    for (auto& component : *result) {
      component = std::complex<double>(0., 0.);
    }
    return;
  }

  if (circular_orbit.version() == 3) {
    // Kerr, version 3 (vrz) variables. Coefficients are generated for m >= 2
    // prefactors and M = 1; m = 0, 1 use different prefactors.
    if (m_mode < 2) {
      ERROR("Lorenz gauge condition for version 3 is only implemented for "
            "m >= 2, but m = "
            << m_mode);
    }
    ASSERT(circular_orbit.penetrating_horizon(),
           "Version 3 Lorenz gauge condition requires penetrating_horizon");
    ASSERT(M == 1., "Version 3 Lorenz gauge coefficients assume M = 1");
    const size_t num_pts = get<0>(x).size();
    if (get<0, 0>(field).size() != num_pts or
        get<0, 0, 0>(deriv_field).size() != num_pts) {
      return;  // sizes inconsistent (observer mesh mismatch or uninit field)
    }
    // x = (r, z), except in the compactified u region where
    // x^0 = sigma = 2 r_u - r_u^2 / r and d/dr = (r_u^2 / r^2) d/dsigma.
    const auto& sigma_or_r = get<0>(x);
    const auto& z = get<1>(x);
    const DataVector H = circular_orbit.hyperboloidal_boost_function(sigma_or_r);
    const bool compactified = circular_orbit.compactification();
    const double r_u =
        compactified ? (*circular_orbit.hyperboloidal_slicing_transitions())[3]
                     : 0.;
    std::array<std::array<std::complex<double>, 10>, 4> c{};
    std::array<std::array<std::complex<double>, 10>, 4> dr{};
    std::array<std::array<std::complex<double>, 10>, 4> dz{};
    for (size_t i = 0; i < num_pts; ++i) {
      double r = sigma_or_r[i];
      double dsigma_dr = 1.;
      if (compactified and sigma_or_r[i] > r_u) {
        r = square(r_u) / (2. * r_u - sigma_or_r[i]);
        dsigma_dr = square(r_u / r);
      }
      lorenz_gauge_coefficients_vrz(static_cast<double>(m_mode), a,
                                    m_mode * omega, H[i], r, z[i], c, dr, dz);
      for (size_t nu = 0; nu < 4; ++nu) {
        std::complex<double> lg = 0.;
        for (size_t k = 0; k < 10; ++k) {
          const size_t i0 = gsl::at(psi_index_pairs, k)[0];
          const size_t i1 = gsl::at(psi_index_pairs, k)[1];
          lg += gsl::at(gsl::at(c, nu), k) * field.get(i0, i1)[i] +
                gsl::at(gsl::at(dr, nu), k) * dsigma_dr *
                    deriv_field.get(0, i0, i1)[i] +
                gsl::at(gsl::at(dz, nu), k) * deriv_field.get(1, i0, i1)[i];
        }
        result->get(nu)[i] = lg;
      }
    }
    return;
  }

  // Version 2 variables, Schwarzschild (a = 0) only.

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
