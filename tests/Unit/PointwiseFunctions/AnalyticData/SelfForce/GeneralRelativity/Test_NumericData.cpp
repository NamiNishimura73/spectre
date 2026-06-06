// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/CircularOrbit.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/NumericData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GrSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrSelfForce.NumericData",
                  "[PointwiseFunctions][Unit]") {
  // Test that NumericData (h5-based) agrees with CircularOrbit (analytic)
  // for a 1st-order h5 dataset. Three checks:
  //   1. Seff inside the worldtube (2D mesh, field_is_regularized=true)
  //   2. hS and its normal derivative at the Left face (r = r_wt_left)
  //   3. hS and its normal derivative at the Bottom face (theta = theta_wt_bot)

  const std::string h5_file =
      "/u/namni/spectre_copy/data/"
      "NamiD2G_m2_a0.600_r8.000comoving_moregrid5000_ascii.h5";
  const double bh_mass = 1.;
  const double bh_spin = 0.6;
  const double orbital_radius = 8.;
  const int m_mode = 2;
  // Transitions from RetRetV/T/U grid bounds in h5
  const std::array<double, 4> transitions{3.8667, 3.8667, 14.2, 14.2};

  // Worldtube face coordinates (from Seff.dat attributes)
  const double r_wt_left = 5.933333333333333;
  const double r_wt_right = 10.066666666666666;
  // thetaMin = pi/3 -> cos = 0.5;  thetaMax = 2*pi/3 -> cos = -0.5
  const double cos_wt_bot = 0.5;
  const double cos_wt_top = -0.5;

  const NumericData numeric_data{h5_file, bh_mass,     bh_spin, orbital_radius,
                                 m_mode,  transitions, true,    false};
  const CircularOrbit circular_orbit{bh_mass, bh_spin,     orbital_radius,
                                     m_mode,  transitions, true};

  const Approx approx = Approx::custom().epsilon(1.e-6).scale(1.);

  // -----------------------------------------------------------------------
  // Test 1: Seff on a 2D interior mesh (field_is_regularized=true uses the
  // high-resolution 500x500 Seff.dat grid)
  // -----------------------------------------------------------------------
  {
    const size_t npoints = 10;
    // Domain well inside worldtube bounds
    const domain::creators::Rectangle domain_creator{{{6.5, -0.3}},
                                                     {{9.5, 0.3}},
                                                     {{0, 0}},
                                                     {{npoints, npoints}},
                                                     {{false, false}}};
    const auto domain = domain_creator.create_domain();
    const ElementMap<2, Frame::Inertial> element_map{ElementId<2>{0},
                                                     domain.blocks()[0]};
    const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss};
    const auto x = element_map(logical_coordinates(mesh));

    const auto nd_vars =
        numeric_data.variables(x, NumericData::source_tags{}, true);
    const auto co_vars =
        circular_orbit.variables(x, CircularOrbit::source_tags{}, true);
    const auto& nd_seff = get<::Tags::FixedSource<Tags::MMode>>(nd_vars);
    const auto& co_seff = get<::Tags::FixedSource<Tags::MMode>>(co_vars);
    // double max_diff = 0.;
    // size_t max_i = 0, max_j = 0;
    // for (size_t i = 0; i < nd_seff.size(); ++i)
    //   for (size_t j = 0; j < nd_seff[i].size(); ++j) {
    //     const double diff = std::abs(nd_seff[i][j] - co_seff[i][j]);
    //     if (diff > max_diff) {
    //       max_diff = diff;
    //       max_i = i;
    //       max_j = j;
    //     }
    //   }
    // printf(
    //     "Test 1 Seff:   max|diff|=%.3e  at r=%.4f cos_theta=%.4f (component "
    //     "%zu)\n",
    //     max_diff, get<0>(x)[max_j], get<1>(x)[max_j], max_i);
    // printf("# Test1_Seff r  cos_theta  abs_diff\n");
    // for (size_t j = 0; j < nd_seff[0].size(); ++j) {
    //   double diff_j = 0.;
    //   for (size_t i = 0; i < nd_seff.size(); ++i)
    //     diff_j = std::max(diff_j, std::abs(nd_seff[i][j] - co_seff[i][j]));
    //   printf("%.6e  %.6e  %.6e\n", get<0>(x)[j], get<1>(x)[j], diff_j);
    // }
    // // for (size_t i = 0; i < nd_seff.size(); ++i) {
    // //   CHECK_ITERABLE_CUSTOM_APPROX(nd_seff[i], co_seff[i], approx);
    // // }
    // CHECK(max_diff < 1.e-3);
    for (size_t i = 0; i < nd_seff.size(); ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(nd_seff[i], co_seff[i], approx);
    }
  }

  // -----------------------------------------------------------------------
  // Test 2: hS and dhS/dr at Left face (r = r_wt_left, cos_theta varies)
  // NumericData fills singular_field from Left.dat (1D, theta-parameterized)
  // and stores dhS/dr in deriv_singular_field.get(0,...); theta-deriv = 0.
  // -----------------------------------------------------------------------
  {
    const size_t nface = 10;
    tnsr::I<DataVector, 2> x_left{};
    get<0>(x_left) = DataVector(nface, r_wt_left);
    get<1>(x_left) = DataVector(nface, 0.);
    for (size_t i = 0; i < nface; ++i) {
      // cos_theta strictly inside worldtube (avoid corners)
      get<1>(x_left)[i] =
          cos_wt_top + (cos_wt_bot - cos_wt_top) * (i + 1.) / (nface + 1.);
    }

    const auto nd_vars =
        numeric_data.variables(x_left, NumericData::source_tags{}, true);
    const auto co_vars =
        circular_orbit.variables(x_left, CircularOrbit::source_tags{}, true);
    const auto& nd_hS = get<Tags::SingularField>(nd_vars);
    const auto& co_hS = get<Tags::SingularField>(co_vars);
    const auto& nd_dhS = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        nd_vars);
    const auto& co_dhS = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        co_vars);
    // double max_diff_hS = 0., max_diff_dhS = 0.;
    // size_t max_j_hS = 0, max_j_dhS = 0;
    // for (size_t i = 0; i < nd_hS.size(); ++i)
    //   for (size_t j = 0; j < nd_hS[i].size(); ++j) {
    //     const double diff = std::abs(nd_hS[i][j] - co_hS[i][j]);
    //     if (diff > max_diff_hS) {
    //       max_diff_hS = diff;
    //       max_j_hS = j;
    //     }
    //   }
    // for (size_t a1 = 0; a1 < 4; ++a1)
    //   for (size_t b = 0; b <= a1; ++b)
    //     for (size_t j = 0; j < nd_dhS.get(0, a1, b).size(); ++j) {
    //       const double diff =
    //           std::abs(nd_dhS.get(0, a1, b)[j] - co_dhS.get(0, a1, b)[j]);
    //       if (diff > max_diff_dhS) {
    //         max_diff_dhS = diff;
    //         max_j_dhS = j;
    //       }
    //     }
    // printf(
    //     "Test 2 Left:   max|hS diff|=%.3e at cos_theta=%.4f"
    //     "  max|dhS/dr diff|=%.3e at cos_theta=%.4f\n",
    //     max_diff_hS, get<1>(x_left)[max_j_hS], max_diff_dhS,
    //     get<1>(x_left)[max_j_dhS]);
    // printf("# Test2_Left cos_theta  abs_hS_diff  abs_dhS_dr_diff\n");
    // for (size_t j = 0; j < nd_hS[0].size(); ++j) {
    //   double diff_hS_j = 0., diff_dhS_j = 0.;
    //   for (size_t i = 0; i < nd_hS.size(); ++i)
    //     diff_hS_j = std::max(diff_hS_j, std::abs(nd_hS[i][j] - co_hS[i][j]));
    //   for (size_t a1 = 0; a1 < 4; ++a1)
    //     for (size_t b = 0; b <= a1; ++b)
    //       diff_dhS_j = std::max(diff_dhS_j, std::abs(nd_dhS.get(0, a1, b)[j]
    //       -
    //                                                  co_dhS.get(0, a1,
    //                                                  b)[j]));
    //   printf("%.6e  %.6e  %.6e\n", get<1>(x_left)[j], diff_hS_j, diff_dhS_j);
    // }
    // // for (size_t i = 0; i < nd_hS.size(); ++i) {
    // //   CHECK_ITERABLE_CUSTOM_APPROX(nd_hS[i], co_hS[i], approx);
    // // }
    // CHECK(max_diff_hS < 1.e-3);
    // CHECK(max_diff_dhS < 1.e-3);
    for (size_t i = 0; i < nd_hS.size(); ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(nd_hS[i], co_hS[i], approx);
    }
    // Only the r-derivative (index 0) is populated at Left face
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        CHECK_ITERABLE_CUSTOM_APPROX(nd_dhS.get(0, a1, b), co_dhS.get(0, a1, b),
                                     approx);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Test 3: hS and dhS/dtheta at Bottom face (cos_theta = cos_wt_bot, r varies)
  // NumericData fills from Bottom.dat (1D, r-parameterized)
  // and stores dhS/dtheta in deriv_singular_field.get(1,...); r-deriv = 0.
  // -----------------------------------------------------------------------
  {
    const size_t nface = 10;
    tnsr::I<DataVector, 2> x_bot{};
    get<0>(x_bot) = DataVector(nface, 0.);
    get<1>(x_bot) = DataVector(nface, cos_wt_bot);
    for (size_t i = 0; i < nface; ++i) {
      // r strictly inside worldtube (avoid corners)
      get<0>(x_bot)[i] =
          r_wt_left + (r_wt_right - r_wt_left) * (i + 1.) / (nface + 1.);
    }

    const auto nd_vars =
        numeric_data.variables(x_bot, NumericData::source_tags{}, true);
    const auto co_vars =
        circular_orbit.variables(x_bot, CircularOrbit::source_tags{}, true);
    const auto& nd_hS = get<Tags::SingularField>(nd_vars);
    const auto& co_hS = get<Tags::SingularField>(co_vars);
    const auto& nd_dhS = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        nd_vars);
    const auto& co_dhS = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        co_vars);
    // double max_diff_hS = 0., max_diff_dhS = 0.;
    // size_t max_j_hS = 0, max_j_dhS = 0;
    // for (size_t i = 0; i < nd_hS.size(); ++i)
    //   for (size_t j = 0; j < nd_hS[i].size(); ++j) {
    //     const double diff = std::abs(nd_hS[i][j] - co_hS[i][j]);
    //     if (diff > max_diff_hS) {
    //       max_diff_hS = diff;
    //       max_j_hS = j;
    //     }
    //   }
    // for (size_t a1 = 0; a1 < 4; ++a1)
    //   for (size_t b = 0; b <= a1; ++b)
    //     for (size_t j = 0; j < nd_dhS.get(1, a1, b).size(); ++j) {
    //       const double diff =
    //           std::abs(nd_dhS.get(1, a1, b)[j] - co_dhS.get(1, a1, b)[j]);
    //       if (diff > max_diff_dhS) {
    //         max_diff_dhS = diff;
    //         max_j_dhS = j;
    //       }
    //     }
    // printf(
    //     "Test 3 Bottom: max|hS diff|=%.3e at r=%.4f"
    //     "  max|dhS/dtheta diff|=%.3e at r=%.4f\n",
    //     max_diff_hS, get<0>(x_bot)[max_j_hS], max_diff_dhS,
    //     get<0>(x_bot)[max_j_dhS]);
    // printf("# Test3_Bottom r  abs_hS_diff  abs_dhS_dtheta_diff\n");
    // for (size_t j = 0; j < nd_hS[0].size(); ++j) {
    //   double diff_hS_j = 0., diff_dhS_j = 0.;
    //   for (size_t i = 0; i < nd_hS.size(); ++i)
    //     diff_hS_j = std::max(diff_hS_j, std::abs(nd_hS[i][j] - co_hS[i][j]));
    //   for (size_t a1 = 0; a1 < 4; ++a1)
    //     for (size_t b = 0; b <= a1; ++b)
    //       diff_dhS_j = std::max(diff_dhS_j, std::abs(nd_dhS.get(1, a1, b)[j]
    //       -
    //                                                  co_dhS.get(1, a1,
    //                                                  b)[j]));
    //   printf("%.6e  %.6e  %.6e\n", get<0>(x_bot)[j], diff_hS_j, diff_dhS_j);
    // }

    for (size_t i = 0; i < nd_hS.size(); ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(nd_hS[i], co_hS[i], approx);
    }
    // Only the theta-derivative (index 1) is populated at Bottom face
    // CHECK(max_diff_hS < 1.e-3);
    // CHECK(max_diff_dhS < 1.e-3);
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        CHECK_ITERABLE_CUSTOM_APPROX(nd_dhS.get(1, a1, b), co_dhS.get(1, a1, b),
                                     approx);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Old test: elliptic PDE x singular field = effective source.
  // Not active: NumericData only provides hS at worldtube face points (zero
  // at interior), so the derivative check is trivial and the PDE check does
  // not hold. Kept here for potential future use if analytic hS is restored.
  // -----------------------------------------------------------------------
  // {
  //   const double r_offset = 5.;
  //   const double delta_r = 5.;
  //   const double cos_theta_offset = 0.89;
  //   const double delta_cos_theta = 0.03;
  //   const size_t npoints = 20;
  //   const domain::creators::Rectangle domain_creator{
  //       {{r_offset, cos_theta_offset}},
  //       {{r_offset + delta_r, cos_theta_offset + delta_cos_theta}},
  //       {{0, 0}}, {{npoints, npoints}}, {{false, false}}};
  //   const auto domain = domain_creator.create_domain();
  //   const ElementMap<2, Frame::Inertial> element_map{
  //       ElementId<2>{0}, domain.blocks()[0]};
  //   const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
  //                      Spectral::Quadrature::Gauss};
  //   const auto xi = logical_coordinates(mesh);
  //   const auto x = element_map(xi);
  //   const auto inv_jacobian = element_map.inv_jacobian(xi);
  //
  //   const auto nd = NumericData{h5_file, bh_mass, bh_spin, orbital_radius,
  //                               m_mode, transitions, true, false};
  //   CAPTURE(nd.puncture_position());
  //   const auto background = nd.variables(x, NumericData::background_tags{});
  //   const auto& alpha = get<Tags::Alpha>(background);
  //   const auto& beta = get<Tags::Beta>(background);
  //   const auto& gamma_rstar = get<Tags::GammaRstar>(background);
  //   const auto& gamma_theta = get<Tags::GammaTheta>(background);
  //   const auto vars = nd.variables(x, NumericData::source_tags{}, true);
  //   const auto& singular_field = get<Tags::SingularField>(vars);
  //   const auto& deriv_singular_field =
  //       get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
  //                         Frame::Inertial>>(vars);
  //   const auto& effective_source =
  //       get<::Tags::FixedSource<Tags::MMode>>(vars);
  //
  //   const auto numeric_deriv =
  //       partial_derivative(singular_field, mesh, inv_jacobian);
  //   const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.);
  //   for (size_t i = 0; i < deriv_singular_field.size(); ++i) {
  //     CAPTURE(i);
  //     CHECK_ITERABLE_CUSTOM_APPROX(numeric_deriv[i],
  //                                  deriv_singular_field[i], custom_approx);
  //   }
  //
  //   Variables<tmpl::list<
  //       ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>
  //       fluxes{mesh.number_of_grid_points()};
  //   auto& flux_singular_field =
  //       get<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>(
  //           fluxes);
  //   GrSelfForce::Fluxes::apply(make_not_null(&flux_singular_field), alpha,
  //                              {}, deriv_singular_field);
  //   auto divs = divergence(fluxes, mesh, inv_jacobian);
  //   auto& scalar_eqn =
  //       get<::Tags::div<::Tags::Flux<Tags::MMode, tmpl::size_t<2>,
  //                                   Frame::Inertial>>>(divs);
  //   for (size_t i = 0; i < scalar_eqn.size(); ++i) {
  //     scalar_eqn[i] *= -1.;
  //   }
  //   GrSelfForce::Sources::apply(make_not_null(&scalar_eqn), beta,
  //                               gamma_rstar, gamma_theta, singular_field,
  //                               deriv_singular_field, flux_singular_field);
  //   for (size_t i = 0; i < scalar_eqn.size(); ++i) {
  //     CAPTURE(i);
  //     CHECK_ITERABLE_CUSTOM_APPROX(scalar_eqn[i], -effective_source[i],
  //                                  custom_approx);
  //   }
  // }
}

}  // namespace GrSelfForce::AnalyticData
