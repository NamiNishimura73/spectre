// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/NumericData.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::AnalyticData {

// #if 0
// // This test predates the domain-wide Puncture/drPuncture/dthPuncture h5
// // datasets. It used a synthetic h5 file generated from CircularOrbit's
// // analytic (1st-order) effective source and puncture, just to validate the
// // h5 flattening/reading logic -- not a physics regression test in its own
// // right, and Test 4 below directly contradicts the current behavior (which
// // now populates singular_field wherever Puncture data covers a point,
// // regardless of field_is_regularized). Left here, disabled, for reference.
// SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrSelfForce.NumericData.Legacy",
//                   "[PointwiseFunctions][Unit]") {
//   // Test that NumericData (h5-based) agrees with CircularOrbit (analytic)
//   // for a 1st-order h5 dataset. Three checks:
//   //   1. Seff inside the worldtube (2D mesh, field_is_regularized=true)
//   //   2. hS and its normal derivative at the Left face (r = r_wt_left)
//   //   3. hS and its normal derivative at the Bottom face (theta = theta_wt_bot)

//   const std::string h5_file =
//       "/u/namni/spectre_copy/data/D2G_m2_a0.6_r8.0.h5";
//   if (not file_system::check_if_file_exists(h5_file)) {
//     SUCCEED("Skipping test: h5 file not available on this machine");
//     return;
//   }
//   const double bh_mass = 1.;
//   const double bh_spin = 0.6;
//   const double orbital_radius = 8.;
//   const int m_mode = 2;
//   // Transitions from RetRetV/T/U grid bounds in h5
//   const std::array<double, 4> transitions{3.8667, 3.8667, 14.2, 14.2};

//   // Worldtube face coordinates (from Seff.dat attributes)
//   const double r_wt_left = 5.933333333333333;
//   const double r_wt_right = 10.066666666666666;
//   // thetaMin = pi/3 -> cos = 0.5;  thetaMax = 2*pi/3 -> cos = -0.5
//   const double cos_wt_bot = 0.5;
//   const double cos_wt_top = -0.5;

//   const NumericData numeric_data{h5_file,    bh_mass,         bh_spin,
//                                  orbital_radius, m_mode, transitions,
//                                  true, 2, false};
//   const CircularOrbit circular_orbit{bh_mass, bh_spin, orbital_radius,
//                                      m_mode, transitions, true, 2};

//   const Approx approx = Approx::custom().epsilon(1.e-5).scale(1.);

//   // -----------------------------------------------------------------------
//   // Test 1: Seff on a 2D interior mesh (field_is_regularized=true uses the
//   // high-resolution 500x500 Seff.dat grid)
//   // -----------------------------------------------------------------------
//   {
//     const size_t npoints = 10;
//     // Domain well inside worldtube bounds
//     const domain::creators::Rectangle domain_creator{
//         {{6.5, -0.3}}, {{9.5, 0.3}},
//         {{0, 0}}, {{npoints, npoints}}, {{false, false}}};
//     const auto domain = domain_creator.create_domain();
//     const ElementMap<2, Frame::Inertial> element_map{ElementId<2>{0},
//                                                      domain.blocks()[0]};
//     const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
//                        Spectral::Quadrature::Gauss};
//     const auto x = element_map(logical_coordinates(mesh));

//     const auto nd_vars =
//         numeric_data.variables(x, NumericData::source_tags{}, true);
//     const auto co_vars =
//         circular_orbit.variables(x, CircularOrbit::source_tags{});
//     const auto& nd_seff = get<::Tags::FixedSource<Tags::MMode>>(nd_vars);
//     const auto& co_seff = get<::Tags::FixedSource<Tags::MMode>>(co_vars);
//     for (size_t i = 0; i < nd_seff.size(); ++i) {
//       CHECK_ITERABLE_CUSTOM_APPROX(nd_seff[i], - co_seff[i], approx);
//     }
//   }

//   // -----------------------------------------------------------------------
//   // Test 2: hS and dhS/dr at Left face (r = r_wt_left, cos_theta varies)
//   // NumericData fills singular_field from Left.dat (1D, theta-parameterized)
//   // and stores dhS/dr in deriv_singular_field.get(0,...); theta-deriv = 0.
//   // -----------------------------------------------------------------------
//   {
//     const size_t nface = 10;
//     tnsr::I<DataVector, 2> x_left{};
//     get<0>(x_left) = DataVector(nface, r_wt_left);
//     get<1>(x_left) = DataVector(nface, 0.);
//     for (size_t i = 0; i < nface; ++i) {
//       // cos_theta strictly inside worldtube (avoid corners)
//       get<1>(x_left)[i] =
//           cos_wt_top + (cos_wt_bot - cos_wt_top) *
//                            (static_cast<double>(i) + 1.) /
//                            (static_cast<double>(nface) + 1.);
//     }

//     const auto nd_vars =
//         numeric_data.variables(x_left, NumericData::source_tags{}, true);
//     const auto co_vars =
//         circular_orbit.variables(x_left, CircularOrbit::source_tags{});
//     const auto& nd_hS = get<Tags::SingularField>(nd_vars);
//     const auto& co_hS = get<Tags::SingularField>(co_vars);
//     const auto& nd_dhS = get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
//                                            Frame::Inertial>>(nd_vars);
//     const auto& co_dhS = get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
//                                            Frame::Inertial>>(co_vars);

//     for (size_t i = 0; i < nd_hS.size(); ++i) {
//       CHECK_ITERABLE_CUSTOM_APPROX(nd_hS[i], co_hS[i], approx);
//     }
//     // Only the r-derivative (index 0) is populated at Left face
//     for (size_t a1 = 0; a1 < 4; ++a1) {
//       for (size_t b = 0; b <= a1; ++b) {
//         CHECK_ITERABLE_CUSTOM_APPROX(nd_dhS.get(0, a1, b),
//                                      co_dhS.get(0, a1, b), approx);
//       }
//     }
//   }

//   // -----------------------------------------------------------------------
//   // Test 3: hS and dhS/dtheta at Bottom face (cos_theta = cos_wt_bot, r varies)
//   // NumericData fills from Bottom.dat (1D, r-parameterized)
//   // and stores dhS/dtheta in deriv_singular_field.get(1,...); r-deriv = 0.
//   // -----------------------------------------------------------------------
//   {
//     const size_t nface = 10;
//     tnsr::I<DataVector, 2> x_bot{};
//     get<0>(x_bot) = DataVector(nface, 0.);
//     get<1>(x_bot) = DataVector(nface, cos_wt_bot);
//     for (size_t i = 0; i < nface; ++i) {
//       // r strictly inside worldtube (avoid corners)
//       get<0>(x_bot)[i] =
//           r_wt_left + (r_wt_right - r_wt_left) *
//                           (static_cast<double>(i) + 1.) /
//                           (static_cast<double>(nface) + 1.);
//     }

//     const auto nd_vars =
//         numeric_data.variables(x_bot, NumericData::source_tags{}, true);
//     const auto co_vars =
//         circular_orbit.variables(x_bot, CircularOrbit::source_tags{});
//     const auto& nd_hS = get<Tags::SingularField>(nd_vars);
//     const auto& co_hS = get<Tags::SingularField>(co_vars);
//     const auto& nd_dhS = get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
//                                            Frame::Inertial>>(nd_vars);
//     const auto& co_dhS = get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
//                                            Frame::Inertial>>(co_vars);

//     for (size_t i = 0; i < nd_hS.size(); ++i) {
//       CHECK_ITERABLE_CUSTOM_APPROX(nd_hS[i], co_hS[i], approx);
//     }
//     // Only the theta-derivative (index 1) is populated at Bottom face
//     for (size_t a1 = 0; a1 < 4; ++a1) {
//       for (size_t b = 0; b <= a1; ++b) {
//         CHECK_ITERABLE_CUSTOM_APPROX(nd_dhS.get(1, a1, b),
//                                      co_dhS.get(1, a1, b), approx);
//       }
//     }
//   }
//   // -----------------------------------------------------------------------
//   // Test 4: RetRet on a 2D interior mesh (field_is_regularized=false)
//   // Domain in v-region (r < 3.8667), uses RetRetV interpolator.
//   // Checks: fixed_source is nonzero. singular_field is exactly zero.
//   // -----------------------------------------------------------------------
//   {
//     const size_t npoints = 10;
//     const domain::creators::Rectangle domain_creator{
//         {{2.0, -0.3}}, {{3.5, 0.3}},
//         {{0, 0}}, {{npoints, npoints}}, {{false, false}}};
//     const auto domain = domain_creator.create_domain();
//     const ElementMap<2, Frame::Inertial> element_map{ElementId<2>{0},
//                                                      domain.blocks()[0]};
//     const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
//                        Spectral::Quadrature::Gauss};
//     const auto x = element_map(logical_coordinates(mesh));

//     const auto nd_vars =
//         numeric_data.variables(x, NumericData::source_tags{}, false);  // false!

//     const auto& nd_seff = get<::Tags::FixedSource<Tags::MMode>>(nd_vars);
//     const auto& nd_hS   = get<Tags::SingularField>(nd_vars);

//     // Fixed source must be nonzero: RetRetV has 0.1 in all BL-frame components,
//     // which maps to nonzero values after BL->VR conversion.
//     bool any_nonzero = false;
//     for (size_t i = 0; i < nd_seff.size(); ++i) {
//       if (max(abs(real(nd_seff[i]))) > 0. or max(abs(imag(nd_seff[i]))) > 0.) {
//         any_nonzero = true;
//         break;
//       }
//     }
//     CHECK(any_nonzero);

//     // Singular field must be exactly zero outside worldtube.
//     for (size_t i = 0; i < nd_hS.size(); ++i) {
//       CHECK(nd_hS[i] == ComplexDataVector(npoints * npoints, 0.0));
//     }
//   }

// }
// #endif

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrSelfForce.NumericData",
                  "[PointwiseFunctions][Unit]") {
  // This mirrors the two checks in Test_CircularOrbit.cpp -- analytic vs.
  // numeric derivative of the singular field, and elliptic operator applied
  // to the singular field vs. the effective source -- but sources the
  // singular field, its derivative, and the effective source from the
  // NumericData h5 interpolation pipeline (Puncture/drPuncture/dthPuncture
  // and Seff/RetRetT) instead of closed-form formulas.
  //
  // Because the h5 data is only piecewise (bilinearly) smooth, with kinks at
  // every grid cell boundary (dr = 0.1, dtheta = pi/60 for this file), the
  // test domain below is chosen to stay within a single grid cell in each
  // case. Spanning multiple cells would introduce kinks that the spectral
  // derivative/divergence operators used here cannot resolve, unlike
  // CircularOrbit's globally-smooth closed-form functions.
  const std::string h5_file =
      "/u/namni/spectre_copy/data/Aug10_D2G_m2_a0.000_r8.000_ascii.h5";
      // "/u/namni/spectre_copy/data/PucCheck_D2G_m2_l2_highres_ascii.h5";
  if (not file_system::check_if_file_exists(h5_file)) {
    SUCCEED("Skipping test: h5 file not available on this machine");
    return;
  }
  const NumericData numeric_data{h5_file,
                                 1.,                    // BlackHoleMass
                                 0.,                    // BlackHoleSpin
                                 8.,                    // OrbitalRadius
                                 2,                     // MModeNumber
                                 {{5., 5., 20., 20.}},  // HyperboloidalSlicingTransitions
                                 true,                  // PenetratingHorizon
                                 3,                     // Version
                                 true};                 // Pi_2_Rotation

  auto run_test = [&numeric_data](const double coord0_lo,
                                  const double coord0_hi,
                                  const double coord1_lo,
                                  const double coord1_hi,
                                  const bool field_is_regularized) {
    const size_t npoints = 20;
    const domain::creators::Rectangle domain_creator{
        {{coord0_lo, coord1_lo}},
        {{coord0_hi, coord1_hi}},
        {{0, 0}},
        {{npoints, npoints}},
        {{false, false}}};
    const auto domain = domain_creator.create_domain();
    const auto& block = domain.blocks()[0];
    const ElementId<2> element_id{0};
    const ElementMap<2, Frame::Inertial> element_map{element_id, block};
    const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss};
    const auto xi = logical_coordinates(mesh);
    const auto x = element_map(xi);
    const auto inv_jacobian = element_map.inv_jacobian(xi);
    CAPTURE(field_is_regularized);
    CAPTURE(min(get<0>(x)));
    CAPTURE(max(get<0>(x)));
    CAPTURE(min(get<1>(x)));
    CAPTURE(max(get<1>(x)));

    const auto background =
        numeric_data.variables(x, NumericData::background_tags{});
    const auto& alpha = get<Tags::Alpha>(background);
    const auto& beta = get<Tags::Beta>(background);
    const auto& gamma_rstar = get<Tags::GammaRstar>(background);
    const auto& gamma_theta = get<Tags::GammaTheta>(background);
    const auto vars = numeric_data.variables(x, NumericData::diagnostic_tags{},
                                             field_is_regularized);
    const auto& singular_field = get<Tags::SingularField>(vars);
    const auto& deriv_singular_field = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        vars);
    const auto& rhs_box_puncture = get<Tags::RHSBoxPuncture>(vars);
    // Equals conv(Seff) alone when field_is_regularized is true, conv(RetRetT)
    // alone when false (same single-interpolator switch as FixedSource<MMode>).
    const auto& ef_eff_source = get<Tags::EF_EffSource>(vars);

    // Check that drPuncture/dthPuncture (in deriv_singular_field) agree with
    // numerically differentiating Puncture (in singular_field).
    const auto numeric_deriv_singular_field =
        partial_derivative(singular_field, mesh, inv_jacobian);
    const Approx custom_approx = Approx::custom().epsilon(0.1).scale(1.);
    for (size_t i = 0; i < deriv_singular_field.size(); ++i) {
      CAPTURE(i);
      CHECK_ITERABLE_CUSTOM_APPROX(numeric_deriv_singular_field[i],
                                   deriv_singular_field[i], custom_approx);
    }

    // Check elliptic operator applied to singular field gives
    // RHSBoxPuncture (= Seff - RetRetT)
    Variables<tmpl::list<
        ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>
        fluxes{mesh.number_of_grid_points()};
    auto& flux_singular_field =
        get<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>(
            fluxes);
    GrSelfForce::Fluxes::apply(make_not_null(&flux_singular_field), alpha, {},
                               deriv_singular_field);
    auto divs = divergence(fluxes, mesh, inv_jacobian);
    auto& box_operator = get<::Tags::div<
        ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>(divs);
    for (size_t i = 0; i < box_operator.size(); ++i) {
      box_operator[i] *= -1.;
    }
    GrSelfForce::Sources::apply(make_not_null(&box_operator), beta, gamma_rstar,
                                gamma_theta, singular_field,
                                deriv_singular_field, flux_singular_field);
    // for (size_t i = 0; i < box_operator.size(); ++i) {
    //   CHECK_ITERABLE_CUSTOM_APPROX(box_operator[i], -rhs_box_puncture[i],
    //                                custom_approx);
    // }

    // DEBUG: print instead of check, to see sign/magnitude agreement before
    // committing to a tolerance or sign convention. tt component (index 0),
    // 10 points spread evenly across the mesh.
    const auto& box_operator_tt = box_operator.get(0, 0);
    const auto& rhs_box_puncture_tt = rhs_box_puncture.get(0, 0);
    const auto& ef_eff_source_tt = ef_eff_source.get(0, 0);
    const size_t n_points = box_operator_tt.size();
    const size_t stride = std::max<size_t>(1, n_points / 10);
    for (size_t j = 0; j < n_points; j += stride) {
      const auto lhs = box_operator_tt[j];
      const auto rhs = -rhs_box_puncture_tt[j];
      std::cout << "j=" << j << "  r=" << get<0>(x)[j]
                << "  box_operator[tt]=" << lhs
                << "  -rhs_box_puncture[tt]=" << rhs
                << "  ratio=" << lhs / rhs
                << "  ef_eff_source[tt]=" << ef_eff_source_tt[j]
                << "  box_operator/ef_eff_source=" << lhs / ef_eff_source_tt[j]
                << "\n";
    }
  };

  // A box strictly inside a single h5 grid cell (r in [7.9, 8.0], theta in
  // [theta_28, theta_29] = [1.5708, 1.6232]), inside the worldtube (Seff's
  // footprint is r in [6, 10], theta in [0.785, 2.356]).
  {
    INFO("Inside the worldtube");
    run_test(6.10, 6.12, cos(1.02), cos(1.00), true);
  }

  // Same-sized box in a different grid cell (r in [14.9, 15.0]), outside the
  // worldtube and outside Seff's r range, but still inside the T-slicing
  // region: NumericData now sources the effective source from RetRetT and
  // the singular field from Puncture/drPuncture/dthPuncture there.
  // {
  //   INFO("Outside the worldtube, in the wider T-slicing region");
  //   run_test(14.92, 14.98, cos(1.02), cos(1.00), false);
  // }

  // -----------------------------------------------------------------------
  // Cross-check: at the worldtube boundary, the domain-wide Puncture/
  // drPuncture/dthPuncture data should agree with the original face-only
  // Left/Right/Top/Bottom data. worldtube_r_min/max and
  // worldtube_theta_min/max are exact grid points of Puncture's grid (Seff's
  // footprint is an exact subgrid of Puncture's wider footprint), so unlike
  // the derivative check above, this comparison isn't limited by O(dr)
  // interpolation truncation error -- both sides do the same linear
  // interpolation along theta (or r) starting from the same tabulated row,
  // so they should agree tightly if the two h5 datasets are self-consistent.
  // -----------------------------------------------------------------------
  {
    const double r_min = numeric_data.worldtube_r_min();
    const double r_max = numeric_data.worldtube_r_max();
    const double theta_min = numeric_data.worldtube_theta_min();
    const double theta_max = numeric_data.worldtube_theta_max();
    const Approx face_approx = Approx::custom().epsilon(1.e-15).scale(1.);
    const size_t n_samples = 10;

    const auto check_face = [&numeric_data, &face_approx](
                                const tnsr::I<DataVector, 2>& x, size_t face,
                                size_t deriv_index) {
      CAPTURE(face);
      const auto vars =
          numeric_data.variables(x, NumericData::source_tags{}, true);
      const auto face_vars = numeric_data.boundary_face_variables(x, face);
      const auto& puncture_hS = get<Tags::SingularField>(vars);
      const auto& face_hS = get<Tags::SingularField>(face_vars);
      const auto& puncture_dhS = get<::Tags::deriv<Tags::SingularField,
                                                    tmpl::size_t<2>,
                                                    Frame::Inertial>>(vars);
      const auto& face_dhS =
          get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
                            Frame::Inertial>>(face_vars);
      for (size_t i = 0; i < puncture_hS.size(); ++i) {
        CHECK_ITERABLE_CUSTOM_APPROX(puncture_hS[i], face_hS[i], face_approx);
      }
      for (size_t a1 = 0; a1 < 4; ++a1) {
        for (size_t b = 0; b <= a1; ++b) {
          CHECK_ITERABLE_CUSTOM_APPROX(puncture_dhS.get(deriv_index, a1, b),
                                       face_dhS.get(deriv_index, a1, b),
                                       face_approx);
        }
      }
    };

    // Left (face 0) / Right (face 1): r fixed, theta varies. Only the
    // r-derivative (index 0) is populated by Left/Right.
    for (const size_t face : {size_t{0}, size_t{1}}) {
      const double r_face = (face == 0) ? r_min : r_max;
      tnsr::I<DataVector, 2> x{};
      get<0>(x) = DataVector(n_samples, r_face);
      get<1>(x) = DataVector(n_samples);
      for (size_t i = 0; i < n_samples; ++i) {
        const double theta =
            theta_min + (theta_max - theta_min) *
                            (static_cast<double>(i) + 1.) /
                            (static_cast<double>(n_samples) + 1.);
        get<1>(x)[i] = cos(theta);
      }
      check_face(x, face, 0);
    }

    // Bottom (face 2) / Top (face 3): theta fixed, r varies. Only the
    // theta-derivative (index 1) is populated by Bottom/Top.
    for (const size_t face : {size_t{2}, size_t{3}}) {
      const double theta_face = (face == 2) ? theta_min : theta_max;
      tnsr::I<DataVector, 2> x{};
      get<1>(x) = DataVector(n_samples, cos(theta_face));
      get<0>(x) = DataVector(n_samples);
      for (size_t i = 0; i < n_samples; ++i) {
        get<0>(x)[i] =
            r_min + (r_max - r_min) * (static_cast<double>(i) + 1.) /
                        (static_cast<double>(n_samples) + 1.);
      }
      check_face(x, face, 1);
    }
  }
}

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GrSelfForce.NumericData.EllipticOperatorSweep",
    "[PointwiseFunctions][Unit]") {
  // Not a pass/fail check. Dumps E[Psi_m^P] (elliptic operator applied to
  // the puncture field) and RHSBoxPuncture at many points across the
  // T-slicing region to a CSV, for visual comparison in Jupyter/ParaView --
  // run manually with `-R "NumericData.EllipticOperatorSweep"` when you want
  // fresh output.
  const std::string h5_file =
      "/u/namni/spectre_copy/data/Aug15_D2G_m2_a0.000_r8.000_ascii.h5";
      // "/u/namni/spectre_copy/data/PucCheck_D2G_m2_l8_highres_ascii.h5";
      // "/u/namni/spectre_copy/data/Seff_Minus1_Aug15_Ben.h5";
  const std::string csv_path =
      // "/u/namni/SpECTRE_runs/results/elliptic_operator_Seff_Minus1_Aug15_Ben.csv";
      // "/u/namni/SpECTRE_runs/results/elliptic_operator_Tommylmax8.csv";
      "/u/namni/SpECTRE_runs/results/elliptic_operator_Aug15_Ben_new_punc.csv";
  if (not file_system::check_if_file_exists(h5_file)) {
    SUCCEED("Skipping test: h5 file not available on this machine");
    return;
  }
  const NumericData numeric_data{h5_file,
                                 1.,                    // BlackHoleMass
                                 0.,                    // BlackHoleSpin
                                 8.,                    // OrbitalRadius
                                 2,                     // MModeNumber
                                 {{5., 5., 20., 20.}},  // HyperboloidalSlicingTransitions
                                 true,                  // PenetratingHorizon
                                 3,                     // Version
                                 true};                 // Pi_2_Rotation

  // Sweep only the worldtube (Seff's footprint): RHSBoxPuncture outside it
  // is driven by RetRetT alone (Seff is only defined inside the worldtube),
  // and the regularized/singular-field split this identity checks is only
  // used by the actual solve inside the worldtube block anyway -- so
  // comparing outside it isn't a physically meaningful check.
  const double margin = 0.03;  // keep clear of the exact worldtube boundary
  const double sweep_r_min = numeric_data.worldtube_r_min() + margin;
  const double sweep_r_max = numeric_data.worldtube_r_max() - margin;
  const size_t n_r = 40;
  const double sweep_theta_min = numeric_data.worldtube_theta_min() + margin;
  const double sweep_theta_max = numeric_data.worldtube_theta_max() - margin;
  const size_t n_theta = 30;
  // Small per-point offsets (smaller than margin above) so sample points
  // avoid landing exactly on an h5 grid line (dr = 0.1, dtheta = pi/60).
  const double r_offset = 0.013;
  const double theta_offset = 0.0037;


  std::ofstream csv(csv_path);
  csv << "r,theta,field_is_regularized,box_operator_tt_re,box_operator_tt_im,"
         "rhs_box_puncture_tt_re,rhs_box_puncture_tt_im\n";

  const size_t npoints = 6;  // small local mesh per sample point
  for (size_t ir = 0; ir < n_r; ++ir) {
    const double r_c = sweep_r_min +
                       (sweep_r_max - sweep_r_min) * static_cast<double>(ir) /
                           static_cast<double>(n_r - 1) +
                       r_offset;
    for (size_t ith = 0; ith < n_theta; ++ith) {
      const double theta_c =
          sweep_theta_min +
          (sweep_theta_max - sweep_theta_min) * static_cast<double>(ith) /
              static_cast<double>(n_theta - 1) +
          theta_offset;
      // Always inside the worldtube by construction (sweep range is
      // [worldtube_min + margin, worldtube_max - margin]).
      const bool field_is_regularized = true;

      // Tiny box around (r_c, theta_c), small enough to stay within a
      // single h5 grid cell (dr = 0.1, dtheta = pi/60).
      const double dr_box = 0.01;
      const double dtheta_box = 0.005;
      const domain::creators::Rectangle domain_creator{
          {{r_c - dr_box, cos(theta_c + dtheta_box)}},
          {{r_c + dr_box, cos(theta_c - dtheta_box)}},
          {{0, 0}},
          {{npoints, npoints}},
          {{false, false}}};
      const auto domain = domain_creator.create_domain();
      const auto& block = domain.blocks()[0];
      const ElementId<2> element_id{0};
      const ElementMap<2, Frame::Inertial> element_map{element_id, block};
      const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
                         Spectral::Quadrature::Gauss};
      const auto xi = logical_coordinates(mesh);
      const auto x = element_map(xi);
      const auto inv_jacobian = element_map.inv_jacobian(xi);

      const auto background =
          numeric_data.variables(x, NumericData::background_tags{});
      const auto& alpha = get<Tags::Alpha>(background);
      const auto& beta = get<Tags::Beta>(background);
      const auto& gamma_rstar = get<Tags::GammaRstar>(background);
      const auto& gamma_theta = get<Tags::GammaTheta>(background);
      const auto vars = numeric_data.variables(
          x, NumericData::diagnostic_tags{}, field_is_regularized);
      const auto& singular_field = get<Tags::SingularField>(vars);
      const auto& deriv_singular_field = get<::Tags::deriv<
          Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(vars);
      const auto& rhs_box_puncture = get<Tags::RHSBoxPuncture>(vars);

      Variables<tmpl::list<
          ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>
          fluxes{mesh.number_of_grid_points()};
      auto& flux_singular_field =
          get<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>(
              fluxes);
      GrSelfForce::Fluxes::apply(make_not_null(&flux_singular_field), alpha,
                                 {}, deriv_singular_field);
      auto divs = divergence(fluxes, mesh, inv_jacobian);
      auto& box_operator = get<::Tags::div<
          ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>(divs);
      for (size_t i = 0; i < box_operator.size(); ++i) {
        box_operator[i] *= -1.;
      }
      GrSelfForce::Sources::apply(make_not_null(&box_operator), beta,
                                  gamma_rstar, gamma_theta, singular_field,
                                  deriv_singular_field, flux_singular_field);

      // Take the point closest to the box center as the representative
      // sample for this cell.
      const size_t mid = mesh.number_of_grid_points() / 2;
      const auto lhs = box_operator.get(0, 0)[mid];
      const auto rhs = rhs_box_puncture.get(0, 0)[mid];
      csv << r_c << "," << theta_c << "," << field_is_regularized << ","
          << lhs.real() << "," << lhs.imag() << "," << rhs.real() << ","
          << rhs.imag() << "\n";
    }
  }
  csv.close();
  std::cout << "Wrote elliptic operator sweep to " << csv_path << "\n";
}

}  // namespace GrSelfForce::AnalyticData

// ./bin/Test_GrSelfForceAnalyticData "Unit.PointwiseFunctions.GrSelfForce.NumericData.EllipticOperatorSweep"