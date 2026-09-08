// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrSelfForce.CircularOrbit",
                  "[PointwiseFunctions][Unit]") {
  // This test checks both the self-force equations and the effective source
  // computation in a very robust way: it ensures that the elliptic operator
  // applied to the singular field gives the effective source.

  // Run twice: once in (r*, theta) coordinates (penetrating_horizon=false)
  // and once in (r, cos_theta) coordinates (penetrating_horizon=true).
  auto run_test = [](double coord0_lo, double coord0_hi, double coord1_lo,
                     double coord1_hi,
                     std::optional<std::array<double, 4>> transitions,
                     bool penetrating_horizon, bool reduced_ABC, int version =3) {
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
    INFO("coord0 [" << min(get<0>(x)) << ", " << max(get<0>(x)) << "], coord1 ["
                    << min(get<1>(x)) << ", " << max(get<1>(x)) << "]");

    for (int m_mode_number = 0; m_mode_number < 1; ++m_mode_number) {
      CAPTURE(m_mode_number);
      const auto circular_orbit = CircularOrbit{
          1., 0.9, 20., m_mode_number, transitions, penetrating_horizon, reduced_ABC, version};
      CAPTURE(circular_orbit.puncture_position());
      const auto vars =
          circular_orbit.variables(x, CircularOrbit::source_tags{});
      const auto& singular_field = get<Tags::SingularField>(vars);
      const auto& deriv_singular_field = get<
          ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          vars);
      const auto background =
          circular_orbit.variables(x, CircularOrbit::background_tags{});
      const auto& alpha = get<Tags::Alpha>(background);
      const auto& beta = get<Tags::Beta>(background);
      const auto& gamma_rstar = get<Tags::GammaRstar>(background);
      const auto& gamma_theta = get<Tags::GammaTheta>(background);
      const auto& effective_source =
          get<::Tags::FixedSource<Tags::MMode>>(vars);

      // Check analytic derivative matches numeric derivative
      const auto numeric_deriv_singular_field =
          partial_derivative(singular_field, mesh, inv_jacobian);
      for (size_t i = 0; i < deriv_singular_field.size(); ++i) {
        const size_t dir = i % 2;   // 0 = d/dr, 1 = d/dz
        const size_t comp = i / 2;  // storage index into tnsr::aa
        const auto& an = deriv_singular_field[i];
        const auto& nu = numeric_deriv_singular_field[i];
        double max_err = 0.;
        double max_val = 0.;
        for (size_t p = 0; p < an.size(); ++p) {
          if (std::abs(an[p] - nu[p]) > max_err) max_err = std::abs(an[p] - nu[p]);
          if (std::abs(an[p]) > max_val) max_val = std::abs(an[p]);
        }
        const double rel = max_val > 0. ? max_err / max_val : max_err;
        CAPTURE(i);
        CAPTURE(dir);
        CAPTURE(comp);
        CAPTURE(rel);
        CHECK(rel < 1e-6);
      }

      // Check elliptic operator applied to singular field
      // gives effective source
      Variables<tmpl::list<
          ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>
          fluxes{mesh.number_of_grid_points()};
      auto& flux_singular_field =
          get<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>(
              fluxes);
      GrSelfForce::Fluxes::apply(make_not_null(&flux_singular_field), alpha,
      circular_orbit, {}, deriv_singular_field);
      auto divs = divergence(fluxes, mesh, inv_jacobian);
      auto& scalar_eqn = get<::Tags::div<
          ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>(divs);
      for (size_t i = 0; i < scalar_eqn.size(); ++i) {
        scalar_eqn[i] *= -1.;
      }
      GrSelfForce::Sources::apply(make_not_null(&scalar_eqn), beta, gamma_rstar,
                                  gamma_theta, singular_field,
                                  deriv_singular_field, flux_singular_field);
      for (size_t i = 0; i < scalar_eqn.size(); ++i) {
        // storage index: 0=vv 1=vr 2=vth 3=vph 4=rr 5=rth 6=rph 7=thth 8=thph
        //                9=phph
        const auto& lhs = scalar_eqn[i];       // want lhs == -effective_source[i]
        const auto& rhs = effective_source[i];
        double max_err = 0.;
        double max_val = 0.;
        for (size_t p = 0; p < lhs.size(); ++p) {
          if (std::abs(lhs[p] + rhs[p]) > max_err) {
            max_err = std::abs(lhs[p] + rhs[p]);
          }
          if (std::abs(rhs[p]) > max_val) max_val = std::abs(rhs[p]);
          if (std::abs(lhs[p]) > max_val) max_val = std::abs(lhs[p]);
        }
        const double rel = max_val > 0. ? max_err / max_val : max_err;
        CAPTURE(i);
        CAPTURE(rel);
        CHECK(rel < 1e-6);
      }
    }
  };

//   // penetrating_horizon = false: (r*, theta) coordinates
//   run_test(0., 5., M_PI_2 + M_PI / 8., M_PI_2 + M_PI / 8. + M_PI / 40.,
//            std::nullopt, false);

//   // penetrating_horizon = true: (r, cos_theta) coordinates (version 3), (original ABC) 
//   run_test(5, 10, -0.4, -0.2,
//            std::array<double, 4>{2., 2., 25., 25.}, true, false, 3);

  // penetrating_horizon = true: (r, cos_theta) coordinates (version 3), (original ABC) 
  run_test(5, 10, -0.4, -0.2,
           std::array<double, 4>{2., 2., 25., 25.}, true, true, 3);
}

}  // namespace GrSelfForce::AnalyticData
