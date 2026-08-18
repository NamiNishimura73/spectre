// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Events/ObserveRedshift.hpp"

#include <complex>
#include <cstddef>
#include <optional>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/Math.hpp"

namespace GrSelfForce::Events::detail {

std::optional<std::complex<double>> extract_redshift(
    const Domain<2>& domain, const ElementId<2>& element_id,
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const Mesh<2>& mesh) {
  // Get element-logical coords of puncture
  const auto puncture_position = circular_orbit.puncture_position();
  const auto& block = domain.blocks()[element_id.block_id()];
  const auto block_logical_coords =
      block_logical_coordinates_single_point(puncture_position, block);
  if (not block_logical_coords.has_value()) {
    return std::nullopt;
  }
  const auto puncture_logical_coords =
      element_logical_coordinates(block_logical_coords.value(), element_id);
  if (not puncture_logical_coords.has_value()) {
    return std::nullopt;
  }
  // Interpolate field to puncture position
  const intrp::Irregular<2> interpolator(mesh, puncture_logical_coords.value());
  tnsr::aa<std::complex<double>, 3> field_at_puncture{};
  ComplexDataVector intrp_result(1_st);
  for (size_t i = 0; i < field.size(); ++i) {
    interpolator.interpolate(make_not_null(&intrp_result), field[i]);
    field_at_puncture[i] = intrp_result[0];
  }
  // Calculate redshift
  const double r0 = circular_orbit.orbital_radius();
  const double M = circular_orbit.black_hole_mass();
  const double spin = circular_orbit.black_hole_spin();
  const double a = M * spin;
  const int m_mode = circular_orbit.m_mode_number();
  const double omega = 1. / (a + sqrt(cube(r0) / M));
  const double delta = square(r0) - 2.0 * M * r0 + a * a;
  const double sigma = square(r0);
  const bool penetrating_horizon = circular_orbit.penetrating_horizon();
  tnsr::aa<double, 3> kerr_metric{0.0};
  get<0, 0>(kerr_metric) = -(1.0 - 2.0 * M * r0 / sigma);
  get<0, 3>(kerr_metric) = -2.0 * M * a * r0 / sigma;
  get<1, 1>(kerr_metric) = sigma / delta;
  get<2, 2>(kerr_metric) = sigma;
  get<3, 3>(kerr_metric) =
      (square(r0) + square(a) + 2.0 * M * square(a) * r0 / sigma);
  const auto inv_kerr_metric = determinant_and_inverse(kerr_metric).second;
    tnsr::aa<std::complex<double>, 3> hbar{};
    if (penetrating_horizon) {
      auto hbar_EF = field_at_puncture;
      get<0, 0>(hbar_EF) *= 1.0 / r0;
      get<0, 1>(hbar_EF) *= 1.0 / r0;
      get<1, 1>(hbar_EF) *= 1.0 / r0;
      get<2, 2>(hbar_EF) *= r0;
      get<2, 3>(hbar_EF) *= r0;
      get<3, 3>(hbar_EF) *= r0;

    //   Parallel::printf(
    //       "hbar_EF = {\n"
    //       "  'vv':   complex(%.17e, %.17e),\n"
    //       "  'vr':   complex(%.17e, %.17e),\n"
    //       "  'vth':  complex(%.17e, %.17e),\n"
    //       "  'vph':  complex(%.17e, %.17e),\n"
    //       "  'rr':   complex(%.17e, %.17e),\n"
    //       "  'rth':  complex(%.17e, %.17e),\n"
    //       "  'rph':  complex(%.17e, %.17e),\n"
    //       "  'thth': complex(%.17e, %.17e),\n"
    //       "  'thph': complex(%.17e, %.17e),\n"
    //       "  'phph': complex(%.17e, %.17e),\n"
    //       "}\n",
    //       get<0, 0>(hbar_EF).real(), get<0, 0>(hbar_EF).imag(),
    //       get<0, 1>(hbar_EF).real(), get<0, 1>(hbar_EF).imag(),
    //       get<0, 2>(hbar_EF).real(), get<0, 2>(hbar_EF).imag(),
    //       get<0, 3>(hbar_EF).real(), get<0, 3>(hbar_EF).imag(),
    //       get<1, 1>(hbar_EF).real(), get<1, 1>(hbar_EF).imag(),
    //       get<1, 2>(hbar_EF).real(), get<1, 2>(hbar_EF).imag(),
    //       get<1, 3>(hbar_EF).real(), get<1, 3>(hbar_EF).imag(),
    //       get<2, 2>(hbar_EF).real(), get<2, 2>(hbar_EF).imag(),
    //       get<2, 3>(hbar_EF).real(), get<2, 3>(hbar_EF).imag(),
    //       get<3, 3>(hbar_EF).real(), get<3, 3>(hbar_EF).imag());

      const double r_plus = M + sqrt(square(M) - square(a));
      const double r_minus = M - sqrt(square(M) - square(a));
      const double r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
          r0 - r_plus, M, spin);
      const double chi = a / (r_plus - r_minus) *
                          log((r0 - r_plus) / (r0 - r_minus));
      // const double mode_omega = m_mode * omega;
      const std::complex<double> ef_to_bl_phase = std::exp(
          std::complex<double>(0.0, m_mode * chi));
      for (size_t i = 0; i < hbar_EF.size(); ++i) {
        hbar_EF[i] *= ef_to_bl_phase;
      }

      const double factor_F = (r0*r0 + a*a)/delta;
      const double factor_G = a/delta;
      // hBL_rr = F^2 hEF_vv + 2F hEF_vr + hEF_rr + 2FG hEF_v\phi + 2G hEF_r\phi + G^2 hEF_\phi\phi
      get<1, 1>(hbar) = get<0, 0>(hbar_EF)*factor_F*factor_F +
                        2*factor_F*get<0, 1>(hbar_EF) + get<1, 1>(hbar_EF) +
                        2*factor_F*factor_G*get<0, 3>(hbar_EF) +
                        2*factor_G*get<1, 3>(hbar_EF) +
                        factor_G*factor_G*get<3, 3>(hbar_EF);
      // hBL_tr = F hEF_vv + hEF_vr + G hEF_v\phi
      get<0, 1>(hbar) = get<0, 0>(hbar_EF)*factor_F + get<0, 1>(hbar_EF) +
                        factor_G*get<0, 3>(hbar_EF);
      // hBL_r\theta = F hEF_v\theta + hEF_r\theta + G hEF_\phi\theta
      get<1, 2>(hbar) = get<0, 2>(hbar_EF)*factor_F + get<1, 2>(hbar_EF) +
                        factor_G*get<2, 3>(hbar_EF);
      // hBL_r\phi = F hEF_v\phi + hEF_r\phi + G hEF_\phi\phi
      get<1, 3>(hbar) = get<0, 3>(hbar_EF)*factor_F + get<1, 3>(hbar_EF) +
                        factor_G*get<3, 3>(hbar_EF);

      // rest are the same
      get<0, 0>(hbar) = get<0, 0>(hbar_EF);
      get<0, 2>(hbar) = get<0, 2>(hbar_EF);
      get<0, 3>(hbar) = get<0, 3>(hbar_EF);
      get<2, 2>(hbar) = get<2, 2>(hbar_EF);
      get<2, 3>(hbar) = get<2, 3>(hbar_EF);
      get<3, 3>(hbar) = get<3, 3>(hbar_EF);
      
    //   Parallel::printf(
    //       "hBL = {\n"
    //       "  'tt':   complex(%.17e, %.17e),\n"
    //       "  'tr':   complex(%.17e, %.17e),\n"
    //       "  'tth':  complex(%.17e, %.17e),\n"
    //       "  'tph':  complex(%.17e, %.17e),\n"
    //       "  'rr':   complex(%.17e, %.17e),\n"
    //       "  'rth':  complex(%.17e, %.17e),\n"
    //       "  'rph':  complex(%.17e, %.17e),\n"
    //       "  'thth': complex(%.17e, %.17e),\n"
    //       "  'thph': complex(%.17e, %.17e),\n"
    //       "  'phph': complex(%.17e, %.17e),\n"
    //       "}\n",
    //       get<0, 0>(hbar).real(), get<0, 0>(hbar).imag(),
    //       get<0, 1>(hbar).real(), get<0, 1>(hbar).imag(),
    //       get<0, 2>(hbar).real(), get<0, 2>(hbar).imag(),
    //       get<0, 3>(hbar).real(), get<0, 3>(hbar).imag(),
    //       get<1, 1>(hbar).real(), get<1, 1>(hbar).imag(),
    //       get<1, 2>(hbar).real(), get<1, 2>(hbar).imag(),
    //       get<1, 3>(hbar).real(), get<1, 3>(hbar).imag(),
    //       get<2, 2>(hbar).real(), get<2, 2>(hbar).imag(),
    //       get<2, 3>(hbar).real(), get<2, 3>(hbar).imag(),
    //       get<3, 3>(hbar).real(), get<3, 3>(hbar).imag());

    } else {
      hbar = field_at_puncture;
      get<0, 0>(hbar) *= 1.0 / r0;
      get<0, 1>(hbar) *= r0 / delta;
      get<1, 1>(hbar) *= cube(r0) / square(delta);
      get<1, 2>(hbar) *= square(r0) / delta;
      get<1, 3>(hbar) *= square(r0) / delta;
      get<2, 2>(hbar) *= r0;
      get<2, 3>(hbar) *= r0;
      get<3, 3>(hbar) *= r0;
    }
  const auto trace_hbar =
      tenex::evaluate(hbar(ti::a, ti::b) * inv_kerr_metric(ti::A, ti::B));
  const auto h = tenex::evaluate<ti::a, ti::b>(
      hbar(ti::a, ti::b) - 0.5 * trace_hbar() * kerr_metric(ti::a, ti::b));
//   Parallel::printf(
//       "trace_hbar = complex(%.17e, %.17e)\n"
//       "h_final = {\n"
//       "  'tt':   complex(%.17e, %.17e),\n"
//       "  'tr':   complex(%.17e, %.17e),\n"
//       "  'tth':  complex(%.17e, %.17e),\n"
//       "  'tph':  complex(%.17e, %.17e),\n"
//       "  'rr':   complex(%.17e, %.17e),\n"
//       "  'rth':  complex(%.17e, %.17e),\n"
//       "  'rph':  complex(%.17e, %.17e),\n"
//       "  'thth': complex(%.17e, %.17e),\n"
//       "  'thph': complex(%.17e, %.17e),\n"
//       "  'phph': complex(%.17e, %.17e),\n"
//       "}\n",
//       get(trace_hbar).real(), get(trace_hbar).imag(),
//       get<0, 0>(h).real(), get<0, 0>(h).imag(),
//       get<0, 1>(h).real(), get<0, 1>(h).imag(),
//       get<0, 2>(h).real(), get<0, 2>(h).imag(),
//       get<0, 3>(h).real(), get<0, 3>(h).imag(),
//       get<1, 1>(h).real(), get<1, 1>(h).imag(),
//       get<1, 2>(h).real(), get<1, 2>(h).imag(),
//       get<1, 3>(h).real(), get<1, 3>(h).imag(),
//       get<2, 2>(h).real(), get<2, 2>(h).imag(),
//       get<2, 3>(h).real(), get<2, 3>(h).imag(),
//       get<3, 3>(h).real(), get<3, 3>(h).imag());

  tnsr::A<double, 3> u_particle{0.0};
  const double u_mag =
      sqrt(cube(r0) / M - 3.0 * square(r0) + 2.0 * a * sqrt(cube(r0) / M));
  get<0>(u_particle) = 1.0 / u_mag / omega;
  get<3>(u_particle) = 1.0 / u_mag;
  const auto h_uu = tenex::evaluate(
      h(ti::a, ti::b) * u_particle(ti::A) * u_particle(ti::B));
//   Parallel::printf(
//       "u_t = %.17e\nu_phi = %.17e\n"
//       "h_uu = complex(%.17e, %.17e)\n"
//       "redshift_returned = %.17e\n",
//       get<0>(u_particle), get<3>(u_particle),
//       get(h_uu).real(), get(h_uu).imag(), 2.0 * get(h_uu).real());

  // Factor of 2 accounts for the m,-m pairing: h_ab u^a u^b is the
  // single-m coefficient; the physical (real) contribution from mode m
  // is 2*Re[h_ab^(m) u^a u^b], confirmed against Ben's reference data.
  return 2.0 * get(h_uu);
}

}  // namespace GrSelfForce::Events::detail
